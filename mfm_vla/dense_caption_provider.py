"""Dynamic Dense-Caption providers for `OpenPiVLAExecutor.prompt_builder`.

At every 5-step replanning boundary, the provider constructs a fresh 10-step Dense
Caption (`scene`, `subgoal`, `geometry`, `next 10 steps`) from:
  - current end-effector position `obs.eef_pos` and gripper opening `obs.gripper_qpos`
  - estimated 3D object/target positions (either from privileged simulator state for
    zero-GPT upper-bound evaluation, or from a single GPT visual grounding call +
    camera depth deprojection at episode start).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from dense_caption import (
    _distinguish_identical_names,
    causal_chunk_motion,
    clean_name,
    format_delta_3d,
    table_sector,
)
from visual_prompt import goal_targets


@dataclass
class OracleDenseCaptioner:
    """Zero-GPT closed-loop Dense Captioner for upper-bound / dev diagnostics.

    Adds configurable Gaussian noise (`pos_noise_std_m`) to simulator object poses so
    we can evaluate under both exact poses (`0.0m`) and realistic perception noise
    (`0.025m`).
    """

    pos_noise_std_m: float = 0.0
    include_task_prefix: bool = False
    seed: int = 0
    history: list[dict] = field(default_factory=list)
    _grasped_once: dict[int, bool] = field(default_factory=dict)
    _lifted_once: dict[int, bool] = field(default_factory=dict)
    _name_map: dict[str, str] | None = None
    _noise_offsets: dict[str, np.ndarray] = field(default_factory=dict)

    def __call__(self, env, obs, default_prompt: str) -> str:
        inner = env._env.env
        goals = [tuple(str(x) for x in g) for g in inner.parsed_problem["goal_state"]]
        pending = [i for i, g in enumerate(goals) if not inner._eval_predicate(list(g))]
        active_idx = pending[0] if pending else max(0, len(goals) - 1)

        # Collect current 3D positions
        raw_pos: dict[str, np.ndarray] = {}
        for k in inner.objects_dict:
            if k in inner.object_states_dict:
                raw_pos[k] = np.asarray(inner.object_states_dict[k].get_geom_state()["pos"], dtype=np.float64).copy()
        for g in goals:
            pick, place = goal_targets(g)
            for name in (pick, place):
                if name and name in inner.object_states_dict and name not in raw_pos:
                    raw_pos[name] = np.asarray(
                        inner.object_states_dict[name].get_geom_state()["pos"], dtype=np.float64
                    ).copy()

        if self._name_map is None:
            self._name_map = _distinguish_identical_names(list(raw_pos.keys()), raw_pos)
            if self.pos_noise_std_m > 0.0:
                rng = np.random.default_rng(self.seed)
                self._noise_offsets = {
                    k: rng.normal(0.0, self.pos_noise_std_m, size=3) for k in raw_pos
                }

        pos = {
            k: p + self._noise_offsets.get(k, 0.0)
            for k, p in raw_pos.items()
        }

        eef_t = np.asarray(obs.eef_pos, dtype=np.float64)
        grip_open_m = float(abs(obs.gripper_qpos[0] - obs.gripper_qpos[1]))

        # Scene string
        center_xy = np.mean([p[:2] for p in pos.values()], axis=0) if pos else np.zeros(2)
        goal_keys = []
        for g in goals:
            pick, place = goal_targets(g)
            if pick and pick in pos:
                goal_keys.append(pick)
            if place and place in pos:
                goal_keys.append(place)
        ordered_keys = goal_keys + [k for k in pos if k not in goal_keys]

        scene_items = []
        seen = set()
        for k in ordered_keys:
            label = self._name_map.get(k, clean_name(k))
            if label in seen:
                continue
            seen.add(label)
            scene_items.append(f"{label} ({table_sector(pos[k], center_xy)})")
            if len(scene_items) >= 5:
                break
        scene_str = ", ".join(scene_items) if scene_items else "tabletop workspace"

        # Active subgoal & geometry
        goal = goals[active_idx]
        pred = goal[0].lower()
        pick_name = goal[1] if len(goal) > 1 else ""
        place_name = goal[2] if (pred in ("in", "on") and len(goal) > 2) else None
        pick_label = self._name_map.get(pick_name, clean_name(pick_name))
        place_label = self._name_map.get(place_name, clean_name(place_name)) if place_name else None
        p_pick = pos.get(pick_name)
        p_place = pos.get(place_name) if place_name else None

        phase_tag = "reach"
        active_delta = None

        if pred in ("in", "on"):
            prep = "into" if pred == "in" else "onto"
            # Detect whether object is currently held by gripper
            dist_to_pick = float(np.linalg.norm(raw_pos.get(pick_name, eef_t) - eef_t))
            is_holding = (grip_open_m > 0.0035 and grip_open_m < 0.068 and dist_to_pick < 0.09)
            if is_holding:
                self._grasped_once[active_idx] = True
                if eef_t[2] > float(raw_pos.get(pick_name, eef_t)[2]) - 0.02 and eef_t[2] > 0.94:
                    self._lifted_once[active_idx] = True
            else:
                self._grasped_once[active_idx] = False
                self._lifted_once[active_idx] = False

            if not self._grasped_once.get(active_idx, False):
                active_target_label = pick_label
                active_target_pos = p_pick
                sec_label, sec_pos = place_label, p_place
                horiz = float(np.hypot(p_pick[0] - eef_t[0], p_pick[1] - eef_t[1])) if p_pick is not None else 1.0
                dz = float(eef_t[2] - p_pick[2]) if p_pick is not None else 1.0
                if horiz < 0.045 and dz < 0.08:
                    phase_tag = "descend_grasp"
                    subgoal_str = f"descend to {pick_label} and close gripper to grasp"
                elif horiz < 0.045:
                    phase_tag = "descend_grasp"
                    subgoal_str = f"lower gripper straight down to {pick_label}"
                else:
                    phase_tag = "reach"
                    subgoal_str = f"reach above {pick_label}"
            elif not self._lifted_once.get(active_idx, False):
                active_target_label = place_label or pick_label
                active_target_pos = p_place if p_place is not None else p_pick
                sec_label, sec_pos = None, None
                phase_tag = "lift"
                subgoal_str = f"lift grasped {pick_label} up from surface"
            else:
                active_target_label = place_label or pick_label
                active_target_pos = p_place if p_place is not None else p_pick
                sec_label, sec_pos = None, None
                horiz = float(np.hypot(p_place[0] - eef_t[0], p_place[1] - eef_t[1])) if p_place is not None else 1.0
                dz = float(eef_t[2] - p_place[2]) if p_place is not None else 1.0
                if horiz < 0.055 and dz < 0.09:
                    phase_tag = "lower_release"
                    subgoal_str = f"lower {pick_label} {prep} {place_label} and open gripper"
                elif horiz < 0.055:
                    phase_tag = "lower_release"
                    subgoal_str = f"align and lower {pick_label} {prep} {place_label}"
                else:
                    phase_tag = "transport"
                    subgoal_str = f"transport held {pick_label} toward {place_label}"

            geom_parts = []
            if active_target_pos is not None:
                active_delta = active_target_pos - eef_t
                role_tag = "destination" if self._grasped_once.get(active_idx, False) else "target"
                geom_parts.append(f"{role_tag} {active_target_label} is {format_delta_3d(active_delta)}")
            if sec_label and sec_pos is not None:
                geom_parts.append(f"destination {sec_label} is {format_delta_3d(sec_pos - eef_t)}")
            geom_str = "; ".join(geom_parts) if geom_parts else "target in view"
        else:
            verb = {"open": "pull open", "close": "push closed", "turnon": "turn on", "turnoff": "turn off"}.get(
                pred, pred
            )
            dist = float(np.linalg.norm(p_pick - eef_t)) if p_pick is not None else 0.0
            if dist > 0.09:
                phase_tag = "artic_approach"
                subgoal_str = f"approach {pick_label} to {verb}"
            else:
                phase_tag = "artic_engage"
                subgoal_str = f"engage and {verb} {pick_label}"
            if p_pick is not None:
                active_delta = p_pick - eef_t
                geom_str = f"target {pick_label} is {format_delta_3d(active_delta)}"
            else:
                geom_str = f"target {pick_label}"

        motion_str = causal_chunk_motion(phase_tag, active_delta)
        caption = (
            f"scene: {scene_str}. "
            f"subgoal: {subgoal_str}. "
            f"geometry: {geom_str}. "
            f"next 10 steps: {motion_str}."
        )
        if self.include_task_prefix and default_prompt:
            caption = f"task: {default_prompt}. {caption}"
        self.history.append({"t": int(env.t), "phase": phase_tag, "caption": caption})
        return caption
