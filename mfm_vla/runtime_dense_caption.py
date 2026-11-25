"""Observable dynamic Dense-v3 prompts for a grounded local-assist episode."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dense_caption import causal_chunk_motion, clean_name, format_delta_3d, table_sector


@dataclass
class GroundedDenseV3Captioner:
    """Generate the four-field prompt format used to train Dense-v3.

    Geometry comes from RGB-only active grounding and current proprioception.
    After the first grounded pick/place leg is released, the provider returns
    the original task prompt so later, ungrounded task stages retain semantics.
    """

    grasp_name: str
    grasp_pos: np.ndarray | None
    place_name: str
    place_pos: np.ndarray | None
    holding_fn: object
    history: list[dict] = field(default_factory=list)
    _lifted_once: bool = False
    _ever_held: bool = False
    _first_carry_released: bool = False

    def __post_init__(self) -> None:
        self.grasp_name = clean_name(self.grasp_name or "target object")
        self.place_name = clean_name(self.place_name or "destination")
        if self.grasp_pos is not None:
            self.grasp_pos = np.asarray(self.grasp_pos, dtype=np.float64)
        if self.place_pos is not None:
            self.place_pos = np.asarray(self.place_pos, dtype=np.float64)

    def mark_first_carry_released(self) -> None:
        self._first_carry_released = True

    def __call__(self, env, obs, default_prompt: str) -> str:
        eef = np.asarray(obs.eef_pos, dtype=np.float64)
        holding = bool(self.holding_fn())
        if holding:
            self._ever_held = True
            if (
                self.grasp_pos is not None
                and eef[2] > self.grasp_pos[2] + 0.05
            ):
                self._lifted_once = True
        elif self._ever_held and self._lifted_once:
            near_place = (
                self.place_pos is None
                or float(np.linalg.norm(self.place_pos[:2] - eef[:2])) < 0.14
            )
            if near_place:
                self._first_carry_released = True
            self._lifted_once = False

        if self._first_carry_released or self.grasp_pos is None:
            self.history.append(
                {
                    "t": int(env.t),
                    "phase": "raw_fallback",
                    "caption": default_prompt,
                }
            )
            return default_prompt

        positions = [
            point[:2]
            for point in (self.grasp_pos, self.place_pos)
            if point is not None
        ]
        centre = np.mean(positions, axis=0) if positions else np.zeros(2)
        scene = [f"{self.grasp_name} ({table_sector(self.grasp_pos, centre)})"]
        if self.place_pos is not None:
            scene.append(
                f"{self.place_name} ({table_sector(self.place_pos, centre)})"
            )

        if not holding:
            delta = self.grasp_pos - eef
            horizontal = float(np.linalg.norm(delta[:2]))
            vertical = float(eef[2] - self.grasp_pos[2])
            if horizontal < 0.045 and vertical < 0.08:
                phase = "descend_grasp"
                subgoal = (
                    f"descend to {self.grasp_name} and close gripper to grasp"
                )
            elif horizontal < 0.045:
                phase = "descend_grasp"
                subgoal = f"lower gripper straight down to {self.grasp_name}"
            else:
                phase = "reach"
                subgoal = f"reach above {self.grasp_name}"
            geometry = (
                f"target {self.grasp_name} is {format_delta_3d(delta)}"
            )
            if self.place_pos is not None:
                geometry += (
                    f"; destination {self.place_name} is "
                    f"{format_delta_3d(self.place_pos - eef)}"
                )
        else:
            active = self.place_pos if self.place_pos is not None else self.grasp_pos
            delta = active - eef
            horizontal = float(np.linalg.norm(delta[:2]))
            vertical = float(eef[2] - active[2])
            if not self._lifted_once:
                phase = "lift"
                subgoal = f"lift grasped {self.grasp_name} up from surface"
            elif horizontal < 0.055 and vertical < 0.09:
                phase = "lower_release"
                subgoal = (
                    f"lower {self.grasp_name} onto {self.place_name} "
                    "and open gripper"
                )
            elif horizontal < 0.055:
                phase = "lower_release"
                subgoal = (
                    f"align and lower {self.grasp_name} onto {self.place_name}"
                )
            else:
                phase = "transport"
                subgoal = (
                    f"transport held {self.grasp_name} toward {self.place_name}"
                )
            geometry = (
                f"destination {self.place_name} is {format_delta_3d(delta)}"
            )

        caption = (
            f"scene: {', '.join(scene)}. "
            f"subgoal: {subgoal}. "
            f"geometry: {geometry}. "
            f"next 10 steps: {causal_chunk_motion(phase, delta)}."
        )
        self.history.append(
            {"t": int(env.t), "phase": phase, "caption": caption}
        )
        return caption
