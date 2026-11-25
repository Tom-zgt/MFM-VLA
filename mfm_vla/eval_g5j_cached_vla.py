"""Zero-GPT evaluation of G5j + fine-tuned π0.5 (e.g., pi05_dense_v3) by replaying cached GPT grounding from g5j_dev.jsonl.

In G5j, GPT is called for:
  1. step-0 `route` (`assist` vs `direct`)
  2. initial 2-view 3D triangulation of `place_target` and `grasp_target` (`estimate`, `support_z`)
  3. `check_grasp` after a lift event

Here we replay the exact cached GPT 3D measurements (`estimate`, `support_z`, `place_target`) from
`reports/generic/g5j_dev.jsonl` (re-triangulating `place_target` from the recorded `place_grounding`
pixels if the reference episode never reached `vla_carry`), and execute the full G5j closed-loop
control with `pi05_dense_v3` (both with Raw Prompt and with GPT-grounded Dynamic 10-Step Dense Caption).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

import libero_compat  # noqa: F401
from dense_caption import causal_chunk_motion, clean_name, format_delta_3d, table_sector
from eval_g5 import CONFIGS, CarryNeeded, HERE, Lifted, OutOfTime, PlaceLoop, Release, _cruise_to
from eval_preposition_pick import LiftTracker, _target_object
from eval_vla_only import SUITE_HORIZONS
from generic_adapter import GraspMonitor, LiberoAdapter
from generic_articulation import drawer_poses
from generic_multiview import _point, height_under, pixel_ray, triangulate
from harness import TaskEnv
from vla_executor import OpenPiVLAExecutor, VLAConfig


class GPTGroundedDenseCaptioner:
    """Dynamic 10-step Dense Captioner driven purely by GPT's grounded 3D coordinates + proprioception."""

    def __init__(
        self,
        grasp_name: str,
        grasp_pos: np.ndarray | None,
        place_name: str,
        place_pos: np.ndarray | None,
        holding_fn,
    ):
        self.grasp_name = clean_name(grasp_name or "target object")
        self.grasp_pos = np.asarray(grasp_pos, dtype=np.float64) if grasp_pos is not None else None
        self.place_name = clean_name(place_name or "destination")
        self.place_pos = np.asarray(place_pos, dtype=np.float64) if place_pos is not None else None
        self.holding_fn = holding_fn
        self.history: list[dict] = []
        self._lifted_once = False
        self._ever_held = False
        self._first_carry_released = False

    def mark_first_carry_released(self) -> None:
        self._first_carry_released = True

    def update_grasp(self, pos: np.ndarray | None) -> None:
        if pos is not None:
            self.grasp_pos = np.asarray(pos, dtype=np.float64)

    def __call__(self, env: TaskEnv, obs, default_prompt: str) -> str:
        eef_t = np.asarray(obs.eef_pos, dtype=np.float64)
        is_holding = bool(self.holding_fn())
        if is_holding:
            self._ever_held = True
            if eef_t[2] > 0.94:
                self._lifted_once = True
        else:
            if self._ever_held and self._lifted_once:
                # If the object was lifted and is now released near the place target (or after PlaceLoop release),
                # the first pick-and-place stage is finished! Fall back to `default_prompt` for any remaining subgoal.
                if self.place_pos is None or float(np.hypot(self.place_pos[0] - eef_t[0], self.place_pos[1] - eef_t[1])) < 0.14:
                    self._first_carry_released = True
            self._lifted_once = False

        if self._first_carry_released or self.grasp_pos is None:
            self.history.append({"t": int(env.t), "phase": "raw_fallback", "caption": default_prompt})
            return default_prompt

        pts = [p[:2] for p in (self.grasp_pos, self.place_pos) if p is not None]
        center_xy = np.mean(pts, axis=0) if pts else np.zeros(2)
        scene_parts = []
        if self.grasp_pos is not None:
            scene_parts.append(f"{self.grasp_name} ({table_sector(self.grasp_pos, center_xy)})")
        if self.place_pos is not None:
            scene_parts.append(f"{self.place_name} ({table_sector(self.place_pos, center_xy)})")
        scene_str = ", ".join(scene_parts) if scene_parts else "tabletop workspace"

        active_delta = None
        if not is_holding:
            p_pick = self.grasp_pos
            horiz = float(np.hypot(p_pick[0] - eef_t[0], p_pick[1] - eef_t[1])) if p_pick is not None else 1.0
            dz = float(eef_t[2] - p_pick[2]) if p_pick is not None else 1.0
            if horiz < 0.045 and dz < 0.08:
                phase_tag = "descend_grasp"
                subgoal_str = f"descend to {self.grasp_name} and close gripper to grasp"
            elif horiz < 0.045:
                phase_tag = "descend_grasp"
                subgoal_str = f"lower gripper straight down to {self.grasp_name}"
            else:
                phase_tag = "reach"
                subgoal_str = f"reach above {self.grasp_name}"
            geom_parts = []
            if p_pick is not None:
                active_delta = p_pick - eef_t
                geom_parts.append(f"target {self.grasp_name} is {format_delta_3d(active_delta)}")
            if self.place_pos is not None:
                geom_parts.append(f"destination {self.place_name} is {format_delta_3d(self.place_pos - eef_t)}")
            geom_str = "; ".join(geom_parts) if geom_parts else "target in view"
        else:
            p_place = self.place_pos if self.place_pos is not None else self.grasp_pos
            horiz = float(np.hypot(p_place[0] - eef_t[0], p_place[1] - eef_t[1])) if p_place is not None else 1.0
            dz = float(eef_t[2] - p_place[2]) if p_place is not None else 1.0
            if not self._lifted_once:
                phase_tag = "lift"
                subgoal_str = f"lift grasped {self.grasp_name} up from surface"
            elif horiz < 0.055 and dz < 0.09:
                phase_tag = "lower_release"
                subgoal_str = f"lower {self.grasp_name} onto {self.place_name} and open gripper"
            elif horiz < 0.055:
                phase_tag = "lower_release"
                subgoal_str = f"align and lower {self.grasp_name} onto {self.place_name}"
            else:
                phase_tag = "transport"
                subgoal_str = f"transport held {self.grasp_name} toward {self.place_name}"
            if p_place is not None:
                active_delta = p_place - eef_t
                geom_str = f"destination {self.place_name} is {format_delta_3d(active_delta)}"
            else:
                geom_str = f"destination {self.place_name}"

        motion_str = causal_chunk_motion(phase_tag, active_delta)
        caption = (
            f"scene: {scene_str}. "
            f"subgoal: {subgoal_str}. "
            f"geometry: {geom_str}. "
            f"next 10 steps: {motion_str}."
        )
        self.history.append({"t": int(env.t), "phase": phase_tag, "caption": caption})
        return caption


def reconstruct_place_target(row: dict, adapter: LiberoAdapter, cfg, frame_obs) -> np.ndarray | None:
    """Extract `place_target` from cached `vla_carry` goal or re-triangulate from `place_grounding`."""
    if row.get("override_place_target") is not None:
        return np.asarray(row["override_place_target"], dtype=np.float64)
    place_profile = json.loads((HERE.parent / cfg.place.policy_profile).read_text())
    offset = np.asarray(place_profile["place_offset_from_target_top_m"], dtype=np.float64)
    evs = row.get("events") or []

    # Case 1: `carry` event has exact `hang_m` and `goal`
    for e in evs:
        if e.get("why") == "carry" and e.get("goal") is not None and "hang_m" in e:
            goal = np.asarray(e["goal"], dtype=np.float64)
            hang = float(e["hang_m"])
            return np.array([
                goal[0] - offset[0],
                goal[1] - offset[1],
                goal[2] - place_profile["held_bottom_above_target_top_m"] - hang,
            ])

    # Case 2: Re-triangulate from `place_grounding` pixels at survey pose
    pg = row.get("place_grounding") or {}
    sv = pg.get("second_view") or {}
    if not sv.get("accepted"):
        return None
    crop = pg.get("crop")
    if not crop:
        return None
    x0, y0, x1, y1 = crop
    render_px = cfg.grounding.render_px

    def crop_to_frame(uv: np.ndarray) -> np.ndarray:
        return np.array([x0 + uv[0] / render_px * (x1 - x0), y0 + uv[1] / render_px * (y1 - y0)])

    chosen_idx = sv.get("chosen_view", len(sv.get("views", [])) - 1)
    views = sv.get("views") or []
    if not views or chosen_idx >= len(views):
        return None
    reply = views[chosen_idx]["reply"]
    wrist = adapter.movable_cameras[0]
    wrist_obs = adapter.observe([wrist], render_px)
    top_a = crop_to_frame(np.asarray(reply["a_top_center"], dtype=np.float64))
    top_b = _point(reply["b_top_center"])
    top, _ = triangulate(pixel_ray(frame_obs, "agentview", top_a), pixel_ray(wrist_obs, wrist, top_b))
    return top


def _single_view_plane_project(grounding: dict | None, frame_obs, suite: str, render_px: int, is_place: bool = False) -> tuple[np.ndarray | None, float | None]:
    if not grounding or not grounding.get("crop") or not grounding.get("fine"):
        return None, None
    x0, y0, x1, y1 = grounding["crop"]
    fine = grounding["fine"]
    bx = fine.get("bbox")
    if not bx or len(bx) != 4:
        return None, None
    uv = np.array([
        x0 + ((bx[0] + bx[2]) * 0.5) / render_px * (x1 - x0),
        y0 + ((bx[1] + bx[3]) * 0.5) / render_px * (y1 - y0),
    ], dtype=np.float64)
    z_nom = 0.0 if suite.startswith("libero_object") else 0.90
    z_top = z_nom + (0.08 if (is_place and suite.startswith("libero_object")) else 0.04)
    origin, direction = pixel_ray(frame_obs, "agentview", uv)
    s = (z_top - origin[2]) / direction[2]
    return origin + s * direction, z_nom


def run_cached_g5j_episode(
    row: dict,
    port: int,
    dense_mode: str,
    executor_name: str,
    *,
    single_view_only: bool = False,
    no_retry: bool = False,
    no_carry_guidance: bool = False,
) -> dict:
    cfg = CONFIGS["g5j"]
    started = time.monotonic()
    suite, task, init = row["suite"], int(row["task"]), int(row["init"])
    horizon = SUITE_HORIZONS.get(suite) or SUITE_HORIZONS[suite.rsplit("_", 1)[0]]

    env = TaskEnv(suite, task, img_size=128, seed=init)
    env.allow_goal_shortcut = False
    env.reset(init)
    try:
        eval_target = _target_object(env)
    except ValueError:
        eval_target = row.get("eval_target")

    tracker = LiftTracker(env)

    def on_move(action, mode):
        tracker.update(env)
        if env.t >= horizon or env.success:
            raise OutOfTime

    adapter = LiberoAdapter(env, None, cfg.move, on_step=on_move)
    if cfg.verify.hold_min_opening_m is not None:
        adapter.hold_opening_m = (cfg.verify.hold_min_opening_m, adapter.hold_opening_m[1])
    home = adapter.proprio().eef_pos.copy()

    executor = OpenPiVLAExecutor(VLAConfig(
        port=port, resize_size=224, replan_steps=5,
        max_skill_steps=horizon, prompt_mode="task",
    ))

    events: list[dict] = []
    captioner = None
    reason = ""
    try:
        rt = row.get("route")
        route_mode = rt.get("route") if isinstance(rt, dict) else rt
        frame_obs = adapter.observe(["agentview"], cfg.grounding.render_px)

        ref_events = row.get("events") or []
        drawer_ev = next((e for e in ref_events if e.get("why") == "drawer"), None)

        if route_mode != "assist":
            res = executor.execute_task(env, max_steps=horizon - env.t, benchmark_stop_on_success=True,
                                        step_callback=lambda s_env, a, c: tracker.update(s_env))
            reason = res.reason
        elif drawer_ev is not None:
            a_cfg = cfg.articulation
            profile = json.loads((HERE.parent / a_cfg.policy_profile).read_text())
            sv_goal = (drawer_ev.get("survey") or {}).get("goal")
            if sv_goal is not None:
                try:
                    adapter.move_to(np.asarray(sv_goal, dtype=np.float64), gripper="open", phase="survey")
                except OutOfTime:
                    pass
            far = drawer_ev.get("far") or {}
            if far.get("accepted") and far.get("point") is not None and far.get("outward_xy") is not None:
                point = np.asarray(far["point"], dtype=np.float64)
                outward = np.asarray(far["outward_xy"], dtype=np.float64)
                for att in (drawer_ev.get("attempts") or []):
                    if horizon - env.t < a_cfg.min_steps_for_attempt:
                        break
                    try:
                        poses = drawer_poses(profile, point, outward, a_cfg.front_clearance_m)
                        eef = adapter.proprio().eef_pos
                        front = poses["front"]
                        adapter.move_to(np.array([front[0], front[1], max(eef[2], front[2])]), gripper="open",
                                        phase="drawer_traverse", tolerance_m=cfg.measure.waypoint_tolerance_m,
                                        quat_xyzw=poses["quat"])
                        adapter.move_to(front, gripper="open", phase="drawer_front", quat_xyzw=poses["quat"])
                        if att.get("used") == "close" and (att.get("close") or {}).get("point") is not None:
                            point = np.asarray(att["close"]["point"], dtype=np.float64)
                            outward = np.asarray(att["close"]["outward_xy"], dtype=np.float64)
                            poses = drawer_poses(profile, point, outward, a_cfg.front_clearance_m)
                        quat, out = poses["quat"], poses["out"]
                        adapter.move_to(poses["front"], gripper="open", phase="drawer_front2", quat_xyzw=quat)
                        adapter.move_to(poses["above"], gripper="open", phase="drawer_in", quat_xyzw=quat)
                        adapter.move_to(poses["pressed"], gripper="open", phase="drawer_press", quat_xyzw=quat)
                        adapter.move_to(poses["pressed"] + out * a_cfg.pull_travel_m, gripper="open",
                                        phase="drawer_pull", quat_xyzw=quat,
                                        max_action_norm=a_cfg.pull_max_action_norm)
                        if (att.get("check") or {}).get("done") or env.success:
                            break
                        hand = adapter.proprio().eef_pos
                        adapter.move_to(hand + out * 0.04 + np.array([0.0, 0.0, 0.06]), gripper="open",
                                        phase="drawer_back_off", quat_xyzw=quat)
                    except OutOfTime:
                        break
            if not env.success and env.t < horizon:
                res = executor.execute_task(env, max_steps=horizon - env.t, benchmark_stop_on_success=True,
                                            step_callback=lambda s_env, a, c: tracker.update(s_env))
                reason = res.reason
            else:
                reason = "success" if env.success else "drawer_done"
        else:
            start_ev = next((e for e in ref_events if e.get("why") == "start"), {})
            if single_view_only:
                estimate, support_z = _single_view_plane_project(
                    start_ev.get("grounding"), frame_obs, suite, cfg.grounding.render_px, is_place=False
                )
                has_place = bool((row.get("place_grounding") or {}).get("second_view") or any(e.get("why") == "carry" for e in ref_events))
                place_target, _ = (
                    _single_view_plane_project(
                        row.get("place_grounding"), frame_obs, suite, cfg.grounding.render_px, is_place=True
                    )
                    if has_place
                    else (None, None)
                )
            else:
                # 1. Replay survey move so wrist camera is in the exact pose used for triangulation
                survey_rec = row.get("survey")
                if survey_rec and survey_rec.get("goal") is not None:
                    try:
                        adapter.move_to(np.asarray(survey_rec["goal"], dtype=np.float64),
                                        gripper="open", phase="survey")
                    except OutOfTime:
                        pass

                place_target = reconstruct_place_target(row, adapter, cfg, frame_obs)
                estimate = (np.asarray(start_ev["estimate"], dtype=np.float64)
                            if start_ev.get("estimate") is not None else None)
                support_z = start_ev.get("support_z")

            if no_carry_guidance:
                place_target = None

            grasp_name = (start_ev.get("grounding", {}).get("fine", {}).get("name")
                          or eval_target or "target object")
            place_name = (row.get("place_grounding", {}).get("fine", {}).get("name")
                          or "destination")

            if dense_mode == "gpt_dense":
                captioner = GPTGroundedDenseCaptioner(
                    grasp_name=grasp_name,
                    grasp_pos=estimate,
                    place_name=place_name,
                    place_pos=place_target,
                    holding_fn=adapter.holding,
                )
                executor.prompt_builder = captioner

            grasp_profile = json.loads((HERE.parent / cfg.approach.policy_profile).read_text())
            retry_profile = json.loads((HERE.parent / cfg.verify.retry_profile).read_text())
            place_profile = json.loads((HERE.parent / cfg.place.policy_profile).read_text())

            def approach(goal: np.ndarray, phase: str):
                cruise = max(float(home[2]), float(goal[2]))
                return _cruise_to(adapter, cfg, goal, cruise, "open", phase)

            if estimate is not None:
                goal = estimate + np.asarray(grasp_profile["handoff_offset_from_object_top_m"], dtype=np.float64)
                approach(goal, "approach")

            loop = PlaceLoop(cfg, env, adapter, home, horizon, events)
            state = {"monitor": None}
            if estimate is not None and place_target is not None:
                state["monitor"] = GraspMonitor(adapter, settle_m=cfg.place.grasp_settle_m,
                                                settle_steps=cfg.place.grasp_settle_steps)

            def on_vla(step_env, action, call):
                adapter.gripper_command = float(np.sign(action[6]) or adapter.gripper_command)
                tracker.update(step_env)
                if loop.step(action):
                    if captioner is not None and loop.state.get("opened") and not adapter.holding():
                        captioner.mark_first_carry_released()
                    return
                if state["monitor"] is not None and state["monitor"].update(action):
                    raise Lifted

            max_regrasp_lim = 0 if no_retry else cfg.verify.max_regrasp
            max_redo_lim = 0 if no_retry else cfg.place.max_redo_after_drop
            regrasps = redo = 0
            while env.t < horizon and not env.success:
                carry_now = False
                try:
                    res = executor.execute_task(env, max_steps=horizon - env.t,
                                                benchmark_stop_on_success=True, step_callback=on_vla)
                    reason = res.reason
                    break
                except Lifted:
                    lifted_objs = [n for n, v in tracker.max_lift.items() if v >= 0.02]
                    # Check if grasped object matches the grounded target (or closest object to GPT estimate)
                    expected_obj = (start_ev.get("grounding", {}).get("eval_only_nearest_object")
                                    or eval_target)
                    correct = bool(lifted_objs and (expected_obj is None or expected_obj in lifted_objs))
                    if not correct and regrasps < max_regrasp_lim and estimate is not None:
                        regrasps += 1
                        lift = adapter.proprio().eef_pos + np.array([0.0, 0.0, cfg.verify.release_rise_m])
                        adapter.move_to(lift, gripper="open", phase="release_wrong")
                        goal = estimate + np.asarray(retry_profile["handoff_offset_from_object_top_m"])
                        approach(goal, "regrasp")
                        state["monitor"] = GraspMonitor(adapter, settle_m=cfg.place.grasp_settle_m,
                                                        settle_steps=cfg.place.grasp_settle_steps)
                        continue
                    if not correct or place_target is None or support_z is None:
                        state["monitor"] = None
                        continue
                    hang = state["monitor"].close_z - support_z
                    offset = np.asarray(place_profile["place_offset_from_target_top_m"], dtype=np.float64)
                    goal = np.array([place_target[0] + offset[0], place_target[1] + offset[1],
                                     place_target[2] + place_profile["held_bottom_above_target_top_m"] + hang])
                    crossing = place_profile.get("held_bottom_above_target_top_crossing_m")
                    cruise = float(place_target[2] + (crossing or 0.0) + hang)
                    pending = {"goal": goal, "hang": hang, "cruise": max(cruise, float(goal[2]))}
                    if not loop.begin(pending):
                        continue
                    carry_now = True
                except CarryNeeded:
                    carry_now = True
                except Release:
                    loop.release()
                    if captioner is not None:
                        captioner.mark_first_carry_released()
                if carry_now:
                    carry = loop.carry()
                    if not carry["still_holding"] and redo < max_redo_lim and estimate is not None:
                        redo += 1
                        adapter.move_to(adapter.proprio().eef_pos, gripper="open", phase="drop_open")
                        goal = estimate + np.asarray(grasp_profile["handoff_offset_from_object_top_m"])
                        approach(goal, "dropped")
                        state["monitor"] = GraspMonitor(adapter, settle_m=cfg.place.grasp_settle_m,
                                                        settle_steps=cfg.place.grasp_settle_steps)
                        continue
                    loop.hand_back()
    except OutOfTime:
        reason = f"agent motion ended at {env.t}/{horizon}"
    finally:
        executor.close()
        success = bool(env.success)
        steps = int(env.t)
        env.close()

    first = tracker.first_lifted()
    pert = row.get("perturbation") or {
        "lan": "semantic", "object": "object", "swap": "position", "task": "task"
    }.get(suite.split("_")[-1], "base")
    return {
        "suite": suite, "task": task, "init": init,
        "perturbation": pert,
        "executor": executor_name,
        "dense_mode": dense_mode,
        "route": route_mode,
        "success": success,
        "steps": steps,
        "move_steps": adapter.move_steps,
        "pi05_steps": steps - adapter.move_steps,
        "first_lifted": first,
        "first_lifted_is_target": first == eval_target,
        "vla_inference_calls": executor.inference_calls,
        "reason": reason,
        "duration_s": round(time.monotonic() - started, 2),
        "crash": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite")
    parser.add_argument("--task", type=int)
    parser.add_argument("--init", type=int)
    parser.add_argument("--vla-port", type=int, default=8130)
    parser.add_argument("--dense-mode", choices=("none", "gpt_dense"), default="gpt_dense")
    parser.add_argument("--executor-name", default="g5j_dense_v3")
    parser.add_argument("--ref-jsonl", type=Path, default=Path("reports/generic/g5j_dev.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    # Batch worker mode args
    parser.add_argument("--ports", type=int, nargs="+")
    parser.add_argument("--workers-per-port", type=int, default=2)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--single-view-only", action="store_true")
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--no-carry-guidance", action="store_true")
    args = parser.parse_args()

    ref_rows = {}
    for line in args.ref_jsonl.read_text().splitlines():
        r = json.loads(line)
        if not r.get("crash"):
            ref_rows[(r["suite"], int(r["task"]), int(r["init"]))] = r

    if args.suite is not None:
        key = (args.suite, int(args.task), int(args.init))
        row = ref_rows[key]
        try:
            out = run_cached_g5j_episode(
                row,
                args.vla_port,
                args.dense_mode,
                args.executor_name,
                single_view_only=args.single_view_only,
                no_retry=args.no_retry,
                no_carry_guidance=args.no_carry_guidance,
            )
        except Exception as exc:
            out = {
                "suite": args.suite, "task": int(args.task), "init": int(args.init),
                "executor": args.executor_name, "dense_mode": args.dense_mode,
                "success": False, "steps": 0, "reason": f"{type(exc).__name__}: {exc}",
                "detail": traceback.format_exc(limit=8), "crash": True,
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as fh:
            fh.write(json.dumps(out, separators=(",", ":")) + "\n")
            fh.flush()
        print(f"{args.executor_name} {args.suite} t{args.task} i{args.init}: "
              f"{'OK' if out['success'] else 'FAIL'} ({out.get('steps')} steps)", flush=True)
        if out.get("crash"):
            raise SystemExit(2)
        return

    # Driver mode across all 560 cached episodes
    assert args.ports and args.log_dir
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(3):
        done = set()
        if args.output.exists():
            for line in args.output.read_text().splitlines():
                r = json.loads(line)
                if not r.get("crash"):
                    done.add((r["suite"], int(r["task"]), int(r["init"])))

        todo = [k for k in sorted(ref_rows) if k not in done]
        print(f"{args.executor_name}: {len(todo)}/{len(ref_rows)} episodes to run", flush=True)
        if not todo:
            break
        work: queue.Queue = queue.Queue()
        for k in todo:
            work.put(k)
        lock = threading.Lock()

        def worker(port: int, wid: int) -> None:
            import os
            while True:
                try:
                    suite, task, init = work.get_nowait()
                except queue.Empty:
                    return
                cmd = [
                    sys.executable, "mfm_vla/eval_g5j_cached_vla.py",
                    "--suite", suite, "--task", str(task), "--init", str(init),
                    "--vla-port", str(port), "--dense-mode", args.dense_mode,
                    "--executor-name", args.executor_name,
                    "--ref-jsonl", str(args.ref_jsonl), "--output", str(args.output),
                ]
                if args.single_view_only:
                    cmd.append("--single-view-only")
                if args.no_retry:
                    cmd.append("--no-retry")
                if args.no_carry_guidance:
                    cmd.append("--no-carry-guidance")
                env = dict(os.environ, MUJOCO_EGL_DEVICE_ID=str((port + wid) % 4))
                log_file = args.log_dir / f"{suite}_t{task}_i{init}.log"
                with log_file.open("w") as fh:
                    code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
                with lock:
                    print(f"{args.executor_name} {suite} t{task} i{init} exit={code}", flush=True)

        threads = [
            threading.Thread(target=worker, args=(p, w))
            for p in args.ports for w in range(args.workers_per_port)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    final_rows = {}
    for line in args.output.read_text().splitlines():
        r = json.loads(line)
        if not r.get("crash"):
            final_rows[(r["suite"], int(r["task"]), int(r["init"]))] = r
    ok = sum(r["success"] for r in final_rows.values())
    print(f"G5J_CACHED_DONE {args.executor_name} episodes={len(final_rows)} "
          f"success={ok}/{len(final_rows)} ({ok / max(1, len(final_rows)):.3f})", flush=True)


if __name__ == "__main__":
    main()
