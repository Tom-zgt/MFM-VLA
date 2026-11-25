"""Generic GPT + π0.5 agent on LIBERO-PRO, one episode per call.

G1: coarse-to-fine GPT grounding of the first object to grasp, contact-free
``move_to`` above it, then π0.5 with the original instruction does every
contact. Simulator state is read only for evaluation logging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

import libero_compat  # noqa: F401

from eval_preposition_pick import LiftTracker, _positions, _target_object
from eval_vla_only import SUITE_HORIZONS
from generic_adapter import GraspMonitor, LiberoAdapter, libero_calibration
from dataclasses import replace

from generic_config import AgentConfig, ApproachConfig, MeasureConfig, MoveConfig, PlaceConfig, VerifyConfig
from generic_verify import check_grasp
from generic_grounding import ground_grasp_target
from generic_multiview import measure_in_second_view
from eval_multiview_measure import object_extent
from harness import TaskEnv
from rollout_recorder import RolloutRecorder, default_staging_root
from vla_executor import OpenPiVLAExecutor, VLAConfig


HERE = Path(__file__).resolve().parent


def _eval_container(env) -> str | None:
    """Evaluation only: the object whose region the BDDL in/on goal names."""
    for goal in env._env.env.parsed_problem["goal_state"]:
        if goal[0].lower() in ("in", "on") and len(goal) > 2:
            names = [n for n in env._env.env.objects_dict if goal[2].startswith(n)]
            names += [n for n in getattr(env._env.env, "fixtures_dict", {}) if goal[2].startswith(n)]
            if names:
                return max(names, key=len)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--mode", default="g1")
    parser.add_argument("--vla-port", type=int, default=8000)
    parser.add_argument("--vla-replan-steps", type=int, default=5)
    parser.add_argument("--clearance-above-top", type=float, default=None,
                        help="G2b dev comparison only; the frozen value lives in generic_config")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AgentConfig()
    if args.mode.startswith("g2b"):
        approach = ApproachConfig(reference="top")
        if args.clearance_above_top is not None:
            approach = replace(approach, clearance_above_top_m=args.clearance_above_top)
        cfg = replace(cfg, stage="G2b", approach=approach)
    elif args.mode.startswith("g2c"):
        cfg = replace(cfg, stage="G2c", approach=ApproachConfig(reference="policy"))
    elif args.mode.startswith("g2d"):
        cfg = replace(cfg, stage="G2d", approach=ApproachConfig(
            reference="policy", policy_profile="configs/policies/pi05_libero_lag10.json"))
    elif args.mode.startswith("g2a"):
        cfg = replace(cfg, stage="G2a")
    elif args.mode.startswith("g3a12"):
        cfg = replace(cfg, stage="G3a12", approach=ApproachConfig(reference="policy"),
                      move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True, order="place_first"),
                      place=PlaceConfig(policy_profile="configs/policies/pi05_libero_place_v2.json",
                                        grasp_settle_m=0.0001, grasp_settle_steps=3,
                                        carry_max_action_change=0.15))
    elif args.mode.startswith("g3a11"):
        cfg = replace(cfg, stage="G3a11", approach=ApproachConfig(reference="policy"),
                      move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True, order="place_first"),
                      place=PlaceConfig(policy_profile="configs/policies/pi05_libero_place_v2.json",
                                        grasp_settle_m=0.001, carry_max_action_change=0.15),
                      verify=VerifyConfig(grasp_check=True, grasp_reference=True))
    elif args.mode.startswith("g3a10"):
        cfg = replace(cfg, stage="G3a10", approach=ApproachConfig(reference="policy"),
                      move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True, order="place_first"),
                      place=PlaceConfig(policy_profile="configs/policies/pi05_libero_place_v2.json",
                                        grasp_settle_m=0.001, carry_max_action_change=0.15),
                      verify=VerifyConfig(grasp_check=True))
    elif args.mode.startswith("g3a9"):
        cfg = replace(cfg, stage="G3a9", approach=ApproachConfig(reference="policy"),
                      move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True, order="place_first"),
                      place=PlaceConfig(policy_profile="configs/policies/pi05_libero_place_v2.json",
                                        grasp_settle_m=0.001, carry_max_action_change=0.15,
                                        carry_max_action_norm=0.85))
    elif args.mode.startswith("g3a8"):
        cfg = replace(cfg, stage="G3a8", approach=ApproachConfig(reference="policy"),
                      move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True, order="place_first"),
                      place=PlaceConfig(policy_profile="configs/policies/pi05_libero_place_v2.json",
                                        grasp_settle_m=0.001, carry_max_action_change=0.15))
    elif args.mode.startswith("g3a7"):
        cfg = replace(cfg, stage="G3a7", approach=ApproachConfig(reference="policy"),
                      move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True, order="grasp_first"),
                      place=PlaceConfig(policy_profile="configs/policies/pi05_libero_place_v2.json",
                                        grasp_settle_m=0.001, carry_max_action_change=0.15))
    elif args.mode.startswith("g3a6"):
        cfg = replace(cfg, stage="G3a6", approach=ApproachConfig(reference="policy"),
                      move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True, order="grasp_first"),
                      place=PlaceConfig(policy_profile="configs/policies/pi05_libero_place_v2.json",
                                        grasp_settle_m=0.001))
    elif args.mode.startswith("g3a5"):
        cfg = replace(cfg, stage="G3a5", approach=ApproachConfig(reference="policy"),
                      move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True),
                      place=PlaceConfig(policy_profile="configs/policies/pi05_libero_place_v2.json",
                                        grasp_settle_m=0.001))
    elif args.mode.startswith("g3a4"):
        cfg = replace(cfg, stage="G3a4", approach=ApproachConfig(reference="policy"),
                      move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True),
                      place=PlaceConfig(policy_profile="configs/policies/pi05_libero_place_v2.json"))
    elif args.mode.startswith("g3a3"):
        cfg = replace(cfg, stage="G3a3", approach=ApproachConfig(reference="policy"),
                      move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True))
    elif args.mode.startswith("g3a2"):
        cfg = replace(cfg, stage="G3a2", approach=ApproachConfig(reference="policy"),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True))
    elif args.mode.startswith("g3a"):
        cfg = replace(cfg, stage="G3a", approach=ApproachConfig(reference="policy"))
    started = time.monotonic()
    env = executor = recorder = session = None
    row: dict = {}
    moves: list[dict] = []
    grounding = None
    try:
        horizon = SUITE_HORIZONS.get(args.suite) or SUITE_HORIZONS[args.suite.rsplit("_", 1)[0]]
        env = TaskEnv(args.suite, args.task, img_size=128, seed=args.init)
        env.allow_goal_shortcut = False
        env.reset(args.init)
        try:
            eval_target = _target_object(env)
        except ValueError:
            eval_target = None
        recorder = RolloutRecorder(args.artifact_dir, staging_root=default_staging_root())
        recorder.begin(
            {
                "suite": args.suite, "task": args.task, "init": args.init,
                "instruction": env.instruction, "executor": "pi05_libero",
                "experiment": f"generic/{args.mode}", "config": cfg.to_dict(),
            },
            env,
        )
        tracker = LiftTracker(env)

        def on_move(action, mode):
            tracker.update(env)
            recorder.record_step(env, action=action, mode=mode, decision=0)

        multiview = cfg.stage.startswith(("G2", "G3"))
        calibration = None if multiview else libero_calibration(args.suite)
        adapter = LiberoAdapter(env, calibration, cfg.move, on_step=on_move)

        from mfm_client import MFMClientSession

        session = MFMClientSession(cfg.models.perception, max_tokens=1000)

        home = adapter.proprio().eef_pos.copy()
        references: dict = {}

        def locate(role: str):
            homing = None
            if cfg.measure.from_home and np.linalg.norm(adapter.proprio().eef_pos - home) > cfg.move.tolerance_m:
                homing = adapter.move_over(home, gripper="hold", phase=f"home_{role}")
            info = ground_grasp_target(adapter, session, cfg.grounding, recorder.root, role=role)
            if homing is not None:
                info["home"] = homing
            frame_obs = info.pop("_obs")
            crop_image = info.pop("_crop_image")
            crop_to_frame = info.pop("_crop_to_frame")
            if role == "grasp" and info.get("fine", {}).get("bbox"):
                boxed = Image.fromarray(crop_image)
                ImageDraw.Draw(boxed).rectangle([float(v) for v in info["fine"]["bbox"]],
                                                outline=(255, 210, 0), width=3)
                references["grasp"] = {"image": np.asarray(boxed), "name": info["fine"].get("name", "object"),
                                       "evidence": info["fine"].get("visual_evidence", "")}
            point = info.pop("world")
            support = None if calibration is None else calibration.support_eef_z
            if multiview:
                mv = measure_in_second_view(
                    adapter, session, info["fine"], crop_image, crop_to_frame,
                    frame_obs, cfg.grounding.render_px, recorder.root, role=role,
                    max_gap_m=cfg.measure.max_ray_gap_m if cfg.stage.startswith("G3") else None,
                    relook_on=cfg.measure.relook_on,
                )
                accepted = mv["visible"] and (
                    cfg.stage == "G2a" or mv["ray_gap_m"] <= cfg.measure.max_ray_gap_m
                )
                top = mv.pop("top", None)
                point, support = (top, mv["support_z"]) if accepted else (None, None)
                mv["accepted"] = bool(accepted)
                info["second_view"] = mv
            return point, support, info

        def measure_place():
            place_target, _, place_info = locate("place")
            container = _eval_container(env)
            if container is not None and container in tracker.start:
                _, _, container_top = object_extent(env, container)
                place_info["eval_only_container"] = container
                place_info["eval_only_true_top_z"] = round(container_top, 4)
                if place_target is not None:
                    place_info["eval_only_top_error_m"] = round(float(place_target[2] - container_top), 4)
                    true_xy = tracker.start[container][:2]
                    place_info["eval_only_xy_error_m"] = round(float(np.linalg.norm(place_target[:2] - true_xy)), 4)
                    place_info["eval_only_correct_object"] = min(
                        tracker.start, key=lambda n: np.linalg.norm(tracker.start[n][:2] - place_target[:2])
                    ) == container
            return place_target, place_info

        place_target = place_info = None
        carry = cfg.stage.startswith("G3")
        if carry and cfg.measure.order == "place_first":
            place_target, place_info = measure_place()
        estimate, support_z, grounding = locate("grasp")
        if carry and cfg.measure.order == "grasp_first":
            place_target, place_info = measure_place()
        if eval_target is not None:
            truth = tracker.start[eval_target]
            _, true_bottom, true_top = object_extent(env, eval_target)
            grounding["eval_only_true_bottom_z"] = round(true_bottom, 4)
            grounding["eval_only_true_top_z"] = round(true_top, 4)
        if estimate is not None and eval_target is not None:
            grounding["eval_only_xy_error_m"] = round(float(np.linalg.norm(estimate[:2] - truth[:2])), 4)
            grounding["eval_only_nearest_object"] = min(
                tracker.start, key=lambda n: np.linalg.norm(tracker.start[n][:2] - estimate[:2])
            )
            grounding["eval_only_correct_object"] = grounding["eval_only_nearest_object"] == eval_target

        if estimate is None or support_z is None:
            grounding["fallback"] = "unmeasured: π0.5 starts from its initial pose"
        else:
            if cfg.approach.reference == "policy":
                profile = json.loads((HERE.parent / cfg.approach.policy_profile).read_text())
                start_pose = np.asarray(estimate, dtype=np.float64) + np.asarray(
                    profile["handoff_offset_from_object_top_m"], dtype=np.float64)
            else:
                if cfg.approach.reference == "top":
                    start_z = float(estimate[2]) + cfg.approach.clearance_above_top_m
                else:
                    start_z = support_z + cfg.approach.clearance_above_support_m
                start_pose = np.array([estimate[0], estimate[1], start_z])
            moves = adapter.move_over(start_pose, gripper="open", phase="approach")
        handoff_step = env.t

        executor = OpenPiVLAExecutor(VLAConfig(
            port=args.vla_port, resize_size=224, replan_steps=args.vla_replan_steps,
            max_skill_steps=horizon, prompt_mode="task",
        ))

        class Handover(Exception):
            pass

        monitor = (GraspMonitor(adapter, settle_m=cfg.place.grasp_settle_m,
                                    settle_steps=cfg.place.grasp_settle_steps)
                   if place_target is not None and support_z is not None else None)
        place = None

        def on_vla(step_env, action, call):
            adapter.gripper_command = float(np.sign(action[6]) or adapter.gripper_command)
            tracker.update(step_env)
            recorder.record_step(step_env, action=action, mode=f"pi05_{call}", decision=call - 1)
            if monitor is not None and place is None and monitor.update(action):
                raise Handover

        grasp_checks: list[dict] = []
        regrasps: list[dict] = []
        handed_over = False
        while True:
            try:
                result = executor.execute_task(
                    env, max_steps=horizon - env.t, benchmark_stop_on_success=True, step_callback=on_vla,
                )
                reason = result.reason
                handed_over = False
                break
            except Handover:
                handed_over = True
                if not cfg.verify.grasp_check:
                    break
                grasp_checks.append(check_grasp(
                    adapter, session, env.instruction, cfg.verify.check_px,
                    reference=references.get("grasp") if cfg.verify.grasp_reference else None))
                if grasp_checks[-1]["correct"] or len(regrasps) >= cfg.verify.max_regrasp:
                    break
                retry = json.loads((HERE.parent / cfg.verify.retry_profile).read_text())
                retry_pose = np.asarray(estimate, dtype=np.float64) + np.asarray(
                    retry["handoff_offset_from_object_top_m"], dtype=np.float64)
                lift = adapter.proprio().eef_pos + np.array([0.0, 0.0, cfg.verify.release_rise_m])
                regrasps.append({
                    "step": env.t,
                    "release": adapter.move_to(lift, gripper="open", phase="release_wrong"),
                    "approach": adapter.move_over(retry_pose, gripper="open", phase="regrasp"),
                })
                monitor = GraspMonitor(adapter, settle_m=cfg.place.grasp_settle_m,
                                    settle_steps=cfg.place.grasp_settle_steps)
                handed_over = False
                if env.t >= horizon:
                    reason = "horizon reached during regrasp"
                    break
        if handed_over:
            profile = json.loads((HERE.parent / cfg.place.policy_profile).read_text())
            hang = monitor.close_z - support_z
            offset = np.asarray(profile["place_offset_from_target_top_m"], dtype=np.float64)
            goal = np.array([
                place_target[0] + offset[0], place_target[1] + offset[1],
                place_target[2] + profile["held_bottom_above_target_top_m"] + hang,
            ])
            crossing = profile.get("held_bottom_above_target_top_crossing_m")
            traverse_z = None if crossing is None else float(place_target[2] + crossing + hang)
            place = {
                "step": env.t, "hang_m": round(float(hang), 4), "goal": np.round(goal, 4).tolist(),
                "traverse_z": traverse_z,
                "moves": adapter.move_over(goal, gripper="hold", phase="carry", traverse_z=traverse_z,
                                           max_action_change=cfg.place.carry_max_action_change,
                                           max_action_norm=cfg.place.carry_max_action_norm),
                "still_holding": adapter.holding(),
            }
            place["handback_step"] = env.t
            result = executor.execute_task(
                env, max_steps=horizon - env.t, benchmark_stop_on_success=True, step_callback=on_vla,
            )
            reason = f"handover at {place['step']}; {result.reason}"
        first = tracker.first_lifted()
        row = {
            "suite": args.suite, "task": args.task, "init": args.init, "mode": args.mode,
            "instruction": env.instruction, "eval_target": eval_target,
            "success": bool(env.success), "steps": env.t, "handoff_step": handoff_step,
            "move_steps": adapter.move_steps, "pi05_steps": env.t - adapter.move_steps,
            "first_lifted": first, "first_lifted_is_target": first == eval_target,
            "max_lift_m": {k: round(v, 3) for k, v in tracker.max_lift.items() if v >= 0.01},
            "moves": moves, "grounding": grounding, "place_grounding": place_info, "place": place,
            "grasp_checks": grasp_checks, "regrasps": regrasps,
            "gpt_usage": session.usage(),
            "reason": reason, "vla_inference_calls": executor.inference_calls,
            "artifacts": str(args.artifact_dir), "crash": False,
        }
    except Exception as exc:
        row = {
            "suite": args.suite, "task": args.task, "init": args.init, "mode": args.mode,
            "success": False, "steps": env.t if env is not None else 0,
            "moves": moves, "grounding": grounding,
            "reason": f"{type(exc).__name__}: {exc}", "detail": traceback.format_exc(limit=10),
            "artifacts": str(args.artifact_dir), "crash": True,
        }
    finally:
        row["duration_s"] = round(time.monotonic() - started, 2)
        if session is not None:
            session.close()
        if executor is not None:
            executor.close()
        if env is not None:
            if recorder is not None:
                row["videos"] = recorder.finalize(
                    "interrupted" if row["crash"] else "completed",
                    step=env.t, success=bool(row["success"]), reason=row["reason"],
                )
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
    print(json.dumps(row, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
