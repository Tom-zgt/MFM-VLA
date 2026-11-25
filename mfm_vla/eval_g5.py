"""G5: route, then assist only where the VLA is out of its training distribution.

At step 0 (no simulator time) GPT plans the instruction and matches it against
the VLA's training tasks; if it is one of them in the same arrangement, or has no
carry step, π0.5 runs alone exactly as in the baseline (G5h: no training-task
match; GPT plans from the instruction and the start image only, and every carry
or drawer step is assisted; G5i: the closed gripper counts as holding only above 3 mm,
since the empty Panda gripper closes to 2.0-2.4 mm). Otherwise the G3a8
hand-off runs (grasp and place targets measured in two views, ``move_over`` to
π0.5's own pre-grasp pose, π0.5 grasps, ``move_over`` carries to its own
pre-release pose, π0.5 places), with:

- a thin-grasp detector (bowl rims) confirmed by the GPT grasp check against the
  grounded reference, and a regrasp from a closer pose when the wrong object is held;
- measurements kept from the closest-meeting view and accepted up to a looser gap;
- a stalled carry retried once through the start pose;
- a second hand-off when the object was dropped during the carry;
- a free-space release if π0.5 does not open the gripper over the target.

π0.5 makes every contact. Simulator state is read only for evaluation logging.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

import libero_compat  # noqa: F401

from eval_generic import _eval_container
from eval_multiview_measure import object_extent
from eval_preposition_pick import LiftTracker, _target_object
from eval_vla_only import SUITE_HORIZONS
from generic_adapter import GRIPPER_COMMAND, GraspMonitor, LiberoAdapter
from generic_articulation import check_open, drawer_poses, ground_handle, locate_in_wrist
from generic_config import (AgentConfig, ApproachConfig, ArticulationConfig, MeasureConfig, MoveConfig,
                            PlaceConfig, ProgramConfig, RouteConfig, VerifyConfig)
from generic_grounding import ground_grasp_target
from generic_multiview import measure_in_second_view, pixel_ray, survey_goal
from generic_route import load_catalog, route
from generic_verify import check_grasp
from harness import TaskEnv
from rollout_recorder import RolloutRecorder, default_staging_root
from runtime_dense_caption import GroundedDenseV3Captioner
from vla_executor import OpenPiVLAExecutor, VLAConfig

HERE = Path(__file__).resolve().parent

CONFIG = AgentConfig(
    stage="G5", approach=ApproachConfig(reference="policy"),
    move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
    measure=MeasureConfig(relook_on="fixed_ray", from_home=True, order="place_first",
                          use_best=True, accept_gap_m=0.025, survey_depth_m=(0.02, 0.35)),
    place=PlaceConfig(policy_profile="configs/policies/pi05_libero_place_v2.json",
                      grasp_settle_m=0.001, carry_max_action_change=0.15,
                      reroute_on_stall=True, release_wait_steps=30, max_redo_after_drop=1,
                      vla_carry_first_steps=15),
    verify=VerifyConfig(grasp_check=True, grasp_reference=True, hold_min_opening_m=0.002),
    route=RouteConfig(enabled=True),
)


class Lifted(Exception):
    pass


class Release(Exception):
    pass


class OutOfTime(Exception):
    pass


class TryFailed(Exception):
    pass


class CarryNeeded(Exception):
    pass



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--mode", default="g5")
    parser.add_argument("--vla-port", type=int, default=8000)
    parser.add_argument("--vla-replan-steps", type=int, default=5)
    parser.add_argument("--force-route", choices=("direct", "assist"), default=None,
                        help="ablation: skip the router")
    parser.add_argument(
        "--dense-caption",
        action="store_true",
        help="use the grounded Dense-v3 prompt format inside forced G5j assist",
    )
    parser.add_argument(
        "--cached-program",
        type=Path,
        help=(
            "cached capability-learning task program; together with --force-route "
            "this bypasses G5j's internal route call"
        ),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _external_route_decision(
    program_path: Path,
    instruction: str,
    forced_route: str,
    cfg,
) -> dict:
    """Build G5j's execution metadata without invoking its internal router.

    The external capability router has already selected direct or G5j.  G5j
    still needs to know whether its carry or drawer executor applies, which is
    derived from the frozen dev-only task program rather than another route
    decision.
    """
    program = json.loads(program_path.read_text())
    if str(program.get("instruction", "")).strip() != instruction.strip():
        raise ValueError("cached task program does not match episode instruction")
    demands = [str(value).lower() for value in program.get("vla_skill_demands", [])]
    carries = any("transport" in value for value in demands)
    searchable = json.dumps(program, ensure_ascii=False).lower()
    assist_acts = (
        ["drawer"]
        if cfg.articulation.enabled and not carries and "drawer" in searchable
        else []
    )
    return {
        "program": program,
        "carries": carries,
        "assist_acts": assist_acts,
        "match": None,
        "match_instruction": None,
        "match_reason": None,
        "same_arrangement": None,
        "route": forced_route,
        "route_by_router": None,
        "route_source": "external-capability-router-v2",
        "cached_program": True,
    }


CONFIGS = {
    "g5b": CONFIG,
    "g5c": replace(CONFIG, stage="G5c", place=replace(
        CONFIG.place, vla_carry_first_steps=5, vla_carry_min_progress_m=0.02, vla_carry_open_far_m=0.08)),
}
CONFIGS["g5d"] = replace(CONFIGS["g5c"], stage="G5d", place=replace(
    CONFIGS["g5c"].place, vla_carry_stall_beyond_m=0.10))
CONFIGS["g5e"] = replace(CONFIGS["g5d"], stage="G5e", articulation=ArticulationConfig(enabled=True))
CONFIGS["g5f"] = replace(CONFIGS["g5e"], stage="G5f", move=replace(CONFIGS["g5e"].move, brake=True),
                         place=replace(CONFIGS["g5e"].place, deadline_release_steps=25, reroute_on_stall=False,
                                       vla_carry_first_steps=20, vla_carry_min_progress_m=0.03))
CONFIGS["g5g"] = replace(CONFIGS["g5f"], stage="G5g", place=replace(CONFIGS["g5f"].place, vla_carry_max_steps=40))
CONFIGS["g5h"] = replace(CONFIGS["g5g"], stage="G5h", route=replace(CONFIGS["g5g"].route, catalog=None))
CONFIGS["g5i"] = replace(CONFIGS["g5h"], stage="G5i",
                         verify=replace(CONFIGS["g5h"].verify, hold_min_opening_m=0.003))
# G5j: π0.5 offsets profiled under the official open-gripper settle.
CONFIGS["g5j"] = replace(
    CONFIGS["g5i"], stage="G5j",
    approach=replace(CONFIGS["g5i"].approach,
                     policy_profile="configs/policies/pi05_open_approach_lag15.json"),
    verify=replace(CONFIGS["g5i"].verify,
                   retry_profile="configs/policies/pi05_open_approach.json"),
    place=replace(CONFIGS["g5i"].place,
                  policy_profile="configs/policies/pi05_open_place.json"))
# G5k: let π0.5 try the grasp first, then assist on a late or clearly off-target close.
CONFIGS["g5k"] = replace(
    CONFIGS["g5j"], stage="G5k",
    route=replace(CONFIGS["g5j"].route,
                  vla_first_steps=90, vla_first_close_xy_m=0.08))
# G5j: π0.5's offsets profiled under the official open-gripper settle (build_oracle_profile.py).
CONFIGS["g5j"] = replace(
    CONFIGS["g5i"], stage="G5j",
    approach=replace(CONFIGS["g5i"].approach, policy_profile="configs/policies/pi05_open_approach_lag15.json"),
    verify=replace(CONFIGS["g5i"].verify, retry_profile="configs/policies/pi05_open_approach.json"),
    place=replace(CONFIGS["g5i"].place, policy_profile="configs/policies/pi05_open_place.json"))
# G5k: π0.5 grasps alone first. 90 steps = p95 of its first lift on its own training tasks
# under the open settle (86); 8 cm = the largest hand-to-object offset at its first close there
# (6.9 cm) plus measurement error.
CONFIGS["g5k"] = replace(CONFIGS["g5j"], stage="G5k", route=replace(
    CONFIGS["g5j"].route, vla_first_steps=90, vla_first_close_xy_m=0.08))


def main() -> None:
    args = parse_args()
    cfg = next((c for m, c in CONFIGS.items() if args.mode.startswith(m)), CONFIG)
    started = time.monotonic()
    env = executor = recorder = session = None
    row: dict = {}
    log: dict = {"events": []}
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
            if env.t >= horizon or env.success:
                raise OutOfTime

        adapter = LiberoAdapter(env, None, cfg.move, on_step=on_move)
        if cfg.verify.hold_min_opening_m is not None:
            adapter.hold_opening_m = (cfg.verify.hold_min_opening_m, adapter.hold_opening_m[1])
        home = adapter.proprio().eef_pos.copy()

        from mfm_client import MFMClientSession

        if args.force_route == "direct":
            session = None
            decision = {
                "route": "direct",
                "carries": False,
                "assist_acts": [],
                "program": [],
                "source": "forced_direct",
            }
        elif args.force_route and args.cached_program is not None:
            session = MFMClientSession(cfg.models.perception, max_tokens=1000)
            decision = _external_route_decision(
                args.cached_program,
                env.instruction,
                args.force_route,
                cfg,
            )
        else:
            session = MFMClientSession(cfg.models.perception, max_tokens=1000)
            catalog, catalog_dir = (load_catalog(HERE.parent / cfg.route.catalog) if cfg.route.catalog
                                    else (None, None))
            start_image = adapter.observe(["agentview"], cfg.route.image_px).rgb["agentview"]
            decision = route(session, env.instruction, start_image, catalog, catalog_dir, ProgramConfig(),
                             assist_acts=("drawer",) if cfg.articulation.enabled else ())
            if args.force_route:
                decision["route_by_router"], decision["route"] = decision["route"], args.force_route
        log["route"] = decision
        assist = decision["route"] == "assist" and decision["carries"]
        drawer = decision["route"] == "assist" and not decision["carries"] and bool(decision.get("assist_acts"))
        decision["effective_executor"] = "g5j-forced-assist" if assist or drawer else "direct"

        executor = OpenPiVLAExecutor(VLAConfig(
            port=args.vla_port, resize_size=224, replan_steps=args.vla_replan_steps,
            max_skill_steps=horizon, prompt_mode="task",
        ))

        def run_vla(callback):
            return executor.execute_task(env, max_steps=horizon - env.t, benchmark_stop_on_success=True,
                                         step_callback=callback)

        def record_only(step_env, action, call):
            tracker.update(step_env)
            recorder.record_step(step_env, action=action, mode=f"pi05_{call}", decision=call - 1)

        if drawer:
            try:
                reason = drawer_episode(cfg, env, adapter, session, recorder, horizon,
                                        lambda: run_vla(record_only).reason, log)
            except OutOfTime:
                reason = f"agent motion ended at {env.t}/{horizon}"
        elif not assist:
            reason = run_vla(record_only).reason
        else:
            try:
                reason = assist_episode(args, cfg, env, adapter, session, executor, recorder, tracker,
                                        home, horizon, run_vla, log)
            except OutOfTime:
                reason = f"agent motion ended at {env.t}/{horizon}"
        first = tracker.first_lifted()
        row = {
            "suite": args.suite, "task": args.task, "init": args.init, "mode": args.mode,
            "instruction": env.instruction, "eval_target": eval_target,
            "route": log["route"]["route"], "success": bool(env.success), "steps": env.t,
            "move_steps": adapter.move_steps, "pi05_steps": env.t - adapter.move_steps,
            "first_lifted": first, "first_lifted_is_target": first == eval_target,
            "max_lift_m": {k: round(v, 3) for k, v in tracker.max_lift.items() if v >= 0.01},
            **log,
            "gpt_usage": (
                session.usage()
                if session is not None
                else {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            ),
            "vla_inference_calls": executor.inference_calls,
            "reason": reason, "artifacts": str(args.artifact_dir), "crash": False,
        }
    except Exception as exc:
        row = {
            "suite": args.suite, "task": args.task, "init": args.init, "mode": args.mode,
            "success": False, "steps": env.t if env is not None else 0, **log,
            "reason": f"{type(exc).__name__}: {exc}", "detail": traceback.format_exc(limit=10),
            "artifacts": str(args.artifact_dir), "crash": True,
        }
        if session is not None:
            # Keep token accounting exact even when a later MFM API call or simulator step fails.
            row["gpt_usage"] = session.usage()
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
        handle.write(json.dumps(row, separators=(",", ":"), default=_json_default) + "\n")
    print(json.dumps(row, indent=2, ensure_ascii=False, default=_json_default), flush=True)


def assist_episode(args, cfg, env, adapter, session, executor, recorder, tracker, home, horizon,
                   run_vla, log) -> str:
    events = log["events"]
    references: dict = {}

    wrist = adapter.movable_cameras[0]

    def ground(role: str, out_dir: Path) -> dict:
        """Fixed-view grounding only: no motion."""
        info = ground_grasp_target(adapter, session, cfg.grounding, out_dir, role=role)
        info.pop("world", None)
        crop_image, crop_to_frame = info["_crop_image"], info["_crop_to_frame"]
        if role == "grasp" and info.get("fine", {}).get("bbox"):
            boxed = Image.fromarray(crop_image)
            ImageDraw.Draw(boxed).rectangle([float(v) for v in info["fine"]["bbox"]],
                                            outline=(255, 210, 0), width=3)
            references["grasp"] = {"image": np.asarray(boxed), "name": info["fine"].get("name", "object"),
                                   "evidence": info["fine"].get("visual_evidence", "")}
        box = np.asarray(info["fine"]["bbox"], dtype=np.float64)
        corners = [crop_to_frame(np.array(c)) for c in
                   ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3]))]
        info["_rays"] = [pixel_ray(info["_obs"], adapter.static_cameras[0], uv) for uv in corners]
        return info

    def survey(infos: list[dict]) -> dict | None:
        """One move that puts every grounded target in the wrist camera's view."""
        if cfg.measure.survey_depth_m is None:
            return None
        wrist_obs = adapter.observe([wrist], cfg.grounding.render_px)
        goal = survey_goal(adapter, wrist_obs, wrist, [r for info in infos for r in info["_rays"]],
                           cfg.measure.survey_depth_m, cfg.grounding.render_px)
        if goal is None:
            return {"goal": None}
        return {"goal": goal, "move": adapter.move_to(goal, gripper="open", phase="survey")}

    def measure(role: str, info: dict, out_dir: Path, look: dict | None):
        if look is not None and look.get("goal") is not None and np.linalg.norm(
                adapter.proprio().eef_pos - look["goal"]) > cfg.move.tolerance_m:
            info["back_to_survey"] = adapter.move_to(look["goal"], gripper="open", phase="survey_back")
        mv = measure_in_second_view(
            adapter, session, info["fine"], info["_crop_image"], info["_crop_to_frame"], info["_obs"],
            cfg.grounding.render_px, out_dir, role=role, max_gap_m=cfg.measure.max_ray_gap_m,
            relook_on=cfg.measure.relook_on, use_best=cfg.measure.use_best,
        )
        for key in ("_obs", "_crop_image", "_crop_to_frame", "_rays"):
            info.pop(key, None)
        limit = cfg.measure.accept_gap_m or cfg.measure.max_ray_gap_m
        accepted = bool(mv["visible"] and mv["ray_gap_m"] <= limit)
        top = mv.pop("top", None)
        mv["accepted"] = accepted
        info["second_view"] = mv
        if accepted:
            info["eval_only_nearest_object"] = min(
                tracker.start, key=lambda n: np.linalg.norm(tracker.start[n][:2] - top[:2]))
        return (np.asarray(top) if accepted else None,
                mv.get("support_z") if accepted else None, info)

    place_info = ground("place", recorder.root)
    grasp_info = ground("grasp", recorder.root)
    look = survey([place_info, grasp_info])
    log["survey"] = look
    place_target, _, place_info = measure("place", place_info, recorder.root, look)
    container = _eval_container(env)
    if container is not None and container in tracker.start:
        place_info["eval_only_container"] = container
        if place_target is not None:
            place_info["eval_only_xy_error_m"] = round(float(np.linalg.norm(
                place_target[:2] - tracker.start[container][:2])), 4)
            place_info["eval_only_top_error_m"] = round(float(
                place_target[2] - object_extent(env, container)[2]), 4)
    log["place_grounding"] = place_info

    grasp_profile = json.loads((HERE.parent / cfg.approach.policy_profile).read_text())
    retry_profile = json.loads((HERE.parent / cfg.verify.retry_profile).read_text())
    place_profile = json.loads((HERE.parent / cfg.place.policy_profile).read_text())

    def hand_off(why: str, profile: dict, info: dict | None = None, look: dict | None = None) -> dict:
        """Measure the grasp target (grounded from the start pose unless ``info`` is given)
        and bring the open hand to π0.5's pre-grasp pose."""
        out_dir = recorder.root if not events else recorder.root / f"event{len(events)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        event = {"t": env.t, "why": why}
        if info is None:
            if np.linalg.norm(adapter.proprio().eef_pos - home) > cfg.move.tolerance_m:
                event["home"] = adapter.move_over(home, gripper="open", phase="home_grasp")
            info = ground("grasp", out_dir)
            look = survey([info])
            event["survey"] = look
        estimate, support_z, info = measure("grasp", info, out_dir, look)
        event.update(grounding=info, estimate=estimate, support_z=support_z)
        if estimate is not None:
            goal = estimate + np.asarray(profile["handoff_offset_from_object_top_m"], dtype=np.float64)
            event["approach"] = approach(goal, "approach")
        events.append(event)
        return event

    def approach(goal, phase: str) -> list[dict]:
        """Open hand to ``goal``: to the point above it at the cruise height (the start
        height, or the goal's if higher), then straight down. From a hand above the cruise
        height (survey) the first leg descends diagonally, so the hand never crosses
        lower than the cruise height or out at the edge of the arm's reach; only the
        last leg needs the precise tolerance."""
        return cruise_to(goal, max(float(home[2]), float(goal[2])), "open", phase)

    def cruise_to(goal, cruise: float, gripper: str, phase: str, **kw) -> list[dict]:
        return _cruise_to(adapter, cfg, goal, cruise, gripper, phase, **kw)

    loop = PlaceLoop(cfg, env, adapter, home, horizon, events)
    state = {"monitor": None, "try": None}

    def new_monitor():
        state["monitor"] = GraspMonitor(adapter, settle_m=cfg.place.grasp_settle_m,
                                        settle_steps=cfg.place.grasp_settle_steps)

    def check_try(action):
        trial = state["try"]
        if action[6] > 0 and not trial["closed"] and cfg.route.vla_first_close_xy_m is not None:
            eef = adapter.proprio().eef_pos
            off = float(np.linalg.norm(eef[:2] - trial["target"][:2]))
            trial.setdefault("closes", []).append({"t": env.t, "xy_m": round(off, 4)})
            if off > cfg.route.vla_first_close_xy_m:
                trial["outcome"] = "closed_away"
                raise TryFailed
        trial["closed"] = action[6] > 0
        if env.t >= trial["until"]:
            trial["outcome"] = "timeout"
            raise TryFailed

    def on_vla(step_env, action, call):
        adapter.gripper_command = float(np.sign(action[6]) or adapter.gripper_command)
        tracker.update(step_env)
        recorder.record_step(step_env, action=action, mode=f"pi05_{call}", decision=call - 1)
        if loop.step(action):
            return
        if state["monitor"] is not None and state["monitor"].update(action):
            raise Lifted
        if state["try"] is not None:
            check_try(action)

    def take_over_try():
        """Open and rise, then the usual hand-off to π0.5's pre-grasp pose."""
        lift = adapter.proprio().eef_pos + np.array([0.0, 0.0, cfg.verify.release_rise_m])
        release = adapter.move_to(lift, gripper="open", phase="release_try")
        goal = first["estimate"] + np.asarray(grasp_profile["handoff_offset_from_object_top_m"])
        events.append({"t": env.t, "why": "try_handoff", "release": release,
                       "approach": approach(goal, "approach")})
        new_monitor()

    if cfg.route.vla_first_steps is None:
        first = hand_off("start", grasp_profile, grasp_info, look)
    else:
        estimate, support_z, grasp_info = measure("grasp", grasp_info, recorder.root, look)
        first = {"t": env.t, "why": "start", "grounding": grasp_info, "estimate": estimate,
                 "support_z": support_z}
        events.append(first)
        if estimate is not None and place_target is not None:
            first["home"] = adapter.move_to(home, gripper="open", phase="home_try")
            state["try"] = {"t": env.t, "why": "vla_try", "until": env.t + cfg.route.vla_first_steps,
                            "target": estimate, "closed": False}
            events.append(state["try"])
    grasp = first
    dense_captioner = None
    if args.dense_caption and first["estimate"] is not None:
        dense_captioner = GroundedDenseV3Captioner(
            grasp_name=(
                first.get("grounding", {}).get("fine", {}).get("name")
                or "target object"
            ),
            grasp_pos=first["estimate"],
            place_name=(
                place_info.get("fine", {}).get("name")
                or "destination"
            ),
            place_pos=place_target,
            holding_fn=adapter.holding,
        )
        executor.prompt_builder = dense_captioner
        log["dense_caption_history"] = dense_captioner.history
    if first["estimate"] is not None and place_target is not None:
        new_monitor()
    regrasps = redo = 0
    reason = ""
    pending = None
    while env.t < horizon and not env.success:
        carry_now = False
        try:
            reason = run_vla(on_vla).reason
            break
        except TryFailed:
            trial, state["try"] = state["try"], None
            trial["t_end"] = env.t
            take_over_try()
            continue
        except Lifted:
            trial, state["try"] = state["try"], None
            verdict = check_grasp(adapter, session, env.instruction, cfg.verify.check_px,
                                  reference=references.get("grasp"))
            verdict["eval_only_lifted"] = [n for n, v in tracker.max_lift.items() if v >= 0.02]
            events.append({"t": env.t, "why": "grasp_check", "verdict": verdict})
            if not verdict["correct"] and trial is not None:
                trial.update(outcome="wrong_lift", t_end=env.t)
                take_over_try()
                continue
            if trial is not None:
                trial.update(outcome="lifted", t_end=env.t)
            if not verdict["correct"] and regrasps < cfg.verify.max_regrasp:
                regrasps += 1
                lift = adapter.proprio().eef_pos + np.array([0.0, 0.0, cfg.verify.release_rise_m])
                release = adapter.move_to(lift, gripper="open", phase="release_wrong")
                goal = grasp["estimate"] + np.asarray(retry_profile["handoff_offset_from_object_top_m"])
                events.append({"t": env.t, "why": "regrasp", "release": release,
                               "approach": approach(goal, "regrasp")})
                new_monitor()
                continue
            if not verdict["correct"]:
                state["monitor"] = None
                continue
            hang = state["monitor"].close_z - grasp["support_z"]
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
        if carry_now:
            carry = loop.carry()
            if not carry["still_holding"] and redo < cfg.place.max_redo_after_drop:
                redo += 1
                open_in_place(adapter, env, horizon, 5)
                grasp = hand_off("dropped", grasp_profile)
                if grasp["estimate"] is not None:
                    new_monitor()
                else:
                    state["monitor"] = None
                continue
            loop.hand_back()
    log.update(regrasps=regrasps, redo=redo)
    return reason or f"ended at {env.t}/{horizon}"


def drawer_episode(cfg, env, adapter, session, recorder, horizon: int, run_vla, log) -> str:
    """G5e: measure the handle in two views, measure it again from close in front, then
    open the drawer with the VLA's own motion; the VLA gets whatever time is left."""
    a = cfg.articulation
    profile = json.loads((HERE.parent / a.policy_profile).read_text())
    wrist = adapter.movable_cameras[0]
    event = {"t": env.t, "why": "drawer", "attempts": []}
    log["events"].append(event)
    handle = ground_handle(adapter, session, cfg.grounding, a.render_px, recorder.root)
    event["ground"] = _public(handle)
    if not handle["accepted"] or handle.get("motion") != "pull_open":
        event["handed_to_vla"] = "not grounded" if not handle["accepted"] else f"motion {handle.get('motion')}"
        return run_vla()
    wrist_obs = adapter.observe([wrist], cfg.grounding.render_px)
    goal = survey_goal(adapter, wrist_obs, wrist, handle["_rays"], cfg.measure.survey_depth_m,
                       cfg.grounding.render_px)
    if goal is not None:
        event["survey"] = adapter.move_to(goal, gripper="open", phase="survey")
    far = locate_in_wrist(adapter, session, handle, a.render_px, recorder.root, "survey")
    event["far"] = _public(far)
    if not far["accepted"] or far["ray_gap_m"] > a.accept_gap_m:
        event["handed_to_vla"] = "not measured"
        return run_vla()
    point, outward = far["point"], far["outward_xy"]
    for attempt in range(a.max_attempts):
        if horizon - env.t < a.min_steps_for_attempt:
            break
        record = {"t": env.t}
        event["attempts"].append(record)
        poses = drawer_poses(profile, point, outward, a.front_clearance_m)
        eef = adapter.proprio().eef_pos
        front = poses["front"]
        record["front"] = [
            adapter.move_to(np.array([front[0], front[1], max(eef[2], front[2])]), gripper="open",
                            phase="drawer_traverse", tolerance_m=cfg.measure.waypoint_tolerance_m,
                            quat_xyzw=poses["quat"]),
            adapter.move_to(front, gripper="open", phase="drawer_front", quat_xyzw=poses["quat"]),
        ]
        close = locate_in_wrist(adapter, session, handle, a.render_px, recorder.root, f"close{attempt}")
        record["close"] = _public(close)
        if (close["accepted"] and close["ray_gap_m"] <= a.accept_gap_m
                and np.linalg.norm(close["point"] - point) < a.close_max_shift_m):
            point, outward = close["point"], close["outward_xy"]
            poses = drawer_poses(profile, point, outward, a.front_clearance_m)
            record["used"] = "close"
        else:
            record["used"] = "far"
        quat, out = poses["quat"], poses["out"]
        record["hook"] = [
            adapter.move_to(poses["front"], gripper="open", phase="drawer_front2", quat_xyzw=quat),
            adapter.move_to(poses["above"], gripper="open", phase="drawer_in", quat_xyzw=quat),
            adapter.move_to(poses["pressed"], gripper="open", phase="drawer_press", quat_xyzw=quat),
            adapter.move_to(poses["pressed"] + out * a.pull_travel_m, gripper="open", phase="drawer_pull",
                            quat_xyzw=quat, max_action_norm=a.pull_max_action_norm),
        ]
        record["check"] = check_open(adapter, session, handle, cfg.verify.check_px)
        if record["check"]["done"]:
            break
        hand = adapter.proprio().eef_pos
        record["back_off"] = adapter.move_to(hand + out * 0.04 + np.array([0.0, 0.0, 0.06]), gripper="open",
                                             phase="drawer_back_off", quat_xyzw=quat)
    event["handed_to_vla"] = "after harness"
    return run_vla()


class PlaceLoop:
    """After a confirmed grasp: π0.5 carries first under a watch and the agent takes the
    carry over when π0.5 makes no progress; once over the target π0.5 places, with release
    fallbacks. Needs no GPT, so offline replays run it unchanged."""

    def __init__(self, cfg, env, adapter, home, horizon: int, events: list):
        self.cfg, self.env, self.adapter, self.home, self.horizon, self.events = (
            cfg, env, adapter, home, horizon, events)
        self.pending = None
        self.state = {"carried": False, "handback_t": None, "opened": False, "watch": None,
                      "goal": None, "deadline": False}

    def horizontal(self, goal) -> float:
        return float(np.linalg.norm(self.adapter.proprio().eef_pos[:2] - goal[:2]))

    def begin(self, pending: dict) -> bool:
        """Start the watch on π0.5's own carry; True if the agent should carry right away."""
        self.pending = pending
        self.state["goal"] = pending["goal"]
        if self.cfg.place.vla_carry_first_steps is None:
            return True
        watch = {"t": self.env.t, "why": "vla_carry", "goal": pending["goal"],
                 "d0": self.horizontal(pending["goal"])}
        self.events.append(watch)
        self.state["watch"] = watch
        return False

    def step(self, action) -> bool:
        """Per π0.5 step; raises Release or CarryNeeded. False if no grasp is being placed."""
        p, state, t = self.cfg.place, self.state, self.env.t
        if ((state["carried"] or state["watch"] is not None) and p.deadline_release_steps is not None
                and not state["deadline"] and self.horizon - t <= p.deadline_release_steps
                and self.adapter.holding() and self.horizontal(state["goal"]) <= p.deadline_release_near_m):
            state["deadline"] = True
            raise Release
        if state["carried"]:
            if action[6] < 0:
                state["opened"] = True
            if (p.release_wait_steps is not None and not state["opened"]
                    and t - state["handback_t"] >= p.release_wait_steps):
                raise Release
            return True
        watch = state["watch"]
        if watch is None:
            return False
        distance = self.horizontal(watch["goal"])
        history = watch.setdefault("_d", [])
        history.append(distance)
        elapsed = t - watch["t"]

        def take_over(outcome: str):
            watch["outcome"], watch["t_end"] = outcome, t
            watch.pop("_d", None)
            raise CarryNeeded

        if action[6] < 0 and p.vla_carry_open_far_m is not None and distance > p.vla_carry_open_far_m:
            take_over("opened_far")
        if action[6] < 0 or distance <= p.vla_carry_arrive_m:
            watch["outcome"] = "opened" if action[6] < 0 else "arrived"
            watch["t_end"] = t
            watch.pop("_d", None)
            state.update(watch=None, carried=True, handback_t=t, opened=action[6] < 0)
        elif p.vla_carry_min_progress_m is None:
            if elapsed >= p.vla_carry_first_steps and distance > p.vla_carry_progress * watch["d0"]:
                take_over("no_progress")
        elif elapsed == p.vla_carry_first_steps and watch["d0"] - distance < p.vla_carry_min_progress_m:
            take_over("no_progress")
        elif (p.vla_carry_max_steps is not None and elapsed >= p.vla_carry_max_steps
              and (p.vla_carry_timeout_progress_m is None or len(history) <= p.vla_carry_stall_window
                   or history[-p.vla_carry_stall_window - 1] - distance < p.vla_carry_timeout_progress_m)):
            take_over("timeout")
        elif (elapsed > p.vla_carry_first_steps and len(history) > p.vla_carry_stall_window
              and history[-p.vla_carry_stall_window - 1] - distance < p.vla_carry_stall_m
              and (p.vla_carry_stall_beyond_m is None or distance > p.vla_carry_stall_beyond_m)):
            take_over("stalled")
        return True

    def release(self) -> None:
        steps = open_in_place(self.adapter, self.env, self.horizon, 10)
        self.events.append({"t": self.env.t, "why": "deadline_release" if self.state["deadline"] else "release",
                            "steps": steps})
        self.state.update(watch=None, carried=True, opened=True)

    def carry(self) -> dict:
        self.state["watch"] = None
        self.adapter.gripper_command = GRIPPER_COMMAND["close"]
        goal, hang = self.pending["goal"], self.pending["hang"]
        carry = {"t": self.env.t, "why": "carry", "hang_m": round(float(hang), 4), "goal": goal,
                 "moves": _carry_to(self.adapter, self.cfg, self.home, goal, self.pending["cruise"], "carry")}
        carry["still_holding"] = self.adapter.holding()
        self.events.append(carry)
        return carry

    def hand_back(self) -> None:
        self.state.update(carried=True, handback_t=self.env.t)


def _cruise_to(adapter, cfg, goal, cruise: float, gripper: str, phase: str, **kw) -> list[dict]:
    """Rise in place if below ``cruise``; then straight to the point above ``goal`` at
    ``cruise`` (descending diagonally if the hand is higher), loosely; then down."""
    moves = []
    eef = adapter.proprio().eef_pos
    loose = cfg.measure.waypoint_tolerance_m
    if eef[2] < cruise - cfg.move.tolerance_m:
        moves.append(adapter.move_to(np.array([eef[0], eef[1], cruise]), gripper=gripper,
                                     phase=f"{phase}_rise", tolerance_m=loose, **kw))
    if not (moves and moves[-1]["stalled"]):
        moves.append(adapter.move_to(np.array([goal[0], goal[1], cruise]), gripper=gripper,
                                     phase=f"{phase}_traverse", tolerance_m=loose, **kw))
    if not moves[-1]["stalled"]:
        moves.append(adapter.move_to(goal, gripper=gripper, phase=f"{phase}_descend", **kw))
    return moves


def _carry_to(adapter, cfg, home, goal, cruise: float, phase: str) -> list[dict]:
    p = cfg.place
    kw = dict(max_action_change=p.carry_max_action_change, max_action_norm=p.carry_max_action_norm)
    moves = _cruise_to(adapter, cfg, goal, cruise, "hold", phase, **kw)
    if (p.reroute_on_stall and moves[-1]["stalled"]
            and moves[-1]["residual_m"] > p.reroute_min_residual_m):
        moves += _cruise_to(adapter, cfg, home, max(cruise, float(home[2])), "hold", f"{phase}_via_home", **kw)
        moves += _cruise_to(adapter, cfg, goal, cruise, "hold", f"{phase}_reroute", **kw)
    return moves


def _public(info: dict) -> dict:
    return {k: v for k, v in info.items() if not k.startswith("_")}


def open_in_place(adapter, env, horizon: int, steps: int) -> int:
    """Open the gripper without moving the hand."""
    done = 0
    while done < steps and env.t < horizon and not env.success:
        action = np.zeros(7)
        action[6] = GRIPPER_COMMAND["open"]
        adapter.apply(action, "move_open")
        adapter.move_steps += 1
        done += 1
    return done


def _json_default(value):
    return value.tolist() if isinstance(value, np.ndarray) else str(value)


if __name__ == "__main__":
    main()
