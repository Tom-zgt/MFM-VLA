"""G3b: GPT task program + per-step G3a hand-offs + π0.5, one episode per call.

GPT splits the instruction into single-skill steps, each with a short VLA
sub-instruction. A ``carry`` step is a G3a episode on its sub-instruction
(measure place and grasp targets, ``move_over`` to π0.5's pre-grasp pose,
carry to its pre-release pose after the grasp, π0.5 releases). An ``act`` step
(open, close, turn on, push...) is π0.5 alone on its sub-instruction. Steps end
on proprioception (carry) or on a GPT completion check (act). Simulator state
is read only for evaluation logging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

import libero_compat  # noqa: F401

from eval_preposition_pick import LiftTracker
from eval_vla_only import SUITE_HORIZONS
from generic_adapter import GRIPPER_COMMAND, GraspMonitor, LiberoAdapter
from generic_config import (AgentConfig, ApproachConfig, MeasureConfig, MoveConfig, PlaceConfig,
                            ProgramConfig)
from generic_grounding import ground_grasp_target
from generic_multiview import measure_in_second_view
from harness import TaskEnv
from rollout_recorder import RolloutRecorder, default_staging_root
from vla_executor import OpenPiVLAExecutor, VLAConfig


HERE = Path(__file__).resolve().parent
WRIST = "robot0_eye_in_hand"

PROGRAM_PROMPT = """You plan for a robot arm with a parallel gripper, seen from a fixed camera.
Instruction: "{instruction}"

Split the instruction into the shortest ordered list of single-skill steps.
Each step is one of:
- "carry": pick up one object and put it into, onto, or next to something.
- "act": operate one thing without carrying it (open, close, turn on or off, push...).
Order the steps so each is physically possible when its turn comes. Do not add
steps the instruction does not ask for.
For each step write "instruction": a short imperative sentence that a robot policy
trained on instructions like the one above would understand, reusing the
instruction's own words and naming objects exactly as it does. For a carry step also
set "onto_object": true if the object goes into or onto another object, false if it
goes next to something or into an empty area.
If the instruction is already a single step, return it unchanged as one step.
Return JSON only: {{"steps": [{{"skill": "carry" or "act", "instruction": "...", "onto_object": true or false}}]}}
"""

CHECK_PROMPT = """A robot arm is carrying out one step of a task.
Task: "{instruction}"
Current step: "{step}"
The first image is a fixed camera; the second is the camera on the robot's wrist.
Has the current step been completed? Judge only this step.
Return JSON only: {{"done": true or false, "evidence": "<one short sentence>"}}
"""


class StepView:
    """The adapter as seen by grounding for one step: same robot, the step's instruction."""

    def __init__(self, adapter: LiberoAdapter, instruction: str):
        self._adapter = adapter
        self.instruction = instruction

    def __getattr__(self, name):
        return getattr(self._adapter, name)


class Handover(Exception):
    pass


class StepEnd(Exception):
    pass


class ReleaseMonitor:
    """Grasped and lifted, then opened, then a short retreat: the carry step is over.

    Opening before a lift is a regrasp, not a release.
    """

    def __init__(self, adapter: LiberoAdapter, cfg: ProgramConfig, settle_m: float | None):
        self.adapter, self.cfg = adapter, cfg
        self.grasp = GraspMonitor(adapter, close_steps=cfg.hold_steps_before_release, settle_m=settle_m)
        self.lifted_at = None
        self.released_at = None

    def update(self, action: np.ndarray) -> bool:
        if self.lifted_at is None:
            if self.grasp.update(action):
                self.lifted_at = self.adapter.t
            return False
        if self.released_at is None:
            if action[6] < 0:
                self.released_at = self.adapter.t
            return False
        return self.adapter.t - self.released_at >= self.cfg.retreat_steps


def parse_program(reply: dict, cfg: ProgramConfig) -> list[dict]:
    steps = reply.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"program has no steps: {reply}")
    program = []
    for step in steps[: cfg.max_steps_in_program]:
        skill = step.get("skill")
        text = step.get("instruction")
        if skill not in ("carry", "act") or not isinstance(text, str) or not text.strip():
            raise ValueError(f"bad program step: {step}")
        program.append({"skill": skill, "instruction": text.strip(),
                        "onto_object": bool(step.get("onto_object", skill == "carry")),
                        "act_kind": step.get("act_kind", "other") if skill == "act" else None})
    return program


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--mode", default="g3b")
    parser.add_argument("--vla-port", type=int, default=8000)
    parser.add_argument("--vla-replan-steps", type=int, default=5)
    parser.add_argument("--no-move", action="store_true",
                        help="ablation: program + sub-instructions only, no move_over hand-offs")
    parser.add_argument("--vla-prompt", choices=("step", "original"), default="step",
                        help="π0.5 gets the step's sub-instruction or the original instruction")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AgentConfig(stage="G3b-prog" if args.no_move else "G3b",
                      approach=ApproachConfig(reference="policy"),
                      move=MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m),
                      measure=MeasureConfig(relook_on="fixed_ray", from_home=True),
                      place=PlaceConfig(policy_profile="configs/policies/pi05_libero_place_v2.json",
                                        grasp_settle_m=0.001))
    pcfg = ProgramConfig()
    started = time.monotonic()
    env = executor = recorder = session = None
    row: dict = {}
    steps_log: list[dict] = []
    program = None
    try:
        horizon = SUITE_HORIZONS.get(args.suite) or SUITE_HORIZONS[args.suite.rsplit("_", 1)[0]]
        env = TaskEnv(args.suite, args.task, img_size=128, seed=args.init)
        env.allow_goal_shortcut = False
        env.reset(args.init)
        recorder = RolloutRecorder(args.artifact_dir, staging_root=default_staging_root())
        recorder.begin(
            {
                "suite": args.suite, "task": args.task, "init": args.init,
                "instruction": env.instruction, "executor": "pi05_libero",
                "experiment": f"generic/{args.mode}",
                "config": {**cfg.to_dict(), "program": pcfg.__dict__},
            },
            env,
        )
        tracker = LiftTracker(env)

        def on_move(action, mode):
            tracker.update(env)
            recorder.record_step(env, action=action, mode=mode, decision=0)

        adapter = LiberoAdapter(env, None, cfg.move, on_step=on_move)

        from mfm_client import MFMClientSession

        session = MFMClientSession(cfg.models.planning, max_tokens=1000)
        plan_obs = adapter.observe(["agentview"], pcfg.plan_px)
        program = parse_program(
            session.decide_json(PROGRAM_PROMPT.format(instruction=env.instruction),
                                {"agentview": plan_obs.rgb["agentview"]}),
            pcfg,
        )

        executor = OpenPiVLAExecutor(VLAConfig(
            port=args.vla_port, resize_size=224, replan_steps=args.vla_replan_steps,
            max_skill_steps=horizon, prompt_mode="task",
        ))
        grasp_profile = json.loads((HERE.parent / cfg.approach.policy_profile).read_text())
        place_profile = json.loads((HERE.parent / cfg.place.policy_profile).read_text())

        def nearest(point):
            if point is None:
                return None
            return min(tracker.start, key=lambda n: np.linalg.norm(tracker.start[n][:2] - point[:2]))

        def locate(view, role, out_dir):
            homing = None
            if cfg.measure.from_home and np.linalg.norm(adapter.proprio().eef_pos - home) > cfg.move.tolerance_m:
                homing = adapter.move_over(home, gripper="hold", phase=f"home_{role}")
            info = ground_grasp_target(view, session, cfg.grounding, out_dir, role=role)
            if homing is not None:
                info["home"] = homing
            frame_obs = info.pop("_obs")
            crop_image = info.pop("_crop_image")
            crop_to_frame = info.pop("_crop_to_frame")
            info.pop("world")
            mv = measure_in_second_view(
                view, session, info["fine"], crop_image, crop_to_frame, frame_obs,
                cfg.grounding.render_px, out_dir, role=role, max_gap_m=cfg.measure.max_ray_gap_m,
                relook_on=cfg.measure.relook_on,
            )
            accepted = mv["visible"] and mv["ray_gap_m"] <= cfg.measure.max_ray_gap_m
            top = mv.pop("top", None)
            mv["accepted"] = bool(accepted)
            info["second_view"] = mv
            point, support = (top, mv["support_z"]) if accepted else (None, None)
            info["eval_only_nearest_object"] = nearest(point)
            return point, support, info

        def step_done_by_check(step) -> dict:
            obs = adapter.observe(["agentview", WRIST], pcfg.check_px)
            reply = session.decide_json(
                CHECK_PROMPT.format(instruction=env.instruction, step=step["instruction"]),
                {"agentview": obs.rgb["agentview"], WRIST: obs.rgb[WRIST]},
            )
            return {"t": env.t, "done": bool(reply.get("done")), "evidence": reply.get("evidence")}

        home = adapter.proprio().eef_pos.copy()
        for index, step in enumerate(program):
            if env.success or env.t >= horizon:
                break
            log = {"index": index, **step, "start": env.t}
            steps_log.append(log)
            if index and not args.no_move:
                log["home"] = adapter.move_over(home, gripper="hold", phase=f"home{index}")
            prompt = step["instruction"] if args.vla_prompt == "step" else env.instruction
            step_dir = recorder.root / f"step{index}"
            view = StepView(adapter, step["instruction"])
            limit = min(horizon, env.t + pcfg.max_step_steps)
            monitor = release = None
            place_target = None
            if step["skill"] == "carry":
                release = ReleaseMonitor(adapter, pcfg, cfg.place.grasp_settle_m)
                if not args.no_move:
                    if step["onto_object"]:
                        place_target, _, log["place_grounding"] = locate(view, "place", step_dir)
                    estimate, support_z, log["grounding"] = locate(view, "grasp", step_dir)
                    if estimate is not None:
                        start_pose = np.asarray(estimate) + np.asarray(
                            grasp_profile["handoff_offset_from_object_top_m"])
                        log["approach"] = adapter.move_over(start_pose, gripper="open", phase=f"approach{index}")
                        if place_target is not None:
                            monitor = GraspMonitor(adapter, settle_m=cfg.place.grasp_settle_m)
                    else:
                        log["fallback"] = "unmeasured grasp target: π0.5 from the current pose"
            state = {"carried": False, "since_check": 0}

            def on_vla(step_env, action, call):
                adapter.gripper_command = float(np.sign(action[6]) or adapter.gripper_command)
                tracker.update(step_env)
                recorder.record_step(step_env, action=action, mode=f"pi05_s{index}_{call}", decision=call - 1)
                if release is not None and release.update(action):
                    raise StepEnd
                if monitor is not None and not state["carried"] and monitor.update(action):
                    raise Handover
                if step["skill"] == "act":
                    state["since_check"] += 1
                    if state["since_check"] >= pcfg.check_every:
                        raise StepEnd

            checks = []
            outcome = "budget"
            while env.t < limit and not env.success:
                try:
                    executor.execute_prompt(env, prompt, limit - env.t,
                                            benchmark_stop_on_success=True, step_callback=on_vla)
                    outcome = "success" if env.success else "budget"
                    break
                except Handover:
                    hang = monitor.close_z - support_z
                    offset = np.asarray(place_profile["place_offset_from_target_top_m"])
                    goal = np.array([
                        place_target[0] + offset[0], place_target[1] + offset[1],
                        place_target[2] + place_profile["held_bottom_above_target_top_m"] + hang,
                    ])
                    state["carried"] = True
                    crossing = place_profile.get("held_bottom_above_target_top_crossing_m")
                    traverse_z = None if crossing is None else float(place_target[2] + crossing + hang)
                    log["carry"] = {
                        "step": env.t, "hang_m": round(float(hang), 4), "goal": np.round(goal, 4).tolist(),
                        "traverse_z": traverse_z,
                        "moves": adapter.move_over(goal, gripper="hold", phase=f"carry{index}",
                                                   traverse_z=traverse_z,
                                                   max_action_change=cfg.place.carry_max_action_change,
                                                   max_action_norm=cfg.place.carry_max_action_norm),
                        "still_holding": adapter.holding(),
                    }
                except StepEnd:
                    if step["skill"] == "carry":
                        outcome = "released"
                        break
                    state["since_check"] = 0
                    checks.append(step_done_by_check(step))
                    if checks[-1]["done"]:
                        outcome = "verified"
                        break
            log.update({"end": env.t, "outcome": outcome, "checks": checks})
            if release is not None:
                log["lifted_at"], log["released_at"] = release.lifted_at, release.released_at

        first = tracker.first_lifted()
        row = {
            "suite": args.suite, "task": args.task, "init": args.init, "mode": args.mode,
            "instruction": env.instruction, "vla_prompt": args.vla_prompt,
            "program": program, "steps_log": steps_log,
            "success": bool(env.success), "steps": env.t,
            "move_steps": adapter.move_steps, "pi05_steps": env.t - adapter.move_steps,
            "first_lifted": first,
            "max_lift_m": {k: round(v, 3) for k, v in tracker.max_lift.items() if v >= 0.01},
            "gpt_usage": session.usage(), "vla_inference_calls": executor.inference_calls,
            "reason": "success" if env.success else f"ended at {env.t}/{horizon}",
            "artifacts": str(args.artifact_dir), "crash": False,
        }
    except Exception as exc:
        row = {
            "suite": args.suite, "task": args.task, "init": args.init, "mode": args.mode,
            "success": False, "steps": env.t if env is not None else 0,
            "program": program, "steps_log": steps_log,
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
