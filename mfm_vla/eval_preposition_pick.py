"""Does π0.5 grasp the instructed object once the gripper is already above it?

Diagnostic for the "global transport by harness, local grasp by π0.5" split.

* ``preposition``: hover target from simulator truth (oracle grounding), the
  upper bound with grounding error removed.
* ``gpt_preposition``: GPT marks the object's footprint centre in the real
  agentview RGB; the calibrated pixel ray is intersected with the touched
  support plane. Simulator truth is only logged, for measuring grounding error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

import libero_compat  # noqa: F401

from eval_vla_only import SUITE_HORIZONS
from harness import TaskEnv
from rollout_recorder import RolloutRecorder, default_staging_root
from vla_executor import OpenPiVLAExecutor, VLAConfig

LIFTED_M = 0.05
# Fixed agentview workspace crop (u0, v0, u1, v1) as frame fractions; the camera is static.
WORKSPACE_CROP = (0.08, 0.30, 0.98, 0.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--mode", choices=("baseline", "preposition", "gpt_preposition"), default="preposition")
    parser.add_argument("--hover-m", type=float, default=0.10)
    parser.add_argument("--plane-eef-z", type=float, default=0.0094,
                        help="end-effector z at which the open gripper stalls on the support plane")
    parser.add_argument("--gpt-image-px", type=int, default=1024)
    parser.add_argument("--gpt-model", default=os.environ.get("MFM_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o")))
    parser.add_argument("--tolerance-m", type=float, default=0.01)
    parser.add_argument("--vla-port", type=int, default=8000)
    parser.add_argument("--vla-replan-steps", type=int, default=5)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _inner(env: TaskEnv):
    return env._env.env


def _target_object(env: TaskEnv) -> str:
    for goal in _inner(env).parsed_problem["goal_state"]:
        if goal[0].lower() in ("in", "on"):
            return goal[1]
    raise ValueError("task has no in/on goal")


def _positions(env: TaskEnv) -> dict[str, np.ndarray]:
    inner = _inner(env)
    return {
        name: np.array(inner.object_states_dict[name].get_geom_state()["pos"], copy=True)
        for name in inner.objects_dict
    }


class LiftTracker:
    def __init__(self, env: TaskEnv):
        self.start = _positions(env)
        self.max_lift = {name: 0.0 for name in self.start}
        self.first_lift_step: dict[str, int] = {}

    def update(self, env: TaskEnv) -> None:
        for name, pos in _positions(env).items():
            lift = float(pos[2] - self.start[name][2])
            self.max_lift[name] = max(self.max_lift[name], lift)
            if lift >= LIFTED_M and name not in self.first_lift_step:
                self.first_lift_step[name] = env.t

    def first_lifted(self) -> str | None:
        if not self.first_lift_step:
            return None
        return min(self.first_lift_step, key=self.first_lift_step.get)


GROUNDING_PROMPT = """You are the perception module of a robot arm.
The image is from a fixed third-person camera, {width}x{height} pixels, origin at the top-left,
u to the right, v downward.

Instruction: "{instruction}"

Find the single object the robot must pick up. Several grocery items look alike and they
are NOT in their usual places, so identify the object by its packaging and printed label,
not by where such an item normally stands.

Return the pixel where that object's footprint centre touches the floor: the centre of its
base, not its top, label, or cap. Reply with JSON only:
{{"object": "<name>", "visual_evidence": "<what identifies it>",
  "pixel_base_center": [u, v], "bbox": [u_min, v_min, u_max, v_max], "confidence": <0-1>}}"""


def _ray_plane(obs, cam: str, uv, plane_z: float) -> np.ndarray:
    K = obs.intrinsics[cam]
    direction = obs.cam_to_world[cam][:3, :3] @ np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    origin = obs.cam_to_world[cam][:3, 3]
    if abs(direction[2]) < 1e-6:
        raise ValueError("pixel ray is parallel to the support plane")
    scale = (plane_z - origin[2]) / direction[2]
    if scale <= 0:
        raise ValueError("pixel ray does not hit the support plane")
    return origin + scale * direction


def _gpt_ground(env: TaskEnv, args, out_dir: Path) -> tuple[np.ndarray, dict]:
    from mfm_client import MFMClientSession
    from PIL import Image, ImageDraw

    cam = "agentview"
    obs = env.observe(cameras=[cam], size=args.gpt_image_px)
    u0, v0, u1, v1 = (np.array(WORKSPACE_CROP) * args.gpt_image_px).astype(int)
    image = obs.rgb[cam][v0:v1, u0:u1]
    session = MFMClientSession(args.gpt_model, max_tokens=800)
    asked = time.monotonic()
    try:
        reply = session.decide_json(
            GROUNDING_PROMPT.format(
                width=image.shape[1], height=image.shape[0], instruction=env.instruction
            ),
            {cam: image},
        )
    finally:
        usage = session.usage()
        session.close()
    local = np.asarray(reply["pixel_base_center"], dtype=np.float64)
    uv = local + np.array([u0, v0], dtype=np.float64)
    world = _ray_plane(obs, cam, uv, args.plane_eef_z)

    marked = Image.fromarray(image)
    draw = ImageDraw.Draw(marked)
    if len(reply.get("bbox") or []) == 4:
        draw.rectangle([float(v) for v in reply["bbox"]], outline=(255, 200, 0), width=2)
    draw.ellipse([local[0] - 7, local[1] - 7, local[0] + 7, local[1] + 7], outline=(0, 255, 0), width=3)
    out_dir.mkdir(parents=True, exist_ok=True)
    marked.save(out_dir / "gpt_grounding.png")
    return world, {
        "reply": reply, "pixel": uv.round(1).tolist(), "world_xy": world[:2].round(4).tolist(),
        "latency_s": round(time.monotonic() - asked, 2), "usage": usage,
    }


def _servo_eef(env, recorder, tracker, goal, *, phase, tolerance_m, max_steps=80):
    for _ in range(max_steps):
        error = goal - env.proprio().eef_pos
        if np.linalg.norm(error) <= tolerance_m:
            break
        action = np.zeros(7, dtype=np.float64)
        action[:3] = np.clip(1.6 * error / env.action_pos_scale, -1.0, 1.0)
        action[6] = -1.0
        env.step(action)
        tracker.update(env)
        recorder.record_step(env, action=action, mode=f"oracle_preposition_{phase}", decision=0)
    residual = goal - env.proprio().eef_pos
    return {
        "phase": phase,
        "goal": goal.round(4).tolist(),
        "residual_m": round(float(np.linalg.norm(residual)), 4),
        "steps_after": env.t,
    }


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    env = executor = recorder = None
    setup: list[dict] = []
    tracker = None
    row: dict = {}
    try:
        horizon = SUITE_HORIZONS.get(args.suite) or SUITE_HORIZONS[args.suite.rsplit("_", 1)[0]]
        env = TaskEnv(args.suite, args.task, img_size=128, seed=args.init)
        env.allow_goal_shortcut = False
        env.reset(args.init)
        target = _target_object(env)
        recorder = RolloutRecorder(args.artifact_dir, staging_root=default_staging_root())
        recorder.begin(
            {
                "suite": args.suite, "task": args.task, "init": args.init,
                "instruction": env.instruction, "executor": "pi05_libero",
                "experiment": f"oracle_preposition_pick/{args.mode}",
                "target_object": target, "hover_m": args.hover_m,
            },
            env,
        )
        tracker = LiftTracker(env)
        grounding = None
        if args.mode in ("preposition", "gpt_preposition"):
            obj = tracker.start[target]
            if args.mode == "preposition":
                hover = obj + np.array([0.0, 0.0, args.hover_m])
            else:
                estimate, grounding = _gpt_ground(env, args, recorder.root)
                grounding["eval_only_xy_error_m"] = round(float(np.linalg.norm(estimate[:2] - obj[:2])), 4)
                grounding["eval_only_nearest_object"] = min(
                    tracker.start, key=lambda n: np.linalg.norm(tracker.start[n][:2] - estimate[:2])
                )
                hover = np.array([estimate[0], estimate[1], args.plane_eef_z + args.hover_m + 0.03])
            eef = env.proprio().eef_pos
            transit = np.array([hover[0], hover[1], max(eef[2], hover[2])])
            setup.append(_servo_eef(env, recorder, tracker, transit, phase="transit", tolerance_m=args.tolerance_m))
            setup.append(_servo_eef(env, recorder, tracker, hover, phase="descend", tolerance_m=args.tolerance_m))
        handoff_step = env.t
        eef_xy_to_target = float(np.linalg.norm(env.proprio().eef_pos[:2] - _positions(env)[target][:2]))

        executor = OpenPiVLAExecutor(VLAConfig(
            port=args.vla_port, resize_size=224, replan_steps=args.vla_replan_steps,
            max_skill_steps=horizon, prompt_mode="task",
        ))

        def on_step(step_env, action, call):
            tracker.update(step_env)
            recorder.record_step(step_env, action=action, mode=f"pi05_{call}", decision=call - 1)

        result = executor.execute_task(
            env, max_steps=horizon - env.t, benchmark_stop_on_success=True, step_callback=on_step,
        )
        first = tracker.first_lifted()
        row = {
            "suite": args.suite, "task": args.task, "init": args.init, "mode": args.mode,
            "instruction": env.instruction, "target_object": target,
            "success": bool(env.success), "steps": env.t, "handoff_step": handoff_step,
            "handoff_eef_xy_to_target_m": round(eef_xy_to_target, 4),
            "first_lifted": first,
            "first_lifted_is_target": first == target,
            "first_lift_step": tracker.first_lift_step,
            "max_lift_m": {k: round(v, 3) for k, v in tracker.max_lift.items() if v >= 0.01},
            "setup": setup, "grounding": grounding, "reason": result.reason,
            "vla_inference_calls": executor.inference_calls,
            "artifacts": str(args.artifact_dir), "crash": False,
        }
    except Exception as exc:
        row = {
            "suite": args.suite, "task": args.task, "init": args.init, "mode": args.mode,
            "success": False, "steps": env.t if env is not None else 0, "setup": setup,
            "reason": f"{type(exc).__name__}: {exc}", "detail": traceback.format_exc(limit=10),
            "artifacts": str(args.artifact_dir), "crash": True,
        }
    finally:
        row["duration_s"] = round(time.monotonic() - started, 2)
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
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
