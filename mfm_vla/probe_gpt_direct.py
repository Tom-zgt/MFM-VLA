"""GPT-only baseline (no VLA): GPT-6 is the sole closed-loop controller for a whole episode.

Every decision GPT-6 sees the two policy cameras (agentview + wrist, current and previous),
the proprioception a VLA also gets, and its own recent commands; it returns one small
world-frame end-effector motion and a gripper command, which a low-level servo executes.
The calibrated camera model draws the world axes at the gripper so directions can be read
from the image. Simulator state is read only for evaluation logging (lifted objects).

Usage: probe_gpt_direct.py OUT.jsonl PART NPARTS [INITS] [FAMILIES]
  FAMILIES: comma list of libero_goal,libero_10,libero_spatial,libero_object (default all)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import libero_compat  # noqa: F401
from mfm_client import MFMClientSession, MFMClientError
from control import orientation_error, quat_xyzw_to_mat, rotvec_to_mat
from eval_preposition_pick import LiftTracker
from eval_vla_only import SUITE_HORIZONS
from harness import AGENT_CAM, WRIST_CAM, TaskEnv

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ("libero_goal", "libero_10", "libero_spatial", "libero_object")
VARIANTS = ("task", "swap", "lan")
MODEL = os.environ.get("MFM_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o"))
HORIZON_SCALE = float(os.environ.get("GPT_HORIZON_SCALE", "1"))
IMAGE_SIZE = 256
MAX_TRANSLATION_M = 0.05
MAX_ROTATION_RAD = 0.3
MAX_SERVO_STEPS = 10
GRIPPER_SETTLE_STEPS = 8
HISTORY = 6
AXIS_M = 0.08
AXES = (("x", (230, 40, 40)), ("y", (40, 200, 40)), ("z", (60, 110, 255)))

PROMPT = """You are the only controller of a Franka Panda arm in a tabletop simulator. No other
policy will act. Complete the task:

  "{instruction}"

Each turn you send ONE small end-effector command; a servo executes it, then you get fresh
images. You have at most {budget} env steps left (each command uses up to {servo} steps).

Cameras (current_* = now, previous_* = before your last command):
- agentview: fixed third-person view of the workspace.
- robot0_eye_in_hand: wrist camera between the fingers, looking along the gripper.
Drawn on both images at the gripper: world axes, red = +x, green = +y, blue = +z (up),
each arrow {axis_cm} cm long. Use them to turn what you see into world directions.

Proprioception (world frame, metres):
- gripper position: {eef}
- gripper orientation quaternion xyzw: {quat}
- finger opening: {opening:.3f} m (about 0.08 fully open, below 0.01 closed on nothing)

Your recent commands and their outcomes (oldest first):
{history}
{error}
Command semantics:
- delta_position_m: world-frame translation of the gripper, norm at most {max_cm} cm.
- delta_rotation_vector_rad: world-frame axis-angle rotation of the gripper, norm at most
  {max_rad} rad. Use [0,0,0] unless the task needs a different gripper orientation.
- gripper: "open" or "close". Closing takes a few steps; the servo waits for it.
- done: true only when the images show the task is complete.

Work like a careful operator: locate the relevant objects, move above, descend, grasp, verify
the grasp in the wrist view, carry, place or push, release. Articulated parts (drawers, doors,
knobs) are moved by grasping or pushing their handle along the direction they move. Prefer
1-2 cm moves near contact. Do not repeat a command that made no visible progress.

Return exactly one JSON object:
{{"evidence": "what the current images show that matters now",
 "plan": "remaining steps, short",
 "delta_position_m": [dx, dy, dz],
 "delta_rotation_vector_rad": [rx, ry, rz],
 "gripper": "open|close",
 "done": false}}"""


def jobs(inits: list[int], families: tuple[str, ...]) -> list[tuple[str, int, int]]:
    split = json.loads((ROOT / "reports/task_split.json").read_text())["suites"]
    return [(f"{fam}_{var}", task, init)
            for fam in families for task in split[fam]["dev"]
            for var in VARIANTS for init in inits]


def annotate(obs, cam: str) -> np.ndarray:
    image = Image.fromarray(obs.rgb[cam])
    draw = ImageDraw.Draw(image)
    T = np.linalg.inv(obs.cam_to_world[cam])

    def project(point):
        p = (T @ np.append(point, 1.0))[:3]
        if p[2] <= 1e-3:
            return None
        uv = obs.intrinsics[cam] @ p
        return float(uv[0] / uv[2]), float(uv[1] / uv[2])

    origin = project(obs.eef_pos)
    if origin is None:
        return np.asarray(image)
    for axis, (label, colour) in enumerate(AXES):
        tip = project(obs.eef_pos + AXIS_M * np.eye(3)[axis])
        if tip is None:
            continue
        draw.line([origin, tip], fill=colour, width=2)
        draw.text((tip[0] + 2, tip[1] - 6), label, fill=colour)
    return np.asarray(image)


def ask(brain, prompt: str, images: dict) -> dict:
    """Wait out shared quota gaps without consuming simulator steps."""
    for wait in (60, 120, 240, 480, 900, 900, 900, 900):
        try:
            return brain.decide_json(prompt, images)
        except MFMClientError as exc:
            if "capacity" not in str(exc) and "rate limited" not in str(exc):
                raise
            print(f"quota wait {wait}s: {str(exc)[:80]}", flush=True)
            time.sleep(wait)
    return brain.decide_json(prompt, images)


def parse(payload: dict) -> dict:
    def vector(key):
        value = np.asarray(payload.get(key, [0, 0, 0]), dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must be three finite numbers")
        return value

    move, turn = vector("delta_position_m"), vector("delta_rotation_vector_rad")
    if np.linalg.norm(move) > MAX_TRANSLATION_M + 1e-6:
        raise ValueError(f"delta_position_m norm exceeds {MAX_TRANSLATION_M} m")
    if np.linalg.norm(turn) > MAX_ROTATION_RAD + 1e-6:
        raise ValueError(f"delta_rotation_vector_rad norm exceeds {MAX_ROTATION_RAD} rad")
    gripper = str(payload.get("gripper", "")).strip().lower()
    if gripper not in ("open", "close"):
        raise ValueError('gripper must be "open" or "close"')
    return {"delta_position_m": move, "delta_rotation_vector_rad": turn, "gripper": gripper,
            "done": bool(payload.get("done", False)),
            "evidence": str(payload.get("evidence", ""))[:400], "plan": str(payload.get("plan", ""))[:300]}


def execute(env, command: dict, gripper_changed: bool, on_step, horizon: int) -> int:
    start = env.proprio()
    goal_pos = start.eef_pos + command["delta_position_m"]
    goal_rot = rotvec_to_mat(command["delta_rotation_vector_rad"]) @ quat_xyzw_to_mat(start.eef_quat)
    grip = 1.0 if command["gripper"] == "close" else -1.0
    steps = 0
    while steps < MAX_SERVO_STEPS and env.t < horizon and not env.success:
        p = env.proprio()
        pos_err = goal_pos - p.eef_pos
        rot_err = orientation_error(quat_xyzw_to_mat(p.eef_quat), goal_rot)
        settled = np.linalg.norm(pos_err) < 0.003 and np.linalg.norm(rot_err) < 0.03
        if steps and settled and not (gripper_changed and steps < GRIPPER_SETTLE_STEPS):
            break
        action = np.zeros(7)
        action[:3] = np.clip(1.6 * pos_err / env.action_pos_scale, -1.0, 1.0)
        action[3:6] = np.clip(1.8 * rot_err, -0.4, 0.4)
        action[6] = grip
        env.step(action)
        on_step(env)
        steps += 1
    return steps


def run(brain, suite: str, task: int, init: int) -> dict:
    horizon = int(SUITE_HORIZONS[suite] * HORIZON_SCALE)
    env = TaskEnv(suite, task, img_size=128, seed=init,
                  env_horizon=max(1000, horizon + 100))
    env.allow_goal_shortcut = False
    env.reset(init)
    tracker = LiftTracker(env)
    history, decisions, previous = [], [], None
    last_gripper, reason, invalid = "open", "horizon", 0
    try:
        while env.t < horizon and not env.success:
            obs = env.observe(size=IMAGE_SIZE)
            current = {cam: annotate(obs, cam) for cam in (AGENT_CAM, WRIST_CAM)}
            images = {f"current_{cam}": im for cam, im in current.items()}
            if previous is not None:
                images.update({f"previous_{cam}": im for cam, im in previous.items()})
            error, command = "", None
            for _ in range(3):
                prompt = PROMPT.format(
                    instruction=env.instruction, budget=horizon - env.t, servo=MAX_SERVO_STEPS,
                    axis_cm=int(AXIS_M * 100), eef=np.round(obs.eef_pos, 3).tolist(),
                    quat=np.round(obs.eef_quat, 3).tolist(),
                    opening=env.proprio().gripper_opening,
                    history="\n".join(history[-HISTORY:]) or "(none yet)",
                    error=f"\nYour previous answer was invalid: {error}. Fix it.\n" if error else "",
                    max_cm=int(MAX_TRANSLATION_M * 100), max_rad=MAX_ROTATION_RAD)
                try:
                    command = parse(ask(brain, prompt, images))
                    break
                except (TypeError, ValueError) as exc:
                    error, invalid = str(exc), invalid + 1
            if command is None:
                reason = f"invalid command: {error}"
                break
            if command["done"]:
                reason = "gpt_done"
                break
            before = env.proprio()
            steps = execute(env, command, command["gripper"] != last_gripper, tracker.update, horizon)
            after = env.proprio()
            last_gripper, previous = command["gripper"], current
            moved = after.eef_pos - before.eef_pos
            history.append(
                f"- t={before.t}: move {np.round(command['delta_position_m'], 3).tolist()}, "
                f"rotate {np.round(command['delta_rotation_vector_rad'], 2).tolist()}, "
                f"gripper {command['gripper']} -> actually moved {np.round(moved, 3).tolist()}, "
                f"finger opening {after.gripper_opening:.3f} m")
            decisions.append({"t": before.t, "steps": steps, "gripper": command["gripper"],
                              "move": np.round(command["delta_position_m"], 4).tolist(),
                              "rot": np.round(command["delta_rotation_vector_rad"], 3).tolist(),
                              "moved": np.round(moved, 4).tolist(),
                              "evidence": command["evidence"], "plan": command["plan"]})
        if env.success:
            reason = "success"
        return {"success": bool(env.success), "steps": env.t, "reason": reason,
                "n_decisions": len(decisions), "invalid_answers": invalid,
                "first_lifted": tracker.first_lifted(),
                "max_lift_m": {k: round(v, 3) for k, v in tracker.max_lift.items() if v >= 0.01},
                "gripper_opening_end": round(env.proprio().gripper_opening, 4),
                "decisions": decisions}
    finally:
        env.close()


def main() -> None:
    out_path, part, nparts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    inits = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2]
    families = tuple(sys.argv[5].split(",")) if len(sys.argv) > 5 else FAMILIES
    done = {
        (r["suite"], r["task"], r["init"])
        for path in Path(out_path).parent.glob("*.jsonl")
        for r in map(json.loads, open(path))
        if not r.get("crash")
    }
    with open(out_path, "a") as handle:
        for suite, task, init in jobs(inits, families)[part::nparts]:
            if (suite, task, init) in done:
                continue
            brain = MFMClientSession(model=MODEL, retries=9)
            started = time.monotonic()
            try:
                row = run(brain, suite, task, init)
                row["crash"] = False
            except Exception as exc:
                row = {"success": False, "crash": True, "reason": f"{type(exc).__name__}: {exc}",
                       "detail": traceback.format_exc(limit=6)}
            finally:
                usage = brain.usage()
                brain.close()
            row = {"suite": suite, "task": task, "init": init, "model": MODEL,
                   "horizon_scale": HORIZON_SCALE, **row,
                   "usage": usage, "duration_s": round(time.monotonic() - started, 1)}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(suite, task, init, row["success"], row["reason"], row.get("steps"),
                  row["duration_s"], flush=True)


if __name__ == "__main__":
    main()
