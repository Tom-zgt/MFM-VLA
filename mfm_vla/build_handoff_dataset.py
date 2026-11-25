"""Hand-off recovery data for π0.5: start from GPT-like poses, finish the demo task.

For each sample a demonstration state ``s_k`` (anchor frame k) is restored, the arm
is driven away to a perturbed pose resembling where the GPT controller hands control
back (statistics from G5j hand-offs), and from there a recording starts: a Cartesian
controller brings the hand back to the demonstrated pose of frame k, then the
demonstrated actions from k on are replayed. Only samples whose replay still solves
the task are kept, so every frame is a valid "take over from here" example under
the original task instruction. Simulator ground truth is used for data generation
only.

Anchor phases (fractions from G5j's regressions): just before a grasp (gripper open,
hand above the object), just before a release (object held), or anywhere.
Perturbations: "near" (a few cm off and above, small tilt, like an approach
hand-off) or "far" (decimetres off with a large tilt, like after a GPT drawer
opening or traverse).

Usage: python build_handoff_dataset.py --hdf5 FILE [...] --out DIR --per-demo N --seed S
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import zlib

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

import libero_compat  # noqa: F401

from build_vp_dataset import find_task
from harness import AGENT_CAM, WRIST_CAM, TaskEnv
from vla_executor import _quat2axisangle

SIZE = 256
MOVE_NORM = 0.8
FAILS: Counter = Counter()


def fail(reason: str) -> None:
    FAILS[reason] += 1
    return None
POS_TOL = 0.01
ROT_TOL = 0.08


def anchor_frame(actions: np.ndarray, rng: np.random.Generator) -> tuple[int, str]:
    grip = np.sign(actions[:, 6])
    flips = np.flatnonzero(grip[1:] != grip[:-1]) + 1
    closes = [i for i in flips if grip[i] > 0]
    opens = [i for i in flips if grip[i] < 0]
    u = rng.random()
    n = len(actions)
    if u < 0.5 and closes:
        c = int(rng.choice(closes))
        return int(np.clip(c - rng.integers(5, 40), 1, n - 20)), "pre_grasp"
    if u < 0.8 and opens:
        o = int(rng.choice(opens))
        return int(np.clip(o - rng.integers(5, 40), 1, n - 20)), "pre_release"
    return int(rng.integers(1, max(2, n - 20))), "anywhere"


def perturbation(rng: np.random.Generator) -> tuple[np.ndarray, Rotation, str]:
    if rng.random() < 0.6:
        offset = np.array([*rng.normal(0, 0.03, 2), rng.uniform(0.0, 0.12)])
        tilt = Rotation.from_rotvec(_random_axis(rng) * np.radians(rng.uniform(0, 10)))
        return offset, tilt, "near"
    r = rng.uniform(0.08, 0.25)
    a = rng.uniform(0, 2 * np.pi)
    offset = np.array([r * np.cos(a), r * np.sin(a), rng.uniform(0.05, 0.25)])
    tilt = Rotation.from_rotvec(_random_axis(rng) * np.radians(rng.uniform(10, 45)))
    return offset, tilt, "far"


def _random_axis(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def servo_action(env: TaskEnv, pos: np.ndarray, rot: Rotation, gripper: float,
                 speed: float) -> tuple[np.ndarray, float, float]:
    p = env.proprio()
    err = pos - p.eef_pos
    turn = (rot * Rotation.from_quat(p.eef_quat).inv()).as_rotvec()
    action = np.zeros(7)
    action[:3] = 1.5 * err / env.action_pos_scale
    norm = float(np.linalg.norm(action[:3]))
    if norm > MOVE_NORM * speed:
        action[:3] *= MOVE_NORM * speed / norm
    action[3:6] = np.clip(1.2 * turn / env.action_rot_scale, -0.6 * speed, 0.6 * speed)
    action[6] = gripper
    return action, float(np.linalg.norm(err)), float(np.linalg.norm(turn))


def drive(env: TaskEnv, pos: np.ndarray, rot: Rotation, gripper: float, max_steps: int, record=None) -> bool:
    speed = 0.5 if gripper > 0 else 1.0
    for _ in range(max_steps):
        action, d, r = servo_action(env, pos, rot, gripper, speed)
        if d < POS_TOL and r < ROT_TOL:
            return True
        if record is not None:
            record(action)
        env.step(action)
    return False


def object_positions(inner) -> dict[str, np.ndarray]:
    return {n: np.asarray(inner.object_states_dict[n].get_geom_state()["pos"], dtype=float).copy()
            for n in inner.objects_dict}


def sample(env: TaskEnv, states: np.ndarray, actions: np.ndarray, ee_pos: np.ndarray, ee_ori: np.ndarray,
           rng: np.random.Generator) -> dict | None:
    k, phase = anchor_frame(actions, rng)
    offset, tilt, kind = perturbation(rng)
    if actions[k - 1, 6] > 0:
        tilt = Rotation.from_rotvec(tilt.as_rotvec() * min(1.0, np.radians(25) / max(tilt.magnitude(), 1e-9)))
    env._env.reset()
    env._render_tuned = False
    env._K.clear()
    env._env.regenerate_obs_from_state(states[k])
    inner = env._env.env
    gripper = float(np.sign(actions[k - 1, 6]) or -1.0)
    p0 = env.proprio()
    home_pos, home_rot = p0.eef_pos.copy(), Rotation.from_quat(p0.eef_quat)
    held = gripper > 0
    before = object_positions(inner)

    # Away: rise first so the hand does not sweep the table, then go to the hand-off pose.
    away_pos = home_pos + offset
    away_rot = tilt * home_rot
    rise = home_pos.copy()
    rise[2] = max(home_pos[2], away_pos[2])
    if not drive(env, rise, home_rot, gripper, 60) or not drive(env, away_pos, away_rot, gripper, 120):
        return fail("away")
    if not held:
        after = object_positions(inner)
        if any(np.linalg.norm(after[n] - before[n]) > 0.015 for n in before):
            return fail("disturbed")

    rec = {k_: [] for k_ in ("image", "wrist_image", "state", "actions")}

    def record(action):
        obs = env.observe(size=SIZE)
        rec["image"].append(np.ascontiguousarray(obs.rgb[AGENT_CAM][:, ::-1]))
        rec["wrist_image"].append(np.ascontiguousarray(obs.rgb[WRIST_CAM][:, ::-1]))
        rec["state"].append(np.concatenate(
            (obs.eef_pos, _quat2axisangle(obs.eef_quat), obs.gripper_qpos)).astype(np.float32))
        rec["actions"].append(np.asarray(action, dtype=np.float32))

    if not drive(env, home_pos, home_rot, gripper, 120, record):
        return fail("return")
    recovery = len(rec["actions"])
    if recovery == 0:
        return fail("no_recovery")
    # Demonstrated actions plus a small pull toward the demonstrated pose of the same
    # frame, which absorbs the drift the detour left behind.
    prev_grip = gripper
    for t in range(k, len(actions)):
        action = actions[t].astype(np.float64).copy()
        if np.linalg.norm(action[:6]) < 1e-4 and action[6] == prev_grip:
            continue
        prev_grip = action[6]
        p = env.proprio()
        turn = (Rotation.from_rotvec(ee_ori[t]) * Rotation.from_quat(p.eef_quat).inv()).as_rotvec()
        action[:3] += np.clip(1.0 * (ee_pos[t] - p.eef_pos) / env.action_pos_scale, -0.3, 0.3)
        action[3:6] += np.clip(0.8 * turn / env.action_rot_scale, -0.3, 0.3)
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        record(action)
        env.step(action)
    if not env.success:
        return fail(f"replay_{phase}_{kind}")
    ep = {k_: np.stack(v) for k_, v in rec.items()}
    ep.update(anchor=k, phase=phase, kind=kind, recovery_len=recovery,
              offset=offset.astype(np.float32), tilt_deg=np.float32(np.degrees(tilt.magnitude())))
    return ep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-demo", type=int, default=1)
    parser.add_argument("--tries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-demos", type=int, default=0)
    args = parser.parse_args()
    for path in args.hdf5:
        suite, task_id = find_task(path)
        out_dir = args.out / suite
        out_dir.mkdir(parents=True, exist_ok=True)
        env = TaskEnv(suite, task_id, img_size=SIZE)
        env.allow_goal_shortcut = False
        kept = tried = 0
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            if args.max_demos:
                demos = demos[: args.max_demos]
            for name in demos:
                states = np.asarray(f["data"][name]["states"])
                actions = np.asarray(f["data"][name]["actions"], dtype=np.float32)
                ee_pos = np.asarray(f["data"][name]["obs"]["ee_pos"], dtype=np.float64)
                ee_ori = np.asarray(f["data"][name]["obs"]["ee_ori"], dtype=np.float64)
                for j in range(args.per_demo):
                    target = out_dir / f"t{task_id:03d}_{name}_s{args.seed}_{j}.npz"
                    if target.exists():
                        kept += 1
                        continue
                    rng = np.random.default_rng([args.seed, task_id, int(name.split("_")[1]), j,
                                                 zlib.crc32(suite.encode())])
                    for _ in range(args.tries):
                        tried += 1
                        ep = sample(env, states, actions, ee_pos, ee_ori, rng)
                        if ep is not None:
                            np.savez_compressed(target, prompt=env.instruction, suite=suite, task=task_id, **ep)
                            kept += 1
                            break
        env.close()
        print(f"{suite} t{task_id} kept={kept} tried={tried} {path.name} fails={dict(FAILS)}", flush=True)
        FAILS.clear()


if __name__ == "__main__":
    main()
