"""Stage 1 of the visual-prompt dataset: replay LIBERO demonstrations and record what
π0.5 sees, plus where the marks go.

Every recorded simulator state of a demonstration is restored in ``TaskEnv``,
rendered at 256x256 by the evaluation harness, and stored as the exact policy
inputs (agentview and wrist images mirrored as ``OpenPiVLAExecutor`` sends them,
the 8-D state) next to the demonstrated action. Restoring states rather than
re-executing actions keeps every demonstration (open-loop replay drifts and fails
on about a fifth of them). No-op actions are dropped, as in the dataset openpi's
LIBERO checkpoints were trained on.

Marks come from the BDDL goal (training data only): the first argument of the
active goal predicate is the pick/act target, the second (In/On) the place target.
Sub-goals are ordered by when the demonstration achieves them for good; each one's
marks are projected into the policy image at the frame it becomes active and kept
fixed until it is achieved, then the next sub-goal's marks take over.

Usage: python build_vp_dataset.py --hdf5 FILE [--hdf5 FILE ...] --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

import libero_compat  # noqa: F401

from libero.libero import benchmark
from harness import AGENT_CAM, WRIST_CAM, TaskEnv
from vla_executor import _quat2axisangle
from visual_prompt import draw_marks, goal_targets, project_target

SIZE = 256


def find_task(hdf5: Path) -> tuple[str, int]:
    suite = hdf5.parent.name
    stem = hdf5.name.removesuffix("_demo.hdf5")
    tasks = benchmark.get_benchmark_dict()[suite]()
    for i in range(tasks.n_tasks):
        if Path(tasks.get_task(i).bddl_file).stem == stem:
            return suite, i
    raise KeyError(f"no {suite} task for {hdf5.name}")


def replay(env: TaskEnv, demo) -> dict | None:
    actions = np.asarray(demo["actions"], dtype=np.float32)
    states = np.asarray(demo["states"])
    env._env.reset()
    env._render_tuned = False
    env._K.clear()
    inner = env._env.env
    goals = list(inner.parsed_problem["goal_state"])
    targets = [goal_targets(g) for g in goals]
    rec = {k: [] for k in ("image", "wrist_image", "state", "actions")}
    uv = {g: [] for g in range(len(goals))}
    held = []
    prev_grip = None
    for state, action in zip(states, actions):
        if prev_grip is not None and np.linalg.norm(action[:6]) < 1e-4 and action[6] == prev_grip:
            continue
        prev_grip = action[6]
        env._env.regenerate_obs_from_state(state)
        obs = env.observe(size=SIZE)
        for g, (pick, place) in enumerate(targets):
            uv[g].append((project_target(inner, pick, obs, AGENT_CAM, SIZE),
                          project_target(inner, place, obs, AGENT_CAM, SIZE)))
        rec["image"].append(np.ascontiguousarray(obs.rgb[AGENT_CAM][:, ::-1]))
        rec["wrist_image"].append(np.ascontiguousarray(obs.rgb[WRIST_CAM][:, ::-1]))
        rec["state"].append(np.concatenate(
            (obs.eef_pos, _quat2axisangle(obs.eef_quat), obs.gripper_qpos)).astype(np.float32))
        rec["actions"].append(action)
        held.append([bool(inner._eval_predicate(g)) for g in goals])
    if not rec["actions"]:
        return None

    # A sub-goal is done from the first frame after which it holds through the end;
    # the demo works on sub-goals in that order, whatever order the BDDL lists them in.
    held = np.asarray(held)
    n = len(held)
    done_at = [n - int(np.argmin(held[::-1, g])) if not held[:, g].all() else 0 for g in range(len(goals))]
    order = sorted(range(len(goals)), key=lambda g: done_at[g])
    ep = {k: np.stack(v) for k, v in rec.items()}
    ep["pick_uv"] = np.full((n, 2), np.nan, dtype=np.float32)
    ep["place_uv"] = np.full((n, 2), np.nan, dtype=np.float32)
    ep["subgoal"] = np.zeros(n, dtype=np.int16)
    start = 0
    for k, g in enumerate(order):
        end = n if k == len(order) - 1 else max(start, done_at[g])
        if end > start:
            ep["pick_uv"][start:end], ep["place_uv"][start:end] = uv[g][start]
            ep["subgoal"][start:end] = k
        start = end
    return ep


def preview(ep: dict) -> np.ndarray:
    """First frame of every sub-goal with its marks, side by side."""
    starts = [0] + [i for i in range(1, len(ep["subgoal"])) if ep["subgoal"][i] != ep["subgoal"][i - 1]]
    return np.concatenate(
        [draw_marks(ep["image"][i], ep["pick_uv"][i], ep["place_uv"][i]) for i in starts], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-demos", type=int, default=0)
    args = parser.parse_args()
    for path in args.hdf5:
        suite, task_id = find_task(path)
        out_dir = args.out / suite
        out_dir.mkdir(parents=True, exist_ok=True)
        env = TaskEnv(suite, task_id, img_size=SIZE)
        kept = failed = 0
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            for name in demos:
                target = out_dir / f"t{task_id:03d}_{name}.npz"
                if target.exists():
                    kept += 1
                    continue
                ep = replay(env, f["data"][name])
                if ep is None:
                    failed += 1
                    continue
                np.savez_compressed(target, prompt=env.instruction, suite=suite, task=task_id, **ep)
                if kept == 0:
                    Image.fromarray(preview(ep)).save(out_dir / f"t{task_id:03d}_preview.png")
                kept += 1
                if args.max_demos and kept >= args.max_demos:
                    break
        env.close()
        print(f"{suite} t{task_id} kept={kept} failed={failed} {path.name}", flush=True)


if __name__ == "__main__":
    main()
