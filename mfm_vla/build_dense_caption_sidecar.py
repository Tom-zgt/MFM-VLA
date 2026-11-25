"""Build the compact Dense-Caption metadata sidecar for `local/libero_handoff` and `local/libero_skill_v2`.

Output:
  - `data/lerobot/local/libero_handoff/meta/dense_captions.pkl` (11,947 episodes)
  - `data/lerobot/local/libero_skill_v2/meta/dense_captions.pkl` (14,075 episodes)

Each entry `episodes[episode_index]` is a compact dict (~1.2 KB) consumed by
`dense_caption.render_dense_caption(meta, frame_index, ...)` on the fly during
OpenPI training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
from pathlib import Path

import h5py
import numpy as np

from dense_caption import (
    build_compact_episode_meta,
    compute_spans_from_subgoal_array,
    render_dense_caption,
)

EXCLUDED = {"libero_90/t021", "libero_90/t038", "libero_90/t077"}


def _extract_hdf5_shard(args: tuple[int, list[str]]) -> dict[str, dict]:
    """Worker: extract initial 3D object/target poses for a shard of HDF5 files."""
    worker_id, hdf5_paths = args
    import os
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(worker_id % 8)

    import libero_compat  # noqa: F401
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from build_vp_dataset import find_task
    from visual_prompt import goal_targets

    root = Path(__file__).resolve().parents[1]
    out: dict[str, dict] = {}
    for p_str in hdf5_paths:
        path = Path(p_str)
        suite, task_id = find_task(path)
        task_key = f"{suite}/t{task_id:03d}"
        task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
        bddl_path = root / "LIBERO/libero/libero/bddl_files" / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
        env.reset()
        inner = env.env
        sim = inner.sim
        raw_goals = [tuple(str(x) for x in g) for g in inner.parsed_problem["goal_state"]]
        obj_names = list(inner.objects_dict.keys())
        tgt_names = set()
        for g in raw_goals:
            pick, place = goal_targets(g)
            if pick and pick in inner.object_states_dict:
                tgt_names.add(pick)
            if place and place in inner.object_states_dict:
                tgt_names.add(place)
        all_names = sorted(set(obj_names) | tgt_names)

        demos_meta: dict[str, dict] = {}
        with h5py.File(path, "r") as f:
            for demo_name in f["data"].keys():
                states = np.asarray(f["data"][demo_name]["states"])
                sim.set_state_from_flattened(states[0])
                sim.forward()
                pos_t0 = {
                    k: np.asarray(inner.object_states_dict[k].get_geom_state()["pos"], dtype=np.float32).copy()
                    for k in all_names
                    if k in inner.object_states_dict
                }
                if len(raw_goals) <= 1:
                    goals_ordered = raw_goals
                else:
                    held = []
                    for s in states[::5]:
                        sim.set_state_from_flattened(s)
                        sim.forward()
                        held.append([bool(inner._eval_predicate(list(g))) for g in raw_goals])
                    held_arr = np.asarray(held, dtype=bool)
                    n_s = len(held_arr)
                    done_at = [
                        n_s - int(np.argmin(held_arr[::-1, g])) if not held_arr[:, g].all() else 0
                        for g in range(len(raw_goals))
                    ]
                    order = sorted(range(len(raw_goals)), key=lambda g: done_at[g])
                    goals_ordered = [raw_goals[g] for g in order]

                demos_meta[demo_name] = {
                    "pos": pos_t0,
                    "goals_ordered": goals_ordered,
                }
        env.close()
        out[task_key] = demos_meta
        print(f"[worker {worker_id}] extracted {task_key} ({len(demos_meta)} demos)", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache_path = root / "data/lerobot/task_demos_3d_cache.pkl"
    if cache_path.exists():
        print(f"Step 1: Loading cached 3D scene poses from {cache_path}...", flush=True)
        with cache_path.open("rb") as fh:
            task_demos = pickle.load(fh)
    else:
        hdf5_files = [str(p) for p in sorted((root / "data/libero_hdf5").glob("*/*.hdf5"))]
        shards = [hdf5_files[w :: args.workers] for w in range(args.workers)]
        print(f"Step 1: Extracting 3D scene poses from {len(hdf5_files)} HDF5 files across {args.workers} workers...", flush=True)
        task_demos: dict[str, dict] = {}
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
            for part in pool.map(_extract_hdf5_shard, enumerate(shards)):
                task_demos.update(part)
        with cache_path.open("wb") as fh:
            pickle.dump(task_demos, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # Step 2: Process vp_episodes (0..6349), handoff_episodes (6350..11946), mimicgen (11947..14074)
    vp_files = [
        f for f in sorted((root / "data/vp_episodes").glob("*/*.npz"))
        if f"{f.parent.name}/{f.name[:4]}" not in EXCLUDED
    ]
    ho_files = [
        f for f in sorted((root / "data/handoff_episodes").glob("*/*.npz"))
        if f"{f.parent.name}/{f.name[:4]}" not in EXCLUDED
    ]
    mg_files = sorted((root / "data/mimicgen/episodes").glob("*/*.npz"))
    print(f"Step 2: Building compact episode metadata for vp={len(vp_files)}, ho={len(ho_files)}, mg={len(mg_files)}...", flush=True)

    episodes: list[dict] = []
    vp_subgoal_cache: dict[str, np.ndarray] = {}

    # 2a. vp_episodes
    for f in vp_files:
        task_key = f"{f.parent.name}/{f.name[:4]}"
        demo_name = f.stem.split("_", 1)[1]  # demo_K
        with np.load(f) as z:
            prompt = str(z["prompt"])
            state = np.asarray(z["state"], dtype=np.float32)
            actions = np.asarray(z["actions"], dtype=np.float32)
            subgoal = np.asarray(z["subgoal"], dtype=np.int16)
        vp_subgoal_cache[f"{task_key}/{demo_name}"] = subgoal
        d_info = task_demos[task_key][demo_name]
        spans = compute_spans_from_subgoal_array(
            actions=actions,
            eef_pos=state[:, :3],
            goals_ordered=d_info["goals_ordered"],
            subgoal_arr=subgoal,
        )
        episodes.append(
            build_compact_episode_meta(
                prompt=prompt,
                actions=actions,
                eef_pos=state[:, :3],
                spans=spans,
                static_positions=d_info["pos"],
                recovery_len=0,
            )
        )

    # 2b. handoff_episodes
    for f in ho_files:
        task_key = f"{f.parent.name}/{f.name[:4]}"
        # stem is e.g. t000_demo_0_s0_0
        parts = f.stem.split("_")
        demo_name = f"{parts[1]}_{parts[2]}"
        with np.load(f) as z:
            prompt = str(z["prompt"])
            state = np.asarray(z["state"], dtype=np.float32)
            actions = np.asarray(z["actions"], dtype=np.float32)
            anchor = int(z["anchor"])
            rec_len = int(z["recovery_len"])
        d_info = task_demos[task_key][demo_name]
        parent_sub = vp_subgoal_cache.get(f"{task_key}/{demo_name}")
        T = len(actions)
        if parent_sub is not None and anchor < len(parent_sub):
            tail = parent_sub[anchor:]
            subgoal = np.empty(T, dtype=np.int16)
            subgoal[:rec_len] = tail[0]
            rem = T - rec_len
            if rem <= len(tail):
                subgoal[rec_len:] = tail[:rem]
            else:
                subgoal[rec_len : rec_len + len(tail)] = tail
                subgoal[rec_len + len(tail) :] = tail[-1]
        else:
            subgoal = None
        spans = compute_spans_from_subgoal_array(
            actions=actions,
            eef_pos=state[:, :3],
            goals_ordered=d_info["goals_ordered"],
            subgoal_arr=subgoal,
        )
        episodes.append(
            build_compact_episode_meta(
                prompt=prompt,
                actions=actions,
                eef_pos=state[:, :3],
                spans=spans,
                static_positions=d_info["pos"],
                recovery_len=rec_len,
            )
        )

    n_handoff_total = len(episodes)

    # 2c. mimicgen episodes
    for f in mg_files:
        with np.load(f) as z:
            prompt = str(z["prompt"])
            state = np.asarray(z["state"], dtype=np.float32)
            actions = np.asarray(z["actions"], dtype=np.float32)
            goal = tuple(str(x) for x in z["goal"])
            template_tid = int(z["task"])
        eef_pos = state[:, :3]
        spans = compute_spans_from_subgoal_array(
            actions=actions,
            eef_pos=eef_pos,
            goals_ordered=[goal],
            subgoal_arr=None,
        )
        # Seed scene positions from template task demo_0, then override active pick/place from trajectory
        t_key = f"libero_90/t{template_tid:03d}"
        pos = {}
        if t_key in task_demos and task_demos[t_key]:
            first_demo = next(iter(task_demos[t_key].values()))
            pos = {k: v.copy() for k, v in first_demo["pos"].items()}
        if spans:
            sp = spans[0]
            if sp.predicate in ("in", "on"):
                c_idx = sp.close_step if sp.close_step is not None else min(len(eef_pos) - 1, 25)
                o_idx = sp.open_step if sp.open_step is not None else len(eef_pos) - 1
                pos[sp.pick_name] = eef_pos[c_idx].copy()
                if sp.place_name:
                    pos[sp.place_name] = (eef_pos[o_idx] - np.array([0.0, 0.0, 0.04], dtype=np.float32)).copy()
            else:
                mid_idx = len(eef_pos) // 2
                pos[sp.pick_name] = eef_pos[mid_idx].copy()

        episodes.append(
            build_compact_episode_meta(
                prompt=prompt,
                actions=actions,
                eef_pos=eef_pos,
                spans=spans,
                static_positions=pos,
                recovery_len=0,
            )
        )

    # Verify against LeRobot episodes.jsonl
    ho_meta_dir = root / "data/lerobot/local/libero_handoff/meta"
    v2_meta_dir = root / "data/lerobot/local/libero_skill_v2/meta"

    with (ho_meta_dir / "dense_captions.pkl").open("wb") as fh:
        pickle.dump(episodes[:n_handoff_total], fh, protocol=pickle.HIGHEST_PROTOCOL)
    with (v2_meta_dir / "dense_captions.pkl").open("wb") as fh:
        pickle.dump(episodes, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # Quick sanity check on rendered captions
    rng = np.random.default_rng(0)
    sample_caps = [
        render_dense_caption(episodes[idx], 10, pos_noise_std_m=0.02, rng=rng)["caption"]
        for idx in (0, n_handoff_total - 1, len(episodes) - 1)
    ]
    print(f"Saved {n_handoff_total} episodes to {ho_meta_dir / 'dense_captions.pkl'}")
    print(f"Saved {len(episodes)} episodes to {v2_meta_dir / 'dense_captions.pkl'}")
    for c in sample_caps:
        print("  sample:", c)


if __name__ == "__main__":
    main()
