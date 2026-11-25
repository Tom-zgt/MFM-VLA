"""Build Part 2 of Scheme A: G5j Hand-off Recovery Rollouts for RoboCasa365.

Generates two complementary sets of hand-off recovery trajectories under
`data/robocasa_dense_v1/handoff_rollouts/`:
  1. `sim_ho_*`: Closed-loop simulator hand-off rollouts starting from `adapter.move_to`'s
     exact pre-contact pose (`+8cm` above object or `-8cm` in front of fixture, +/- 2cm noise),
     retaining ONLY trajectories verified to lift the target object or complete the fixture action.
  2. `stage_ho_*`: Stage-specific hand-off recovery episodes extracted from every stage
     (especially Stage 2 and Stage 3 of Composite-Seen tasks) of the official human
     demonstrations, starting at the G5j pre-contact distance (~8cm from grasp/contact)
     with an explicit G5j hover-to-rim approach bridge for bowl-grasping stages.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Dict, List

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from robocasa_dense_caption import build_episode_dense_meta, compute_world_eef_trajectory


def extract_stage_handoff_episodes(human_dir: Path, out_dir: Path) -> int:
    """Extract stage-specific G5j hand-off recovery episodes from `human_demos/*.pkl`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    human_files = sorted(human_dir.glob("*.pkl"))
    saved = 0

    for p in human_files:
        with p.open("rb") as fh:
            ep = pickle.load(fh)
        task = str(ep["task"])
        ep_idx = int(ep["episode_index"])
        inst = str(ep["instruction"])
        states = np.asarray(ep["states"], dtype=np.float32)
        actions = np.asarray(ep["actions"], dtype=np.float32)
        left_jpg = ep["left_jpg"]
        right_jpg = ep["right_jpg"]
        wrist_jpg = ep["wrist_jpg"]
        meta = ep["dense_meta"]
        T = len(states)
        stages = meta["stages"]

        for stg in stages:
            k = int(stg["stage_idx"])
            out_path = out_dir / f"stage_ho_{task}_ep{ep_idx:03d}_s{k}.pkl"
            if out_path.exists():
                saved += 1
                continue

            gt = int(stg["grasp_t"])
            # Start ~28 frames before grasp/contact (where gripper is ~7-10cm above/in-front-of target)
            c_start = max(int(stg["start_t"]), gt - 28)
            c_end = min(T, max(gt + 50, c_start + 65))
            if c_end - c_start < 35:
                continue

            sub_states = states[c_start:c_end].copy()
            sub_actions = actions[c_start:c_end].copy()
            sub_left = list(left_jpg[c_start:c_end])
            sub_right = list(right_jpg[c_start:c_end])
            sub_wrist = list(wrist_jpg[c_start:c_end])

            # For bowl-grasping stages (e.g. StoreLeftoversInBowl Stage 2, SteamInMicrowave Stage 1),
            # prepend a 12-frame synthetic G5j Bowl-Center Hover -> Rim Approach bridge so that
            # starting directly above the bowl center (+8cm) smoothly drives the gripper to `sub_states[0]`.
            if "bowl" in str(stg["grasp_target"]) and k > 0:
                prev_stg = stages[k - 1]
                bowl_center_world = np.asarray(prev_stg["place_pos_world"], dtype=np.float64)
                R_b0 = np.asarray(meta["R_bases"][c_start], dtype=np.float64)
                base_pos0 = sub_states[0, 7:10].astype(np.float64)
                rim_world = np.asarray(stg["grasp_pos_world"], dtype=np.float64)
                # G5j hover pose: 8cm above bowl center
                hover_world = np.array([bowl_center_world[0], bowl_center_world[1], rim_world[2] + 0.085])
                target_entry_world = base_pos0 + R_b0 @ sub_states[0, 0:3].astype(np.float64)

                bridge_len = 12
                b_states = np.repeat(sub_states[:1], bridge_len, axis=0)
                b_actions = np.zeros((bridge_len, 12), dtype=np.float32)
                for b_i in range(bridge_len):
                    alpha = b_i / float(bridge_len)
                    pos_w = (1.0 - alpha) * hover_world + alpha * target_entry_world
                    next_w = (1.0 - (b_i + 1) / float(bridge_len)) * hover_world + ((b_i + 1) / float(bridge_len)) * target_entry_world
                    b_states[b_i, 0:3] = (R_b0.T @ (pos_w - base_pos0)).astype(np.float32)
                    step_base = R_b0.T @ (next_w - pos_w)
                    b_actions[b_i, 0:3] = np.clip(step_base * 18.0, -0.8, 0.8).astype(np.float32)
                    b_actions[b_i, 3:6] = sub_actions[0, 3:6] * 0.5
                    b_actions[b_i, 6] = -1.0   # open gripper
                    b_actions[b_i, 11] = -1.0  # arm control mode

                sub_states = np.concatenate([b_states, sub_states], axis=0)
                sub_actions = np.concatenate([b_actions, sub_actions], axis=0)
                sub_left = [sub_left[0]] * bridge_len + sub_left
                sub_right = [sub_right[0]] * bridge_len + sub_right
                sub_wrist = [sub_wrist[0]] * bridge_len + sub_wrist
                local_gt = (gt - c_start) + bridge_len
            else:
                local_gt = gt - c_start

            sub_len = len(sub_states)
            local_rt = min(sub_len - 1, max(local_gt + 15, int(stg["release_t"]) - c_start))
            eef_w, R_b = compute_world_eef_trajectory(sub_states)

            # Build single-stage metadata preserving original stage_idx and num_stages!
            stg_copy = copy.deepcopy(stg)
            stg_copy["start_t"] = 0
            stg_copy["grasp_t"] = int(np.clip(local_gt, 0, sub_len - 1))
            stg_copy["release_t"] = int(np.clip(local_rt, stg_copy["grasp_t"], sub_len - 1))
            stg_copy["end_t"] = sub_len

            sub_meta = {
                "task_name": task,
                "instruction": inst,
                "num_frames": sub_len,
                "eef_world": eef_w.astype(np.float32),
                "R_bases": R_b.astype(np.float32),
                "grip_cmd": sub_actions[:, 6].astype(np.float32),
                "stages": [stg_copy],
                "handoff_frames": list(range(sub_len)),
            }

            payload = {
                "source": "stage_handoff",
                "task": task,
                "episode_index": int(ep_idx * 10 + k),
                "instruction": inst,
                "states": sub_states,
                "actions": sub_actions,
                "left_jpg": sub_left,
                "right_jpg": sub_right,
                "wrist_jpg": sub_wrist,
                "dense_meta": sub_meta,
            }
            tmp = out_path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(out_path)
            saved += 1

    return saved


def _collect_sim_task_worker(args: tuple[str, int, int, str]) -> str:
    """Worker running in `.venv-robocasa` to collect verified simulator hand-off rollouts."""
    task, num_target, gpu_id, out_dir_str = args
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    import robocasa  # noqa: F401
    from eval_robocasa import RoboCasaAdapter
    from robocasa_dense_caption import task_stage_templates

    rng = np.random.default_rng(1000 + gpu_id * 97 + abs(hash(task)) % 997)
    saved = 0
    attempts = 0
    max_attempts = num_target * 3

    while saved < num_target and attempts < max_attempts:
        ep_out = out_dir / f"sim_ho_{task}_ep{saved:03d}.pkl"
        if ep_out.exists():
            saved += 1
            continue

        seed = 500 + attempts * 37
        attempts += 1
        env = gym.make(f"robocasa/{task}", split="pretrain", seed=seed)
        try:
            rec_left, rec_right, rec_wrist = [], [], []
            rec_states, rec_actions = [], []
            recording = False

            def on_step(act12: np.ndarray, _mode: str) -> None:
                if not recording:
                    return
                obs = adapter.last_obs
                for cam_key, dst in (
                    ("video.robot0_agentview_left", rec_left),
                    ("video.robot0_agentview_right", rec_right),
                    ("video.robot0_eye_in_hand", rec_wrist),
                ):
                    rgb = np.ascontiguousarray(obs[cam_key])
                    rgb224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                    _, buf = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(rgb224, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                    dst.append(buf)
                st = np.concatenate(
                    (
                        obs["state.end_effector_position_relative"],
                        obs["state.end_effector_rotation_relative"],
                        obs["state.base_position"],
                        obs["state.base_rotation"],
                        obs["state.gripper_qpos"],
                    ),
                    axis=0,
                ).astype(np.float32)
                rec_states.append(st)
                rec_actions.append(np.asarray(act12, dtype=np.float32).copy())

            adapter = RoboCasaAdapter(env, on_step=on_step)
            adapter.reset(seed=seed)
            inner = adapter.inner
            inst = adapter.instruction
            R_base = adapter.base_rotation_matrix()
            templates = task_stage_templates(task, inst)
            stg0 = templates[0]
            noise_xy = rng.uniform(-0.02, 0.02, size=2)

            verified = False
            grasp_pos_w = None
            place_pos_w = None

            if stg0["kind"] == "pick_place" and getattr(inner, "obj_body_id", None):
                # Pick first movable object
                obj_keys = [k for k in inner.obj_body_id.keys() if "bowl" not in k] or list(inner.obj_body_id.keys())
                obj_k = obj_keys[0]
                bid = inner.obj_body_id[obj_k]
                p_obj0 = inner.sim.data.body_xpos[bid].copy()
                grasp_pos_w = p_obj0.copy()
                place_pos_w = p_obj0 + np.array([0.0, 0.0, 0.12])

                # Move unrecorded to G5j hand-off pose (+8cm above object + noise)
                pre_grasp = p_obj0 + np.array([noise_xy[0], noise_xy[1], 0.085])
                adapter.move_to(pre_grasp, gripper="open", max_steps=45)

                # Start recording hand-off recovery!
                recording = True
                adapter.move_to(p_obj0 + np.array([0.0, 0.0, -0.015]), gripper="open", tolerance_m=0.01, max_steps=25, allow_base_assist=False)
                grasp_t_rec = max(0, len(rec_states) - 1)
                for _ in range(14):
                    act = np.zeros(12, dtype=np.float64)
                    act[6] = 1.0
                    act[11] = -1.0
                    adapter.step_12d(act, mode="grasp_close")
                adapter.move_to(p_obj0 + np.array([0.0, 0.0, 0.12]), gripper="close", tolerance_m=0.015, max_steps=25, allow_base_assist=False)
                for _ in range(12):
                    act = np.zeros(12, dtype=np.float64)
                    act[6] = 1.0
                    act[11] = -1.0
                    adapter.step_12d(act, mode="hold_lift")
                recording = False

                p_obj1 = inner.sim.data.body_xpos[bid].copy()
                if (p_obj1[2] - p_obj0[2]) > 0.03 or adapter.holding():
                    verified = True
            else:
                # Fixture action (CloseFridge, CloseToasterOvenDoor, TurnOnMicrowave, TurnOffStove)
                p_eef0 = adapter.proprio().eef_pos.copy()
                # Drive toward fixture and push/press
                recording = True
                for _ in range(28):
                    if adapter.success:
                        break
                    act = np.zeros(12, dtype=np.float64)
                    act[0] = 0.65   # forward push
                    act[2] = 0.15 if "ToasterOven" in task else 0.0
                    act[6] = 1.0
                    act[11] = -1.0
                    adapter.step_12d(act, mode="fixture_push")
                recording = False
                grasp_t_rec = len(rec_states) // 2
                grasp_pos_w = adapter.proprio().eef_pos.copy()
                place_pos_w = grasp_pos_w.copy()
                if adapter.success:
                    verified = True

            if verified and len(rec_states) >= 25:
                s_arr = np.stack(rec_states, axis=0)
                a_arr = np.stack(rec_actions, axis=0)
                eef_w, R_b = compute_world_eef_trajectory(s_arr)
                T_rec = len(s_arr)
                stg_meta = {
                    **stg0,
                    "stage_idx": 0,
                    "num_stages": len(templates),
                    "start_t": 0,
                    "grasp_t": int(np.clip(grasp_t_rec, 0, T_rec - 1)),
                    "release_t": T_rec - 1,
                    "end_t": T_rec,
                    "grasp_pos_world": np.asarray(grasp_pos_w, dtype=np.float32),
                    "place_pos_world": np.asarray(place_pos_w, dtype=np.float32),
                }
                dense_meta = {
                    "task_name": task,
                    "instruction": inst,
                    "num_frames": T_rec,
                    "eef_world": eef_w.astype(np.float32),
                    "R_bases": R_b.astype(np.float32),
                    "grip_cmd": a_arr[:, 6].astype(np.float32),
                    "stages": [stg_meta],
                    "handoff_frames": list(range(T_rec)),
                }
                payload = {
                    "source": "sim_handoff",
                    "task": task,
                    "episode_index": int(saved),
                    "instruction": inst,
                    "states": s_arr,
                    "actions": a_arr,
                    "left_jpg": rec_left,
                    "right_jpg": rec_right,
                    "wrist_jpg": rec_wrist,
                    "dense_meta": dense_meta,
                }
                tmp = ep_out.with_suffix(".tmp")
                with tmp.open("wb") as fh:
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.replace(ep_out)
                saved += 1
        except Exception:
            pass
        finally:
            try:
                env.close()
            except Exception:
                pass

    return f"[sim_handoff:{task}] saved={saved}/{num_target} (attempts={attempts})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-dir", type=Path, default=Path("data/robocasa_dense_v1/human_demos"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/robocasa_dense_v1/handoff_rollouts"))
    parser.add_argument("--sim-per-task", type=int, default=6)
    parser.add_argument("--skip-sim", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    n_stage = extract_stage_handoff_episodes(args.human_dir, args.out_dir)
    print(f"[Step 2A] Extracted {n_stage} stage-specific G5j hand-off recovery episodes from human demos.", flush=True)

    if not args.skip_sim and args.sim_per_task > 0:
        sim_tasks = [
            "PickPlaceCounterToCabinet",
            "PickPlaceCounterToStove",
            "StoreLeftoversInBowl",
            "PrepareCoffee",
            "LoadDishwasher",
            "SteamInMicrowave",
            "StackBowlsCabinet",
            "CloseToasterOvenDoor",
        ]
        print(f"[Step 2B] Collecting simulator G5j hand-off rollouts across {len(sim_tasks)} tasks...", flush=True)
        worker_args = [
            (task, args.sim_per_task, idx % 4, str(args.out_dir))
            for idx, task in enumerate(sim_tasks)
        ]
        with cf.ProcessPoolExecutor(max_workers=4) as pool:
            for res in pool.map(_collect_sim_task_worker, worker_args):
                print(" ", res, flush=True)

    total = len(list(args.out_dir.glob("*.pkl")))
    print(f"HANDOFF_BUILD_DONE total_handoff_episodes={total} elapsed={time.time()-t0:.1f}s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
