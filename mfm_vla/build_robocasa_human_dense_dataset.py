"""Download and build the official human demonstration subset with Dense-Caption metadata
for the 12 Seen RoboCasa365 benchmark tasks (`Atomic-Seen` + `Composite-Seen`).

Note: `Composite-Unseen` is strictly excluded from training so zero-shot cross-task
generalization on `Composite-Unseen` remains 100% held-out.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
from pathlib import Path
import pickle
import sys
import time
from typing import List
import urllib.request

import cv2
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from robocasa_dense_caption import build_episode_dense_meta

SEEN_TASKS = [
    # Atomic-Seen (all 18 official Target-50 tasks)
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
    # Composite-Seen (all 16 official Target-50 tasks)
    "DeliverStraw",
    "GetToastedBread",
    "KettleBoiling",
    "LoadDishwasher",
    "PackIdenticalLunches",
    "PreSoakPan",
    "PrepareCoffee",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WashLettuce",
]

STATE_PERM = [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15]
ACTION_PERM = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
CAMS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]


def download_range_prefix(url: str, dest: Path, max_mb: int, proxy: str, threads: int = 12) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= 2 * 1048576:
        return dest

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with opener.open(req, timeout=25) as resp:
                final_url = resp.geturl()
                cr = resp.headers.get("Content-Range", "")
                total_bytes = int(cr.split("/")[-1]) if "/" in cr else max_mb * 1048576
            break
        except Exception as e:
            if attempt == 7:
                raise
            time.sleep(0.5 * (attempt + 1))

    target_bytes = min(total_bytes, max_mb * 1048576)
    n_chunks = (target_bytes + 1048575) // 1048576

    def fetch_chunk(i: int) -> bytes:
        s = i * 1048576
        e = min(target_bytes - 1, s + 1048576 - 1)
        for att in range(8):
            try:
                op = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
                r = urllib.request.Request(final_url, headers={"Range": f"bytes={s}-{e}"})
                with op.open(r, timeout=30) as resp:
                    return resp.read()
            except Exception:
                if att == 7:
                    raise
                time.sleep(0.4 * (att + 1))
        return b""

    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        chunks = list(ex.map(fetch_chunk, range(n_chunks)))

    tmp = dest.with_suffix(".part")
    with tmp.open("wb") as fh:
        for c in chunks:
            fh.write(c)
    tmp.replace(dest)
    return dest


def download_full_file(url: str, dest: Path, proxy: str, threads: int = 12) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    return download_range_prefix(url, dest, max_mb=200, proxy=proxy, threads=threads)


def process_task(
    task: str,
    out_dir: Path,
    cache_dir: Path,
    episodes_per_task: int,
    prefix_mb: int,
    proxy: str,
) -> str:
    t0 = time.time()
    existing = sorted(out_dir.glob(f"{task}_ep*.pkl"))
    if len(existing) >= episodes_per_task:
        return f"[{task}] skip (already have {len(existing)} episodes)"

    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    base_url = f"{hf_endpoint}/datasets/jellyho/robocasa365-{task}/resolve/main"
    task_cache = cache_dir / task
    task_cache.mkdir(parents=True, exist_ok=True)

    p_eps = download_full_file(
        f"{base_url}/meta/episodes/chunk-000/file-000.parquet",
        task_cache / "episodes.parquet",
        proxy=proxy,
        threads=4,
    )
    p_data = download_full_file(
        f"{base_url}/data/chunk-000/file-000.parquet",
        task_cache / "data.parquet",
        proxy=proxy,
        threads=12,
    )

    t_eps = pq.read_table(p_eps)
    t_data = pq.read_table(p_data)

    # Determine how many MB we need for `episodes_per_task` episodes (~0.85 MB per composite ep, ~0.25 MB per atomic ep)
    total_frames_needed = sum(
        int(t_eps["length"][i].as_py())
        for i in range(min(episodes_per_task + 2, t_eps.num_rows))
    )
    mb_needed = max(6, min(prefix_mb, int(total_frames_needed / 700) + 4))

    vid_paths = {}
    for cam in CAMS:
        short_cam = cam.split(".")[-1]
        vid_paths[cam] = download_range_prefix(
            f"{base_url}/videos/{cam}/chunk-000/file-000.mp4",
            task_cache / f"{short_cam}_{mb_needed}m.mp4",
            max_mb=mb_needed,
            proxy=proxy,
            threads=12,
        )

    caps = {cam: cv2.VideoCapture(str(vid_paths[cam])) for cam in CAMS}
    saved = 0
    frame_cursor = 0

    try:
        for ep_idx in range(min(episodes_per_task, t_eps.num_rows)):
            from_idx = int(t_eps["dataset_from_index"][ep_idx].as_py())
            to_idx = int(t_eps["dataset_to_index"][ep_idx].as_py())
            ep_len = to_idx - from_idx
            tasks_list = t_eps["tasks"][ep_idx].as_py()
            inst = str(tasks_list[0]) if tasks_list else task

            # Advance video cursors if needed
            while frame_cursor < from_idx:
                for cap in caps.values():
                    cap.grab()
                frame_cursor += 1

            cam_jpgs: dict[str, list[np.ndarray]] = {cam: [] for cam in CAMS}
            ok_ep = True
            for _ in range(ep_len):
                for cam, cap in caps.items():
                    ok, bgr = cap.read()
                    if not ok or bgr is None:
                        ok_ep = False
                        break
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                    # Encode RGB directly via cv2 (swap to BGR for imencode so imdecode + BGR2RGB yields RGB)
                    _, buf = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(rgb224, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                    cam_jpgs[cam].append(buf)
                frame_cursor += 1
                if not ok_ep:
                    break

            if not ok_ep or len(cam_jpgs[CAMS[0]]) < max(30, ep_len):
                break

            states_raw = np.stack(
                [t_data["observation.state"][i].as_py() for i in range(from_idx, to_idx)],
                dtype=np.float32,
            )
            actions_raw = np.stack(
                [t_data["action"][i].as_py() for i in range(from_idx, to_idx)],
                dtype=np.float32,
            )
            states_16d = states_raw[:, STATE_PERM]
            actions_12d = actions_raw[:, ACTION_PERM]

            dense_meta = build_episode_dense_meta(task, inst, states_16d, actions_12d)
            ep_payload = {
                "source": "human_demo",
                "task": task,
                "episode_index": int(ep_idx),
                "instruction": inst,
                "states": states_16d,
                "actions": actions_12d,
                "left_jpg": cam_jpgs["observation.images.robot0_agentview_left"],
                "right_jpg": cam_jpgs["observation.images.robot0_agentview_right"],
                "wrist_jpg": cam_jpgs["observation.images.robot0_eye_in_hand"],
                "dense_meta": dense_meta,
            }
            ep_file = out_dir / f"{task}_ep{ep_idx:03d}.pkl"
            tmp_file = ep_file.with_suffix(".tmp")
            with tmp_file.open("wb") as fh:
                pickle.dump(ep_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_file.replace(ep_file)
            saved += 1
    finally:
        for cap in caps.values():
            cap.release()

    dt = time.time() - t0
    return f"[{task}] saved {saved}/{episodes_per_task} episodes in {dt:.1f}s (mb_per_cam={mb_needed}MB)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/robocasa_dense_v1/human_demos"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/robocasa_hf_prefix_cache"))
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--prefix-mb", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY", ""))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building human demo Dense-Caption dataset for {len(SEEN_TASKS)} Seen tasks ({args.episodes_per_task} eps/task)...", flush=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(
                process_task,
                task,
                args.out_dir,
                args.cache_dir,
                args.episodes_per_task,
                args.prefix_mb,
                args.proxy,
            )
            for task in SEEN_TASKS
        ]
        for fut in cf.as_completed(futs):
            print(fut.result(), flush=True)

    total_eps = len(list(args.out_dir.glob("*.pkl")))
    print(f"HUMAN_DEMO_BUILD_DONE total_episodes={total_eps} elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
