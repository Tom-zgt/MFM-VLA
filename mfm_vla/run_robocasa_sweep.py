"""Sweep RoboCasa365 tasks across Direct (`pi05_pretrain_human300`), Forced-G5j (`g5j_assist`),
and Capability-Routed (`g5j_router`) modes, with automatic resume and summary table generation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures as futures
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

DEFAULT_BENCHMARK_TASKS = {
    "atomic_seen": [
        "CloseFridge",
        "CloseToasterOvenDoor",
        "TurnOnMicrowave",
        "TurnOffStove",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
    ],
    "composite_seen": [
        "StoreLeftoversInBowl",
        "PrepareCoffee",
        "GetToastedBread",
        "LoadDishwasher",
        "SteamInMicrowave",
        "StackBowlsCabinet",
    ],
    "composite_unseen": [
        "ArrangeBreadBasket",
        "BreadSelection",
        "GatherTableware",
        "PanTransfer",
        "WaffleReheat",
        "WeighIngredients",
    ],
}

FULL_TARGET50_TASKS = {
    "atomic_seen": [
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
    ],
    "composite_seen": [
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
    ],
    "composite_unseen": [
        "ArrangeBreadBasket",
        "ArrangeTea",
        "BreadSelection",
        "CategorizeCondiments",
        "CuttingToolSelection",
        "GarnishPancake",
        "GatherTableware",
        "HeatKebabSandwich",
        "MakeIceLemonade",
        "PanTransfer",
        "PortionHotDogs",
        "RecycleBottlesByType",
        "SeparateFreezerRack",
        "WaffleReheat",
        "WashFruitColander",
        "WeighIngredients",
    ],
}

REMAINING_32_TASKS = {
    grp: [t for t in FULL_TARGET50_TASKS[grp] if t not in DEFAULT_BENCHMARK_TASKS[grp]]
    for grp in FULL_TARGET50_TASKS
}


def load_done(output: Path) -> set[tuple[str, int, str]]:
    if not output.exists():
        return set()
    done = set()
    for line in output.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("crash"):
            done.add((str(r["task"]), int(r["episode"]), str(r["mode"])))
    return done


def run_job(
    task_group: str,
    task: str,
    ep: int,
    mode: str,
    port: int,
    seed: int,
    max_steps: int,
    artifact_root: Path,
    output: Path,
    log_dir: Path,
    dense_prompt: bool = False,
) -> str:
    name = f"{task}_ep{ep}_{mode}"
    log_path = log_dir / f"{name}.log"
    cmd = [
        sys.executable,
        str(HERE / "eval_robocasa.py"),
        "--task", task,
        "--split", "pretrain",
        "--seed", str(seed),
        "--episode", str(ep),
        "--mode", mode,
        "--port", str(port),
        "--max-steps", str(max_steps),
        "--artifact-dir", str(artifact_root / name),
        "--output", str(output),
    ]
    if dense_prompt:
        cmd.append("--dense-prompt")
    if mode == "g5j_router":
        frozen_json = HERE.parent / "configs/robocasa_task_router_frozen.json"
        if frozen_json.is_file():
            cmd.extend(["--frozen-router", str(frozen_json)])
    num_gpus = max(1, int(os.environ.get("NUM_GPUS", "1")))
    env = dict(os.environ)
    if "MUJOCO_EGL_DEVICE_ID" not in env and num_gpus > 1:
        env["MUJOCO_EGL_DEVICE_ID"] = str(port % num_gpus)
    with log_path.open("w") as fh:
        code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return f"[{task_group}] {name} exit={code}"


def summarize(output: Path) -> None:
    if not output.exists():
        return
    latest = {}
    for line in output.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("crash"):
            latest[(r["task"], int(r["episode"]), r["mode"])] = r

    task_to_group = {t: g for g, ts in FULL_TARGET50_TASKS.items() for t in ts}
    by_mode_group = defaultdict(lambda: [0, 0])
    for (task, _ep, mode), r in latest.items():
        grp = task_to_group.get(task, "other")
        by_mode_group[(mode, grp)][0] += bool(r.get("success"))
        by_mode_group[(mode, grp)][1] += 1
        by_mode_group[(mode, "OVERALL")][0] += bool(r.get("success"))
        by_mode_group[(mode, "OVERALL")][1] += 1

    print("\n=== RoboCasa365 Evaluation Summary ===")
    for mode in ("direct", "g5j_assist", "g5j_router"):
        parts = []
        for grp in ("atomic_seen", "composite_seen", "composite_unseen", "OVERALL"):
            k, n = by_mode_group[(mode, grp)]
            parts.append(f"{grp}: {k}/{n} ({k/n*100 if n else 0:5.1f}%)")
        print(f"  {mode:<12} | " + " | ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["direct", "g5j_assist", "g5j_router"])
    parser.add_argument("--groups", nargs="+", default=["atomic_seen", "composite_seen", "composite_unseen"])
    parser.add_argument("--task-set", default="default18", choices=("default18", "remaining32", "target50"))
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--ports", type=int, nargs="+", default=[8170, 8171, 8172, 8173])
    parser.add_argument("--workers-per-port", type=int, default=1)
    parser.add_argument("--artifact-root", type=Path, default=Path("reports/robocasa/artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/robocasa/results.jsonl"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/robocasa/episodes"))
    parser.add_argument("--dense-prompt", action="store_true", help="Enable dynamic 3D Dense-Caption prompt conditioning")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output)

    task_map = (
        FULL_TARGET50_TASKS
        if args.task_set == "target50"
        else (REMAINING_32_TASKS if args.task_set == "remaining32" else DEFAULT_BENCHMARK_TASKS)
    )
    jobs = []
    for mode in args.modes:
        for grp in args.groups:
            for task in task_map.get(grp, []):
                for ep in range(args.start_episode, args.start_episode + args.episodes):
                    if (task, ep, mode) not in done:
                        jobs.append((grp, task, ep, mode))

    print(f"Total pending jobs: {len(jobs)} (already done: {len(done)})", flush=True)
    slots = [p for p in args.ports for _ in range(args.workers_per_port)]
    with futures.ThreadPoolExecutor(len(slots)) as pool:
        futs = [
            pool.submit(
                run_job,
                grp,
                task,
                ep,
                mode,
                slots[idx % len(slots)],
                args.seed,
                args.max_steps,
                args.artifact_root,
                args.output,
                args.log_dir,
                args.dense_prompt,
            )
            for idx, (grp, task, ep, mode) in enumerate(jobs)
        ]
        for f in futures.as_completed(futs):
            print(f.result(), flush=True)

    summarize(args.output)


if __name__ == "__main__":
    main()
