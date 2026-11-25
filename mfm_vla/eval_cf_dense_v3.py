"""Evaluate fine-tuned pi05_libero_dense_v3 (and MFM-VLA) on the exact 8-suite
Shortcut Detection benchmark (Section 3.2, Table 2) matching `reports/research/cf/`.

Suites (50 episodes each = 400 episodes total):
  - libero_object_lan, libero_object_task
  - libero_spatial_lan, libero_spatial_task
  - libero_goal_lan, libero_goal_task
  - libero_10_lan, libero_10_task

Conditions evaluated for `pi05_dense_v3` (pure VLA, 0 VLM calls):
  - `own`        : raw new instruction `l_new`
  - `own_dense`  : prefix-free 4-field dense geometric caption `c_t` (noisy 2.5cm)
  - `null`       : empty string `""`
  - `train`      : old training instruction `l_train` (for the 6 suites with `train`)

Plus `MFM-VLA` (Routed `Dense-v3 Direct` + `G5j + Dense-v3` on the exact same 400 episodes),
computing the exact same 5 Table-2 metrics (`Succ`, `Lift o_new`, `Lift o_train`,
`Null Lifts o_train`, `Same First Lift`).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mfm_vla"))
sys.path.insert(0, str(ROOT / "LIBERO"))

import libero_compat  # noqa: F401
from dense_caption_provider import OracleDenseCaptioner
from eval_preposition_pick import LiftTracker
from eval_vla_only import SUITE_HORIZONS
from harness import TaskEnv
from libero.libero import benchmark
from libero_pro import get_libero_pro_task, is_libero_pro_suite
from vla_executor import OpenPiVLAExecutor, VLAConfig


def run_single_cf_job(
    suite: str,
    task: int,
    init: int,
    condition: str,
    prompt: str,
    orig_task: int,
    port: int,
) -> dict:
    os.environ["HARNESS_SETTLE_OPEN"] = "1"
    horizon = SUITE_HORIZONS.get(suite) or SUITE_HORIZONS[suite.rsplit("_", 1)[0]]
    env = TaskEnv(suite, task, img_size=128, seed=init)
    env.allow_goal_shortcut = False
    env.reset(init)

    tracker = LiftTracker(env)
    path_xyz: list[list[float]] = []
    first_close_t = None

    executor = OpenPiVLAExecutor(VLAConfig(
        port=port,
        resize_size=224,
        replan_steps=5,
        max_skill_steps=horizon,
        prompt_mode="task",
    ))

    if condition == "own_dense":
        captioner = OracleDenseCaptioner(
            pos_noise_std_m=0.025,
            include_task_prefix=False,
            seed=suite.__hash__() % 10000 + task * 100 + init,
        )
        executor.prompt_builder = captioner
    else:
        # Force exact string prompt (including "" for condition == "null")
        executor.prompt_builder = lambda _e, _o, _d: prompt

    def on_step(s_env: TaskEnv, action: np.ndarray, _call: int) -> None:
        nonlocal first_close_t
        tracker.update(s_env)
        p = s_env.proprio().eef_pos
        path_xyz.append([round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)])
        if first_close_t is None and float(action[6]) > 0.5:
            first_close_t = int(s_env.t)

    try:
        executor.execute_prompt(
            env,
            prompt=prompt,
            max_steps=horizon,
            benchmark_stop_on_success=True,
            step_callback=on_step,
        )
    finally:
        executor.close()
        succ = bool(env.success)
        steps = int(env.t)
        env.close()

    return {
        "suite": suite,
        "task": int(task),
        "init": int(init),
        "condition": condition,
        "prompt": prompt,
        "orig_task": int(orig_task),
        "success": succ,
        "steps": steps,
        "first_lifted": tracker.first_lifted(),
        "max_lift_m": {k: round(float(v), 4) for k, v in tracker.max_lift.items() if v > 0.005},
        "first_close_t": first_close_t,
        "first_eef_10": path_xyz[:10],
        "crash": False,
    }


def load_cf_reference_jobs() -> list[dict]:
    """Extract unique (suite, task, init, condition) jobs from `reports/research/cf/part*.jsonl`."""
    jobs_map = {}
    ep_meta = {}
    for p in sorted(glob.glob(str(ROOT / "reports/research/cf/part*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            cond = r["condition"]
            if cond == "own_repeat":
                continue
            key = (r["suite"], int(r["task"]), int(r["init"]), cond)
            jobs_map[key] = {
                "suite": r["suite"],
                "task": int(r["task"]),
                "init": int(r["init"]),
                "condition": cond,
                "prompt": r["prompt"],
                "orig_task": int(r["orig_task"]),
            }
            if cond == "own":
                ep_meta[(r["suite"], int(r["task"]), int(r["init"]))] = {
                    "suite": r["suite"],
                    "task": int(r["task"]),
                    "init": int(r["init"]),
                    "condition": "own_dense",
                    "prompt": r["prompt"],
                    "orig_task": int(r["orig_task"]),
                }
    for k_ep, meta in ep_meta.items():
        jobs_map[(k_ep[0], k_ep[1], k_ep[2], "own_dense")] = meta

    return [jobs_map[k] for k in sorted(jobs_map.keys())]


def compute_and_print_summary(out_jsonl: Path, summary_json: Path) -> None:
    bms = {}

    def parse_first_goal_target(suite: str, task_id: int) -> str | None:
        if is_libero_pro_suite(suite):
            bddl_path = get_libero_pro_task(suite, task_id).bddl_path
        else:
            if suite not in bms:
                bms[suite] = benchmark.get_benchmark_dict()[suite]()
            bddl_path = Path(bms[suite].get_task_bddl_file_path(task_id))
        txt = bddl_path.read_text()
        goal_part = txt.split("(:goal")[-1]
        matches = re.findall(r"\((?:In|On)\s+([^\s\)]+)", goal_part, re.IGNORECASE)
        return matches[0] if matches else None

    # Load Dense-v3 CF runs
    by_suite_ep = defaultdict(lambda: defaultdict(dict))
    for line in out_jsonl.read_text().splitlines():
        r = json.loads(line)
        if not r.get("crash"):
            by_suite_ep[r["suite"]][(int(r["task"]), int(r["init"]))][r["condition"]] = r

    # Also load G5j + Dense-v3 (NEED Guided) results on the exact same (suite, task, init) episodes
    g5j_map = {}
    for line in (ROOT / "reports/generic/g5j_full_1600_pi05_dense_v3_raw.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not r.get("crash"):
            g5j_map[(r["suite"], int(r["task"]), int(r["init"]))] = r

    G5_SUITES = {
        "libero_goal_swap",
        "libero_goal_task",
        "libero_object_swap",
        "libero_object_task",
        "libero_spatial_task",
    }

    orig_pi05 = json.loads((ROOT / "reports/research/cf_summary.json").read_text())
    summary_out = {}

    print("\n" + "=" * 115)
    print("SHORTCUT DETECTION FULL COMPARISON (8 Suites x 50 Eps = 400 Paired Episodes)")
    print("=" * 115)

    for s in sorted(by_suite_ep.keys()):
        base_suite = s.rsplit("_", 1)[0]
        eps = sorted(by_suite_ep[s].keys())

        def eval_mode(mode_tag: str):
            succ, lift_own, lift_trn, null_trn, same_null = [], [], [], [], []
            for (t, i) in eps:
                c = by_suite_ep[s][(t, i)]
                null_row = c["null"]
                orig_t = int(c["own"]["orig_task"])
                own_tgt = parse_first_goal_target(s, t)
                trn_tgt = parse_first_goal_target(base_suite, orig_t)
                f_null = null_row.get("first_lifted")

                if mode_tag == "dense_raw":
                    row = c["own"]
                elif mode_tag == "dense_caption":
                    row = c["own_dense"]
                elif mode_tag == "need":
                    row = g5j_map[(s, t, i)] if s in G5_SUITES else c["own"]
                else:
                    raise ValueError(mode_tag)

                f_own = row.get("first_lifted")
                succ.append(bool(row["success"]))
                lift_own.append((f_own == own_tgt) if own_tgt else False)
                lift_trn.append((f_own == trn_tgt) if (trn_tgt and trn_tgt != own_tgt) else False)
                null_trn.append((f_null == trn_tgt) if trn_tgt else False)
                same_null.append(f_own == f_null)

            return {
                "n": len(eps),
                "success_own": round(100.0 * float(np.mean(succ)), 1),
                "own_lifts_own_target": round(100.0 * float(np.mean(lift_own)), 1),
                "own_lifts_train_target": round(100.0 * float(np.mean(lift_trn)), 1),
                "null_lifts_train_target": round(100.0 * float(np.mean(null_trn)), 1),
                "same_first_lift_own_null": round(100.0 * float(np.mean(same_null)), 1),
            }

        summary_out[s] = {
            "base_pi05": orig_pi05[s],
            "adapted_pi05_raw_prompt": eval_mode("dense_raw"),
            "adapted_pi05_dense_caption": eval_mode("dense_caption"),
            "need_full": eval_mode("need"),
        }

        b = summary_out[s]["base_pi05"]
        dr = summary_out[s]["adapted_pi05_raw_prompt"]
        dc = summary_out[s]["adapted_pi05_dense_caption"]
        nd = summary_out[s]["need_full"]
        print(f"\n[{s}] (N={len(eps)})")
        for label, m in [
            ("Base pi0.5", b),
            ("Adapted pi0.5 (Raw l_new)", dr),
            ("Adapted pi0.5 (Dense c_t)", dc),
            ("MFM-VLA (Full Framework)", nd),
        ]:
            print(
                f"  {label:28s} | Succ={m['success_own']:5.1f}% | "
                f"Lift o_new={m['own_lifts_own_target']:5.1f}% | "
                f"Lift o_train={m['own_lifts_train_target']:5.1f}% | "
                f"Null->o_train={m['null_lifts_train_target']:5.1f}% | "
                f"SameLift={m['same_first_lift_own_null']:5.1f}%"
            )

    summary_json.write_text(json.dumps(summary_out, indent=2))
    print(f"\nSaved full summary JSON to {summary_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite")
    parser.add_argument("--task", type=int)
    parser.add_argument("--init", type=int)
    parser.add_argument("--condition")
    parser.add_argument("--prompt")
    parser.add_argument("--orig-task", type=int)
    parser.add_argument("--vla-port", type=int, default=8250)
    parser.add_argument("--output", type=Path, required=True)
    # Batch driver args
    parser.add_argument("--ports", type=int, nargs="+")
    parser.add_argument("--workers-per-port", type=int, default=2)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    if args.suite is not None:
        try:
            out = run_single_cf_job(
                args.suite,
                args.task,
                args.init,
                args.condition,
                args.prompt,
                args.orig_task,
                args.vla_port,
            )
        except Exception as exc:
            out = {
                "suite": args.suite,
                "task": int(args.task),
                "init": int(args.init),
                "condition": args.condition,
                "prompt": args.prompt,
                "orig_task": int(args.orig_task),
                "success": False,
                "steps": 0,
                "reason": f"{type(exc).__name__}: {exc}",
                "detail": traceback.format_exc(limit=8),
                "crash": True,
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as fh:
            fh.write(json.dumps(out, separators=(",", ":")) + "\n")
            fh.flush()
        if out.get("crash"):
            raise SystemExit(2)
        return

    # Batch driver mode across all 1,500 jobs
    assert args.ports and args.log_dir
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_jobs = load_cf_reference_jobs()

    for _ in range(3):
        done = set()
        if args.output.exists():
            for line in args.output.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("crash"):
                    done.add((r["suite"], int(r["task"]), int(r["init"]), r["condition"]))

        todo = [j for j in all_jobs if (j["suite"], j["task"], j["init"], j["condition"]) not in done]
        print(f"CF_DENSE_V3: {len(todo)}/{len(all_jobs)} jobs to run", flush=True)
        if not todo:
            break

        work: queue.Queue = queue.Queue()
        for j in todo:
            work.put(j)
        lock = threading.Lock()
        done_counter = [len(all_jobs) - len(todo)]

        def worker(port: int, wid: int) -> None:
            while True:
                try:
                    j = work.get_nowait()
                except queue.Empty:
                    return
                cmd = [
                    sys.executable,
                    "mfm_vla/eval_cf_dense_v3.py",
                    "--suite", j["suite"],
                    "--task", str(j["task"]),
                    "--init", str(j["init"]),
                    "--condition", j["condition"],
                    "--prompt", j["prompt"],
                    "--orig-task", str(j["orig_task"]),
                    "--vla-port", str(port),
                    "--output", str(args.output),
                ]
                env = dict(os.environ, MUJOCO_EGL_DEVICE_ID=str((port + wid) % 4))
                log_f = args.log_dir / f"{j['suite']}_t{j['task']}_i{j['init']}_{j['condition']}.log"
                with log_f.open("w") as fh:
                    code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
                with lock:
                    done_counter[0] += (1 if code == 0 else 0)
                    if done_counter[0] % 25 == 0 or code != 0:
                        print(
                            f"Progress {done_counter[0]}/{len(all_jobs)} | "
                            f"{j['suite']} t{j['task']} i{j['init']} [{j['condition']}] exit={code}",
                            flush=True,
                        )

        threads = [
            threading.Thread(target=worker, args=(p, w))
            for p in args.ports
            for w in range(args.workers_per_port)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    if args.summary_json:
        compute_and_print_summary(args.output, args.summary_json)
    print("CF_DENSE_V3_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
