"""Batch / single-episode evaluation driver for MFM-VLA on LIBERO-PRO.

Reads the frozen development-calibrated suite router (`configs/libero_suite_router_frozen.json`):
- 11 Direct suites -> executes `eval_g5.py --mode g5j --force-route direct` (0 MFM API calls)
- 5 Assisted suites -> executes `eval_g5.py --mode g5j --force-route assist --dense-caption`

Examples:
  # Single episode (Direct suite, zero API calls):
  python -m run_libero_pro_benchmark --suites libero_spatial_lan --tasks 0 --inits 5 --vla-port 8130

  # Held-out 800-episode evaluation split (16 suites x 10 tasks x inits 5..9):
  python -m run_libero_pro_benchmark --inits 5 6 7 8 9 --vla-port 8130 --output ./outputs/libero_pro_eval.jsonl

  # Full 8,000-episode held-out evaluation (inits 5..49):
  python -m run_libero_pro_benchmark --init-range 5 50 --vla-port 8130 --output ./outputs/libero_pro_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_ROUTER = REPO_ROOT / "configs" / "libero_suite_router_frozen.json"


def load_suite_routes(router_path: Path) -> dict[str, str]:
    data = json.loads(router_path.read_text())
    if "routes" in data:
        return {
            suite: ("assist" if route in ("assist", "g5j") else "direct")
            for suite, route in data["routes"].items()
        }
    routes: dict[str, str] = {}
    for s in data.get("direct_suites", []):
        routes[s] = "direct"
    for s in data.get("assisted_suites", []):
        routes[s] = "assist"
    return routes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MFM-VLA evaluation on LIBERO-PRO.")
    parser.add_argument("--router-config", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--suites", nargs="+", default=None, help="Suites to run (default: all 16 LIBERO-PRO suites)")
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)), help="Task indices (default: 0..9)")
    parser.add_argument("--inits", type=int, nargs="+", default=[5, 6, 7, 8, 9], help="Held-out init indices")
    parser.add_argument("--init-range", type=int, nargs=2, metavar=("START", "END"), default=None)
    parser.add_argument("--mode", default="g5j", help="Agent mode passed to eval_g5.py (default: g5j)")
    parser.add_argument("--force-route", choices=("auto", "direct", "assist"), default="auto")
    parser.add_argument("--no-dense-caption", action="store_true", help="Disable runtime Dense-v3 caption in assist mode")
    parser.add_argument("--vla-port", type=int, default=8130)
    parser.add_argument("--vla-replan-steps", type=int, default=5)
    parser.add_argument("--artifact-root", type=Path, default=Path("./outputs/libero_pro_artifacts"))
    parser.add_argument("--output", type=Path, default=Path("./outputs/libero_pro_eval.jsonl"))
    return parser.parse_args()


def load_completed(output_path: Path) -> set[tuple[str, int, int]]:
    done: set[tuple[str, int, int]] = set()
    if not output_path.is_file():
        return done
    for line in output_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("crash"):
            done.add((str(row["suite"]), int(row["task"]), int(row["init"])))
    return done


def main() -> None:
    args = parse_args()
    routes = load_suite_routes(args.router_config)
    suites = args.suites or sorted(routes.keys())
    inits = list(range(args.init_range[0], args.init_range[1])) if args.init_range else args.inits

    needs_api = any(
        (args.force_route == "assist") or (args.force_route == "auto" and routes.get(s, "assist") == "assist")
        for s in suites
    )
    if needs_api and not (os.environ.get("OPENAI_API_KEY") or os.environ.get("MFM_API_KEY")):
        raise SystemExit(
            "Error: OPENAI_API_KEY (or MFM_API_KEY) is required for MFM-assisted suites "
            f"({[s for s in suites if routes.get(s) == 'assist']})."
        )

    done = load_completed(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        (suite, task, init)
        for suite in suites
        for task in args.tasks
        for init in inits
        if (suite, task, init) not in done
    ]
    print(f"[MFM-VLA LIBERO-PRO] {len(jobs)} episodes queued (skipping {len(done)} already completed).", flush=True)

    successes = 0
    for idx, (suite, task, init) in enumerate(jobs, start=1):
        route_choice = routes.get(suite, "assist") if args.force_route == "auto" else args.force_route
        ep_artifact = args.artifact_root / suite / f"t{task:02d}_i{init:02d}"
        cmd = [
            sys.executable,
            "-u",
            str(HERE / "eval_g5.py"),
            "--suite",
            suite,
            "--task",
            str(task),
            "--init",
            str(init),
            "--mode",
            args.mode,
            "--vla-port",
            str(args.vla_port),
            "--vla-replan-steps",
            str(args.vla_replan_steps),
            "--force-route",
            route_choice,
            "--artifact-dir",
            str(ep_artifact),
            "--output",
            str(args.output),
        ]
        if route_choice == "assist" and not args.no_dense_caption:
            cmd.append("--dense-caption")

        rc = subprocess.call(cmd)
        last_succ = False
        if args.output.is_file():
            lines = [ln for ln in args.output.read_text().splitlines() if ln.strip()]
            if lines:
                last_row = json.loads(lines[-1])
                last_succ = bool(last_row.get("success"))
        successes += int(last_succ)
        print(
            f"[{idx}/{len(jobs)}] {suite} task={task} init={init} route={route_choice} "
            f"rc={rc} success={last_succ} (running SR={successes}/{idx} = {100.0 * successes / idx:.1f}%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
