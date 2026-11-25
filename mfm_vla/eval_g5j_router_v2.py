"""Evaluate one episode with the frozen direct-vs-forced-G5j capability router."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

import libero_compat  # noqa: F401
from budgeted_mfm import AtomicTokenLedger, BudgetedMFMSession
from g5j_router_v2 import FrozenG5JCapabilityRouter
from harness import TaskEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--init", type=int, required=True)
    parser.add_argument("--vla-port", type=int, required=True)
    parser.add_argument("--vla-replan-steps", type=int, default=5)
    parser.add_argument("--router-manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tool-variant", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-gpt", action="store_true")
    return parser.parse_args()


def read_last(path: Path) -> dict | None:
    if not path.is_file():
        return None
    values = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return values[-1] if values else None


def append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def initial_context(args: argparse.Namespace) -> tuple[str, dict[str, np.ndarray], dict]:
    env = TaskEnv(args.suite, args.task, img_size=128, seed=args.init)
    try:
        env.allow_goal_shortcut = False
        env.reset(args.init)
        observation = env.observe(
            ["agentview", "robot0_eye_in_hand"],
            768,
        )
        proprio = env.proprio()
        return (
            env.instruction,
            {
                "A_fixed": observation.rgb["agentview"],
                "B_wrist_initial": observation.rgb["robot0_eye_in_hand"],
            },
            {
                "eef_pos": proprio.eef_pos.tolist(),
                "eef_quat": proprio.eef_quat.tolist(),
                "gripper_opening": proprio.gripper_opening,
            },
        )
    finally:
        env.close()


def run_direct(
    args: argparse.Namespace,
    raw_output: Path,
) -> tuple[int, dict | None]:
    command = [
        sys.executable,
        "-u",
        "mfm_vla/eval_vla_only.py",
        "--suite",
        args.suite,
        "--task",
        str(args.task),
        "--inits",
        str(args.init),
        "--vla-port",
        str(args.vla_port),
        "--vla-replan-steps",
        str(args.vla_replan_steps),
        "--artifact-dir",
        str(args.artifact_dir.parent),
        "--output",
        str(raw_output),
        "--executor-name",
        f"{args.tool_variant}-direct",
    ]
    code = subprocess.call(command)
    return code, read_last(raw_output)


def run_g5j(
    args: argparse.Namespace,
    raw_output: Path,
    program: dict,
) -> tuple[int, dict | None]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="g5j_router_program.",
        delete=False,
    ) as handle:
        json.dump(program, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        program_path = Path(handle.name)
    try:
        command = [
            sys.executable,
            "-u",
            "mfm_vla/eval_g5.py",
            "--suite",
            args.suite,
            "--task",
            str(args.task),
            "--init",
            str(args.init),
            "--mode",
            "g5j_forced_v2",
            "--vla-port",
            str(args.vla_port),
            "--vla-replan-steps",
            str(args.vla_replan_steps),
            "--force-route",
            "assist",
            "--cached-program",
            str(program_path),
            "--artifact-dir",
            str(args.artifact_dir),
            "--output",
            str(raw_output),
        ]
        if args.tool_variant == "dense-v3-hybrid":
            command.append("--dense-caption")
        code = subprocess.call(command)
        return code, read_last(raw_output)
    finally:
        program_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    brain = None
    row = None
    try:
        ledger = AtomicTokenLedger(args.ledger)
        if not args.no_gpt:
            brain = BudgetedMFMSession(
                ledger,
                args.cache_dir,
                max_tokens=1000,
                reserve_tokens=8000,
            )
        router = FrozenG5JCapabilityRouter(
            args.router_manifest,
            brain=brain,
        )
        if router.manifest["tool_id"] != args.tool_variant:
            raise ValueError("tool variant does not match frozen G5j router")
        instruction, images, proprioception = initial_context(args)
        decision = router.route(instruction, images, proprioception)
        _, program, _ = router.cached(instruction)

        with tempfile.TemporaryDirectory(prefix="g5j_router_result.") as directory:
            raw_output = Path(directory) / "result.jsonl"
            if decision["route"] == "g5j":
                code, raw = run_g5j(args, raw_output, program)
            else:
                code, raw = run_direct(args, raw_output)
        if raw is None:
            raise RuntimeError(
                f"{decision['route']} executor exited {code} without a result"
            )
        route_call = None
        if brain is not None and brain.last_call is not None:
            usage = brain.last_call.get("usage") or {}
            route_call = {
                "input_hash": brain.last_call["input_hash"],
                "cache_hit": bool(brain.last_call["cache_hit"]),
                "actual_tokens": (
                    0
                    if brain.last_call["cache_hit"]
                    else int(usage.get("total_tokens", 0) or 0)
                ),
                "reported_tokens": int(usage.get("total_tokens", 0) or 0),
            }
        row = {
            **raw,
            "schema_version": 2,
            "algorithm": "G5J-Capability-Router-v2",
            "tool_variant": args.tool_variant,
            "route": decision["route"],
            "external_route": decision,
            "route_gpt_call": route_call,
            "g5j_gpt_usage": (
                raw.get("gpt_usage") or {}
                if decision["route"] == "g5j"
                else {}
            ),
            "split": "capability-dev" if args.init <= 4 else "held-out",
            "crash": bool(raw.get("crash")),
        }
    except Exception as exc:
        row = {
            "schema_version": 2,
            "algorithm": "G5J-Capability-Router-v2",
            "tool_variant": args.tool_variant,
            "suite": args.suite,
            "task": args.task,
            "init": args.init,
            "split": "capability-dev" if args.init <= 4 else "held-out",
            "success": False,
            "crash": True,
            "reason": f"{type(exc).__name__}: {exc}",
            "detail": traceback.format_exc(limit=12),
            "artifacts": str(args.artifact_dir),
        }
    finally:
        if brain is not None:
            brain.close()
    row["duration_s_router_total"] = round(time.monotonic() - started, 2)
    append(args.output, row)
    print(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str))
    if row.get("crash"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
