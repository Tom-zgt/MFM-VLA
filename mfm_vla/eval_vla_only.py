"""Evaluate π0.5 without a code-agent supervisor on selected LIBERO tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import libero_compat  # noqa: F401

from harness import TaskEnv
from libero_pro import LIBERO_PRO_PERTURBATIONS
from rollout_recorder import RolloutRecorder, default_staging_root
from vla_executor import OpenPiVLAExecutor, VLAConfig
from vp_markers import OracleMarker
from dense_caption_provider import OracleDenseCaptioner


SUITE_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_spatial_swap": 220,
    "libero_spatial_task": 220,
}
for _suite in LIBERO_PRO_PERTURBATIONS:
    _base = _suite.rsplit("_", 1)[0]
    SUITE_HORIZONS[_suite] = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }[_base]


def recover_completed_artifact(artifact_dir: Path) -> dict | None:
    """Rebuild a result row if artifacts committed before the JSONL append."""
    recorder = RolloutRecorder(artifact_dir)
    if not recorder.status_path.is_file() or not recorder.metadata_path.is_file():
        return None
    status = json.loads(recorder.status_path.read_text())
    if status.get("state") != "completed":
        return None
    metadata = json.loads(recorder.metadata_path.read_text())
    decisions = []
    if recorder.trace_path.is_file():
        for line in recorder.trace_path.read_text().splitlines():
            record = json.loads(line)
            if str(record.get("mode", "")).startswith("pi05_proposal_"):
                decisions.append(int(record.get("decision", -1)))
    videos = {
        name: str(path)
        for name, path in {
            "agentview": artifact_dir / "agentview.mp4",
            "robot0_eye_in_hand": artifact_dir / "robot0_eye_in_hand.mp4",
            "both": artifact_dir / "both.mp4",
        }.items()
        if path.is_file()
    }
    return {
        "suite": metadata["suite"],
        "benchmark": (
            "libero-pro"
            if metadata["suite"] in LIBERO_PRO_PERTURBATIONS
            else "libero"
        ),
        "perturbation": LIBERO_PRO_PERTURBATIONS.get(metadata["suite"]),
        "task": int(metadata["task"]),
        "init": int(metadata["init"]),
        "executor": "pi05_libero",
        "brain": False,
        "benchmark_stop_on_success": True,
        "success": bool(status.get("success")),
        "steps": int(status.get("step", 0)),
        "vla_inference_calls": max(decisions, default=-1) + 1,
        "artifacts": str(artifact_dir),
        "reason": str(status.get("reason", "recovered completed artifact")),
        "duration_s": None,
        "crash": False,
        "recovered_from_artifacts": True,
        "videos": videos,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=tuple(SUITE_HORIZONS), required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--inits", default="0")
    parser.add_argument("--vla-host", default="127.0.0.1")
    parser.add_argument("--vla-port", type=int, default=8000)
    parser.add_argument("--vla-replan-steps", type=int, default=5)
    parser.add_argument(
        "--benchmark-stop-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match official LIBERO evaluation by stopping on its hidden predicate.",
    )
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--replace-existing-artifacts",
        action="store_true",
        help="Discard an incomplete artifact directory before rerunning its missing row.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-name", default="pi05_libero")
    parser.add_argument(
        "--marks", choices=("none", "oracle"), default="none",
        help="Visual prompt drawn on the agentview image (for pi05_libero_vp).",
    )
    parser.add_argument(
        "--dense-caption", choices=("none", "oracle", "oracle_noisy"), default="none",
        help="Dynamic 10-step Dense Caption conditioning at every replan step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["HOI_DISABLE_MEMORY_WRITE"] = "1"
    for init in (int(value) for value in args.inits.split(",") if value.strip()):
        artifact_base = args.artifact_dir or (
            args.output.parent / f"{args.output.stem}_artifacts"
        )
        artifact_dir = artifact_base / f"{args.suite}_t{args.task}_i{init}"
        recovered = recover_completed_artifact(artifact_dir)
        if recovered is not None:
            expected = {"suite": args.suite, "task": args.task, "init": init}
            if any(recovered[key] != value for key, value in expected.items()):
                raise ValueError(
                    f"completed artifact identity mismatch at {artifact_dir}"
                )
            with args.output.open("a") as handle:
                handle.write(json.dumps(recovered, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(
                f"vla-only {args.suite} t{args.task} i{init}: "
                "RECOVERED completed episode artifact",
                flush=True,
            )
            continue
        if args.replace_existing_artifacts and artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        env = None
        executor = None
        recorder = None
        row = None
        started = time.monotonic()
        try:
            env = TaskEnv(args.suite, args.task, img_size=128, seed=init)
            env.allow_goal_shortcut = False
            env.reset(init)
            executor = OpenPiVLAExecutor(
                VLAConfig(
                    host=args.vla_host,
                    port=args.vla_port,
                    resize_size=224,
                    replan_steps=args.vla_replan_steps,
                    max_skill_steps=SUITE_HORIZONS[args.suite],
                    prompt_mode="task",
                )
            )
            marker = OracleMarker() if args.marks == "oracle" else None
            executor.image_marker = marker
            captioner = None
            if args.dense_caption in ("oracle", "oracle_noisy"):
                captioner = OracleDenseCaptioner(
                    pos_noise_std_m=0.025 if args.dense_caption == "oracle_noisy" else 0.0,
                    seed=init,
                )
                executor.prompt_builder = captioner
            recorder = RolloutRecorder(artifact_dir, staging_root=default_staging_root())
            recorder.begin(
                {
                    "suite": args.suite,
                    "task": args.task,
                    "init": init,
                    "instruction": env.instruction,
                    "executor": "pi05_libero",
                    "supervisor": None,
                    "vla_replan_steps": args.vla_replan_steps,
                    "horizon": SUITE_HORIZONS[args.suite],
                },
                env,
            )
            result = executor.execute_task(
                env,
                max_steps=SUITE_HORIZONS[args.suite],
                benchmark_stop_on_success=args.benchmark_stop_on_success,
                step_callback=lambda step_env, action, inference_call: (
                    recorder.record_step(
                        step_env,
                        action=action,
                        mode=f"pi05_proposal_{inference_call}",
                        decision=inference_call - 1,
                    )
                ),
            )
            row = {
                "suite": args.suite,
                "benchmark": (
                    "libero-pro" if args.suite in LIBERO_PRO_PERTURBATIONS else "libero"
                ),
                "perturbation": LIBERO_PRO_PERTURBATIONS.get(args.suite),
                "task": args.task,
                "init": init,
                "executor": args.executor_name,
                "marks": args.marks,
                "mark_history": marker.history if marker is not None else None,
                "dense_caption": args.dense_caption,
                "caption_history": captioner.history if captioner is not None else None,
                "brain": False,
                "benchmark_stop_on_success": args.benchmark_stop_on_success,
                "success": bool(env.success),
                "steps": env.t,
                "vla_inference_calls": executor.inference_calls,
                "artifacts": str(artifact_dir),
                "reason": result.reason,
                "duration_s": round(time.monotonic() - started, 3),
                "crash": False,
            }
        except Exception as exc:
            row = {
                "suite": args.suite,
                "benchmark": (
                    "libero-pro" if args.suite in LIBERO_PRO_PERTURBATIONS else "libero"
                ),
                "perturbation": LIBERO_PRO_PERTURBATIONS.get(args.suite),
                "task": args.task,
                "init": init,
                "executor": "pi05_libero",
                "brain": False,
                "benchmark_stop_on_success": args.benchmark_stop_on_success,
                "success": False,
                "steps": env.t if env is not None else 0,
                "vla_inference_calls": (
                    executor.inference_calls if executor is not None else 0
                ),
                "artifacts": str(recorder.final_root) if recorder is not None else None,
                "reason": f"{type(exc).__name__}: {exc}",
                "detail": traceback.format_exc(limit=10),
                "duration_s": round(time.monotonic() - started, 3),
                "crash": True,
            }
        finally:
            if executor is not None:
                executor.close()
            if env is not None:
                if args.video_dir is not None:
                    args.video_dir.mkdir(parents=True, exist_ok=True)
                    base = args.video_dir / (
                        f"pi05_only_{args.suite}_t{args.task}_i{init}.mp4"
                    )
                    if env.save_video(str(base)) is not None and row is not None:
                        row["video"] = str(base)
                if recorder is not None:
                    videos = recorder.finalize(
                        "completed" if row is not None and not row["crash"] else "interrupted",
                        step=env.t,
                        success=bool(row and row.get("success")),
                        reason=row.get("reason") if row else "initialization failed",
                    )
                    if row is not None:
                        row["videos"] = videos
                env.close()
        assert row is not None
        with args.output.open("a") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(
            f"vla-only {args.suite} t{args.task} i{init}: "
            f"{'OK' if row['success'] else 'FAIL'} {row['reason']}",
            flush=True,
        )
        if row["crash"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
