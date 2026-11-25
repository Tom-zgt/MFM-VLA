"""Capability learning for routing between direct π0.5 and forced G5j."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw


def rows(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    result = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return result


def atomic_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_id(cluster_id: str, tool_id: str, repeat_index: int) -> str:
    encoded = json.dumps(
        [cluster_id, tool_id, "g5j_forced_assist", repeat_index],
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()[:20]


def plan(args: argparse.Namespace) -> None:
    clusters = {value["cluster_id"]: value for value in rows(args.clusters)}
    hypotheses = {value["cluster_id"]: value for value in rows(args.hypotheses)}
    programs = {value["instruction_id"]: value for value in rows(args.programs)}
    old = {value["probe_id"]: value for value in rows(args.existing_specs)}
    reviews = {value["cluster_id"]: value for value in rows(args.reviews)}
    selected = sorted(
        (
            clusters[cluster_id]
            for cluster_id in hypotheses
            if cluster_id in clusters
        ),
        key=lambda value: (-int(value["count"]), value["cluster_id"]),
    )
    if args.max_clusters is not None:
        selected = selected[: args.max_clusters]

    specs = dict(old)
    for cluster in selected:
        cluster_id = cluster["cluster_id"]
        existing = [
            value
            for value in specs.values()
            if value["cluster_id"] == cluster_id
        ]
        if args.round_index == 0:
            repeat_index = 0
        else:
            review = reviews.get(cluster_id)
            if (
                review is None
                or review.get("next_experiment") != "repeat_g5j"
                or len(existing) >= args.max_repeats
            ):
                continue
            repeat_index = max(
                (int(value.get("repeat_index", 0)) for value in existing),
                default=-1,
            ) + 1
        identifier = probe_id(cluster_id, args.tool_id, repeat_index)
        if identifier in specs:
            continue
        rep = cluster["representative"]
        program = programs.get(rep["instruction_id"])
        if program is None:
            raise ValueError(
                f"missing cached program for {rep['instruction_id']}"
            )
        if int(rep["init"]) > 4:
            raise ValueError("G5j probe planner selected a held-out init")
        specs[identifier] = {
            "schema_version": 2,
            "algorithm": "G5J-Capability-Router-v2",
            "split": "capability-dev",
            "tool_id": args.tool_id,
            "round_index": args.round_index,
            "repeat_index": repeat_index,
            "cluster_id": cluster_id,
            "probe_id": identifier,
            "probe_type": "g5j_forced_assist",
            "suite": rep["suite"],
            "task": int(rep["task"]),
            "init": int(rep["init"]),
            "instruction_id": rep["instruction_id"],
            "instruction": rep["instruction"],
            "program": program,
            "direct_baseline": {
                key: rep.get(key)
                for key in (
                    "success",
                    "steps",
                    "failure_stage",
                    "first_close_step",
                    "max_eef_lift_after_close_m",
                    "target_max_movement_m",
                    "target_max_lift_m",
                    "predicate_max",
                    "predicate_total",
                    "artifact",
                )
            },
        }
    output = [specs[key] for key in sorted(specs)]
    atomic_jsonl(args.output, output)
    print(
        f"G5J_V2_PLANNED specs={len(output)} clusters={len(selected)} "
        f"round={args.round_index}",
        flush=True,
    )


REVIEW_PROMPT = """You are learning how to route a robot task between two tools:
(A) direct π0.5 and (B) the original G5j assist executor with only G5j's own
direct/assist router removed. G5j still uses its full two-view GPT grounding,
grasp verification/regrasp, drop recovery, carry progress watch, and
Watch-and-Takeover PlaceLoop.

Use the direct baseline and every same-init forced-G5j experiment below. Actual
success/failure overrides verbal expectations. Infer conditions only from the
instruction, initial RGB, proprioception, and observable behavior; never use
suite labels, simulator coordinates, BDDL, or hidden success predicates as
future router inputs.

Task pattern:
{program}
Direct failure hypothesis:
{hypothesis}
Direct baseline:
{direct}
Forced-G5j outcomes:
{g5j}

Return JSON only:
{{"conclusion":"...","direct_capability":"...","g5j_capability":"...",
"route_recommendation":"direct|g5j|uncertain",
"applicable_when":["observable conditions"],
"unsupported_when":["observable conditions"],
"remaining_uncertainty":"...",
"next_experiment":"repeat_g5j|stop",
"stop_reason":"...","confidence":0.0}}"""


def nearest(directory: Path, last: bool) -> Path:
    paths = sorted(directory.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no frames in {directory}")
    return paths[-1] if last else paths[0]


def montage(
    cluster: dict,
    outcomes: list[dict],
    destination: Path,
) -> Path:
    branches = [
        ("direct", Path(cluster["representative"]["artifact"]))
    ] + [
        (f"g5j-r{value.get('repeat_index', 0)}", Path(value["artifacts"]))
        for value in sorted(
            outcomes,
            key=lambda value: int(value.get("repeat_index", 0)),
        )[-3:]
        if not value.get("crash")
    ]
    canvas = Image.new("RGB", (256 * len(branches), 512))
    for column, (label, artifact) in enumerate(branches):
        for row_index, camera in enumerate(("agentview", "robot0_eye_in_hand")):
            directory = artifact / "frames" / camera
            start = Image.open(nearest(directory, False)).convert("RGB").resize((128, 256))
            end = Image.open(nearest(directory, True)).convert("RGB").resize((128, 256))
            image = Image.new("RGB", (256, 256))
            image.paste(start, (0, 0))
            image.paste(end, (128, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 255, 24), fill=(0, 0, 0))
            draw.text((4, 5), f"{label} {camera}: start | end", fill=(255, 220, 0))
            canvas.paste(image, (256 * column, 256 * row_index))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=90)
    return destination


def review(args: argparse.Namespace) -> None:
    from budgeted_mfm import AtomicTokenLedger, BudgetedMFMSession

    clusters = {value["cluster_id"]: value for value in rows(args.clusters)}
    hypotheses = {value["cluster_id"]: value for value in rows(args.hypotheses)}
    programs = {value["instruction_id"]: value for value in rows(args.programs)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for value in rows(args.probes):
        if int(value["init"]) > 4:
            raise ValueError("G5j review input contains held-out data")
        if not value.get("crash"):
            grouped[value["cluster_id"]].append(value)
    output = args.output
    existing = {value["cluster_id"]: value for value in rows(output)}
    ledger = AtomicTokenLedger(args.ledger)
    brain = BudgetedMFMSession(
        ledger,
        args.cache_dir,
        max_tokens=args.max_tokens,
        reserve_tokens=8000,
    )
    try:
        for cluster_id, outcomes in sorted(grouped.items()):
            identifiers = sorted(value["probe_id"] for value in outcomes)
            if (
                cluster_id in existing
                and existing[cluster_id].get("probe_ids") == identifiers
            ):
                continue
            cluster = clusters[cluster_id]
            rep = cluster["representative"]
            image = montage(
                cluster,
                outcomes,
                args.keyframe_dir / f"{cluster_id}.jpg",
            )
            direct = {
                key: rep.get(key)
                for key in (
                    "success",
                    "steps",
                    "failure_stage",
                    "first_close_step",
                    "max_eef_lift_after_close_m",
                    "target_max_movement_m",
                    "target_max_lift_m",
                    "predicate_max",
                    "predicate_total",
                )
            }
            g5j = [
                {
                    "probe_id": value["probe_id"],
                    "repeat_index": value.get("repeat_index"),
                    "success": bool(value["success"]),
                    "steps": value.get("steps"),
                    "move_steps": value.get("move_steps"),
                    "pi05_steps": value.get("pi05_steps"),
                    "first_lifted_is_target": value.get(
                        "first_lifted_is_target"
                    ),
                    "reason": value.get("reason"),
                    "gpt_usage": value.get("gpt_usage"),
                }
                for value in outcomes
            ]
            decision = brain.decide_json(
                REVIEW_PROMPT.format(
                    program=json.dumps(
                        programs[rep["instruction_id"]],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    hypothesis=json.dumps(
                        hypotheses[cluster_id],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    direct=json.dumps(direct, separators=(",", ":")),
                    g5j=json.dumps(g5j, separators=(",", ":")),
                ),
                {"direct_vs_forced_g5j": np.asarray(Image.open(image))},
                bucket="discovery",
                reason=f"g5j_v2_review:{args.tool_id}:{cluster_id}",
            )
            if decision.get("route_recommendation") not in {
                "direct",
                "g5j",
                "uncertain",
            }:
                raise ValueError("invalid G5j route recommendation")
            if decision.get("next_experiment") not in {
                "repeat_g5j",
                "stop",
            }:
                raise ValueError("invalid G5j next experiment")
            existing[cluster_id] = {
                "schema_version": 2,
                "algorithm": "G5J-Capability-Router-v2",
                "tool_id": args.tool_id,
                "cluster_id": cluster_id,
                "instruction_id": rep["instruction_id"],
                "member_instruction_ids": cluster["member_instruction_ids"],
                "probe_ids": identifiers,
                "keyframes": str(image),
                "status": "gpt_reviewed",
                **decision,
            }
            atomic_jsonl(
                output,
                [existing[key] for key in sorted(existing)],
            )
            print(
                f"G5J_V2_REVIEWED {len(existing)}/{len(grouped)} "
                f"tokens={ledger.snapshot()['used_tokens']}",
                flush=True,
            )
    finally:
        brain.close()


def beta_summary(successes: int, trials: int) -> dict:
    return {
        "trials": trials,
        "successes": successes,
        "failures": trials - successes,
        "posterior_alpha": 1 + successes,
        "posterior_beta": 1 + trials - successes,
        "posterior_mean": round((1 + successes) / (2 + trials), 6),
    }


def make_records(args: argparse.Namespace) -> None:
    features_by_instruction: dict[str, list[dict]] = defaultdict(list)
    for value in rows(args.features):
        features_by_instruction[value["instruction_id"]].append(value)
    clusters = {value["cluster_id"]: value for value in rows(args.clusters)}
    probes_by_cluster: dict[str, list[dict]] = defaultdict(list)
    for value in rows(args.probes):
        if not value.get("crash"):
            probes_by_cluster[value["cluster_id"]].append(value)
    reviews = {value["cluster_id"]: value for value in rows(args.reviews)}

    evidence_by_instruction: dict[str, list[tuple[dict, dict, dict]]] = defaultdict(list)
    for cluster_id, values in probes_by_cluster.items():
        cluster = clusters[cluster_id]
        review_value = reviews.get(cluster_id, {})
        for instruction_id in cluster["member_instruction_ids"]:
            for value in values:
                evidence_by_instruction[instruction_id].append(
                    (cluster, value, review_value)
                )

    output = []
    for instruction_id in sorted(features_by_instruction):
        direct_rows = features_by_instruction[instruction_id]
        evidence = evidence_by_instruction.get(instruction_id, [])
        g5j_rows_by_id = {
            value["probe_id"]: value
            for _, value, _ in evidence
        }
        g5j_rows = list(g5j_rows_by_id.values())
        review_values = {
            review_value["cluster_id"]: review_value
            for _, _, review_value in evidence
            if review_value
        }
        recommendations = [
            value.get("route_recommendation", "uncertain")
            for value in review_values.values()
        ]
        output.append(
            {
                "schema_version": 2,
                "algorithm": "G5J-Capability-Router-v2",
                "tool_id": args.tool_id,
                "instruction_id": instruction_id,
                "instruction": direct_rows[0]["instruction"],
                "direct": beta_summary(
                    sum(bool(value["success"]) for value in direct_rows),
                    len(direct_rows),
                ),
                "g5j": beta_summary(
                    sum(bool(value["success"]) for value in g5j_rows),
                    len(g5j_rows),
                ),
                "g5j_probe_ids": sorted(g5j_rows_by_id),
                "cluster_reviews": [
                    {
                        key: value.get(key)
                        for key in (
                            "cluster_id",
                            "route_recommendation",
                            "applicable_when",
                            "unsupported_when",
                            "remaining_uncertainty",
                            "confidence",
                            "conclusion",
                        )
                    }
                    for value in review_values.values()
                ],
                "review_recommendations": recommendations,
                "fallback_route": (
                    "g5j"
                    if g5j_rows
                    and sum(bool(value["success"]) for value in g5j_rows)
                    / len(g5j_rows)
                    > sum(bool(value["success"]) for value in direct_rows)
                    / len(direct_rows)
                    else "direct"
                ),
            }
        )
    atomic_jsonl(args.output, output)
    print(f"G5J_V2_RECORDS {len(output)}", flush=True)


def freeze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name, source in (
        ("task_programs.jsonl", args.programs),
        ("capability_records.jsonl", args.records),
        ("probe_reviews.jsonl", args.reviews),
        ("G5J_CAPABILITY_ROUTER_V2_SPEC.md", args.algorithm_spec),
    ):
        target = args.output_dir / name
        shutil.copy2(source, target)
        copied[name] = {"path": name, "sha256": sha256(target)}
    ledger_target = args.output_dir / "gpt_ledger_at_freeze.json"
    shutil.copy2(args.ledger, ledger_target)
    manifest = {
        "schema_version": 1,
        "algorithm": "G5J-Capability-Router-v2",
        "tool_id": args.tool_id,
        "frozen": True,
        "frozen_at": time.time(),
        "dev_split": "init 0--4",
        "heldout_split": "init 5--49",
        "routes": ["direct", "g5j"],
        "route_policy": (
            "one external GPT capability route per episode; G5j internal route "
            "is bypassed"
        ),
        "g5j_executor": {
            "mode": "g5j_forced_v2",
            "internal_route": False,
            "full_grounding_verify_regrasp_place_loop": True,
            "dense_caption": args.tool_id == "dense-v3-hybrid",
        },
        "api_failure_fallback": "per-instruction empirical fallback_route",
        "router_inputs": [
            "instruction",
            "cached task program",
            "initial fixed and wrist RGB",
            "proprioception",
            "frozen direct/G5j capability record",
        ],
        "forbidden_router_inputs": [
            "suite label",
            "BDDL",
            "success predicate",
            "simulator object coordinates",
        ],
        "files": copied,
        "gpt_ledger_at_freeze": {
            "path": ledger_target.name,
            "sha256": sha256(ledger_target),
        },
        "tool_identity": {
            "policy_config": args.policy_config,
            "checkpoint": str(args.checkpoint),
            "checkpoint_exists": args.checkpoint.exists(),
        },
    }
    atomic_json(args.output_dir / "router_manifest.json", manifest)
    print(
        f"G5J_V2_FROZEN {args.output_dir / 'router_manifest.json'}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("plan")
    command.add_argument("--clusters", type=Path, required=True)
    command.add_argument("--hypotheses", type=Path, required=True)
    command.add_argument("--programs", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--existing-specs", type=Path)
    command.add_argument("--reviews", type=Path)
    command.add_argument("--tool-id", required=True)
    command.add_argument("--round-index", type=int, default=0)
    command.add_argument("--max-clusters", type=int)
    command.add_argument("--max-repeats", type=int, default=4)
    command.set_defaults(function=plan)

    command = commands.add_parser("review")
    command.add_argument("--clusters", type=Path, required=True)
    command.add_argument("--hypotheses", type=Path, required=True)
    command.add_argument("--programs", type=Path, required=True)
    command.add_argument("--probes", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--keyframe-dir", type=Path, required=True)
    command.add_argument("--ledger", type=Path, required=True)
    command.add_argument("--cache-dir", type=Path, required=True)
    command.add_argument("--tool-id", required=True)
    command.add_argument("--max-tokens", type=int, default=1600)
    command.set_defaults(function=review)

    command = commands.add_parser("records")
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--clusters", type=Path, required=True)
    command.add_argument("--probes", type=Path, required=True)
    command.add_argument("--reviews", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--tool-id", required=True)
    command.set_defaults(function=make_records)

    command = commands.add_parser("freeze")
    command.add_argument("--programs", type=Path, required=True)
    command.add_argument("--records", type=Path, required=True)
    command.add_argument("--reviews", type=Path, required=True)
    command.add_argument("--ledger", type=Path, required=True)
    command.add_argument("--algorithm-spec", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--tool-id", required=True)
    command.add_argument("--policy-config", required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.set_defaults(function=freeze)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
