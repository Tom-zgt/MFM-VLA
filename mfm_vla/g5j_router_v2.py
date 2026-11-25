"""Frozen GPT router between direct π0.5 and forced original G5j."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROUTE_PROMPT = """You are selecting one of two robot tools after controlled
capability learning.

Tool direct: π0.5 executes the original instruction from the initial state.
Tool g5j: the original G5j assist executor is forced on. Its internal route
decision is removed, but its GPT two-view grounding, policy-aligned pre-grasp
handoff, grasp verification/regrasp, drop recovery, carry progress watch,
Watch-and-Takeover PlaceLoop, pre-place handoff, and release fallback remain.

Instruction:
{instruction}
Frozen task pattern:
{program}
Frozen experimental capability record for this exact VLA:
{capability}
Initial proprioception:
{proprioception}

Use the attached initial fixed and wrist RGB images to decide whether this
episode resembles conditions where direct π0.5 is reliable or where forced G5j
repairs a known spatial/semantic routing failure. Actual probe outcomes in the
capability record override speculation. Do not infer a benchmark suite, BDDL,
hidden predicate, or simulator coordinates. Do not choose G5j merely because it
exists; its extra observations and handoffs can hurt an already reliable direct
policy.

Return JSON only:
{{"route":"direct|g5j","confidence":0.0,
"matched_capability_conditions":["..."],"reason":"..."}}"""


def instruction_id(instruction: str) -> str:
    return hashlib.sha256(instruction.strip().encode()).hexdigest()


def load_jsonl(path: Path) -> dict[str, dict]:
    return {
        value["instruction_id"]: value
        for value in (
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        )
    }


class FrozenG5JCapabilityRouter:
    def __init__(self, manifest_path: Path, brain=None):
        self.manifest_path = manifest_path
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text())
        if (
            self.manifest.get("algorithm") != "G5J-Capability-Router-v2"
            or not self.manifest.get("frozen")
        ):
            raise ValueError("router manifest is not a frozen G5J v2 manifest")
        files = self.manifest["files"]
        self.programs = load_jsonl(
            self.root / files["task_programs.jsonl"]["path"]
        )
        self.records = load_jsonl(
            self.root / files["capability_records.jsonl"]["path"]
        )
        self.brain = brain

    def cached(self, instruction: str) -> tuple[str, dict, dict]:
        key = instruction_id(instruction)
        if key not in self.programs or key not in self.records:
            raise KeyError(f"no frozen capability record for instruction {key}")
        return key, self.programs[key], self.records[key]

    def route(
        self,
        instruction: str,
        images: dict[str, np.ndarray],
        proprioception: dict,
    ) -> dict:
        key, program, capability = self.cached(instruction)
        if self.brain is None:
            return {
                "route": capability.get("fallback_route", "direct"),
                "confidence": 0.0,
                "matched_capability_conditions": [],
                "reason": "GPT unavailable; frozen empirical fallback",
                "fallback": True,
                "instruction_id": key,
            }
        decision = self.brain.decide_json(
            ROUTE_PROMPT.format(
                instruction=instruction,
                program=json.dumps(
                    program, ensure_ascii=False, separators=(",", ":")
                ),
                capability=json.dumps(
                    capability, ensure_ascii=False, separators=(",", ":")
                ),
                proprioception=json.dumps(
                    proprioception, separators=(",", ":"), default=float
                ),
            ),
            images,
            bucket="heldout_routing",
            reason=f"g5j_v2_route:{self.manifest['tool_id']}:{key}",
        )
        if decision.get("route") not in {"direct", "g5j"}:
            raise ValueError("external capability router returned an invalid route")
        return {
            **decision,
            "instruction_id": key,
            "fallback": False,
        }
