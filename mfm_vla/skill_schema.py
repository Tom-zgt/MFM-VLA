"""Validated skill-command schema for high-level robot task decisions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

ACTIONS = {
    "pick_place",
    "open",
    "close",
    "turn_on",
    "continue_skill",
    "abort_skill",
    "stop_success",
    "stop_failure",
}
RELATIONS = {"on", "in", ""}


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S)
    candidate = fenced.group(1) if fenced else stripped
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"brain returned no JSON object: {stripped[:240]!r}")
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("brain response must be a JSON object")
    return value


@dataclass(frozen=True)
class BrainDecision:
    action: str
    target: str = ""
    receptacle: str = ""
    relation: str = ""
    rationale: str = ""
    success_evidence: str = ""
    retry_strategy: str = ""
    confidence: float = 0.0

    @classmethod
    def from_text(cls, text: str) -> "BrainDecision":
        payload = _extract_json(text)
        action = str(payload.get("action", "")).strip().lower()
        relation = str(payload.get("relation", "")).strip().lower()
        if relation in {"empty", "none", "null", "n/a"}:
            relation = ""
        if action not in ACTIONS:
            raise ValueError(f"unsupported brain action: {action!r}")
        if relation not in RELATIONS:
            raise ValueError(f"unsupported relation: {relation!r}")
        if action == "pick_place":
            if not str(payload.get("target", "")).strip():
                raise ValueError("pick_place requires target")
            if not str(payload.get("receptacle", "")).strip():
                raise ValueError("pick_place requires relation on/in" if relation not in {"on", "in"} else "pick_place requires receptacle")
            if relation not in {"on", "in"}:
                raise ValueError("pick_place requires relation on/in")
        confidence = float(payload.get("confidence", 0.0))
        return cls(
            action=action,
            target=str(payload.get("target", "")).strip().lower(),
            receptacle=str(payload.get("receptacle", "")).strip().lower(),
            relation=relation,
            rationale=str(payload.get("rationale", "")).strip(),
            success_evidence=str(payload.get("success_evidence", "")).strip(),
            retry_strategy=str(payload.get("retry_strategy", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
        )
