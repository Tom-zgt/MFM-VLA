"""Seed g5d.jsonl with g5c rows whose run G5d would have repeated exactly.

G5d differs from G5c only when π0.5's carry window check fires within the stall radius,
so rows where it never did are kept and the rest are rerun."""
import json
import sys
from pathlib import Path

import numpy as np

RADIUS = 0.10
out = Path("reports/generic/g5d.jsonl")
done = {(r["suite"], r["task"], r["init"]) for r in map(json.loads, out.open())} if out.exists() else set()
kept = rerun = 0
lines = []
for line in open("reports/generic/g5c.jsonl"):
    r = json.loads(line)
    if r.get("crash") or (r["suite"], r["task"], r["init"]) in done:
        continue
    affected = False
    if not r.get("carried_over_from"):
        steps = None
        for e in r.get("events", []):
            if e.get("why") == "vla_carry" and e.get("outcome") == "stalled":
                steps = steps or [json.loads(x) for x in open(Path(r["artifacts"]) / "steps.jsonl")]
                eef = np.array(steps[e["t_end"] - 1]["eef_position"][:2])
                affected |= float(np.linalg.norm(eef - np.array(e["goal"][:2]))) <= RADIUS
    if affected:
        rerun += 1
        continue
    r["mode"], r["carried_over_from"] = "g5d", r.get("carried_over_from") or "g5c"
    lines.append(json.dumps(r, separators=(",", ":")) + "\n")
    kept += 1
with out.open("a") as handle:
    handle.writelines(lines)
print(f"kept {kept}, rerun {rerun}", file=sys.stderr)
