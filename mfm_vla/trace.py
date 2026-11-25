"""Step-by-step record of what the policy saw and commanded."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Trace:
    events: List[str] = field(default_factory=list)

    def add(self, line: str) -> None:
        self.events.append(line)

    def see(self, t: int, scene, grounding=None, extra: str = "") -> None:
        self.add(f"t={t:4d}  SEE  {scene.describe().replace(chr(10), ' | ')}")
        if grounding is not None:
            pick = None if grounding.pick is None else grounding.pick.describe()
            place = None if grounding.place is None else grounding.place.describe()
            self.add(f"t={t:4d}  SEE  grounding: {grounding.reason}")
            self.add(f"t={t:4d}  SEE  pick  {pick}")
            self.add(f"t={t:4d}  SEE  place {place}")
        if extra:
            self.add(f"t={t:4d}  SEE  {extra}")

    def act(self, t: int, eef, action, target=None, note: str = "") -> None:
        e = " ".join(f"{x:+.3f}" for x in eef[:3])
        a = " ".join(f"{x:+.3f}" for x in action[:3])
        g = "CLOSE" if action[6] > 0 else "OPEN"
        line = f"t={t:4d}  ACT  eef=({e})  dxyz=({a})  grip={g}"
        if target is not None:
            tg = " ".join(f"{x:+.3f}" for x in target[:3])
            line += f"  target=({tg})"
        if note:
            line += f"  {note}"
        self.add(line)

    def phase(self, t: int, name: str, detail: str = "") -> None:
        extra = f"  {detail}" if detail else ""
        self.add(f"t={t:4d}  ---- {name}{extra} ----")

    def dump(self) -> str:
        return "\n".join(self.events) + "\n"


def attach(env, trace: Optional[Trace]) -> Optional[Trace]:
    env.trace = trace
    return trace
