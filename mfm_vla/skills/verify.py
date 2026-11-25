"""Visual residual after a primitive, in metres.

This is the practice instrument. After you move, look once, measure the gap,
and decide whether to nudge or to write a new lesson. Privileged poses are
optional labels for the log; the residual the policy may use comes only from
the RGB-D cloud.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional

import numpy as np

from perception import segment_scene

MEMORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")


@dataclass
class Residual:
    phase: str
    intended_xy: np.ndarray
    measured_xy: Optional[np.ndarray]
    residual_m: Optional[np.ndarray]
    opening_m: float
    note: str

    def ok(self, tol_m: float = 0.03) -> bool:
        if self.measured_xy is None or self.residual_m is None:
            return False
        return float(np.linalg.norm(self.residual_m[:2])) < tol_m


def nearest_blob_xy(obs, intended_xy, max_dist=0.12):
    scene = segment_scene(obs)
    if not scene.blobs:
        return None, scene
    b = min(scene.blobs, key=lambda x: np.linalg.norm(x.xy - intended_xy))
    if np.linalg.norm(b.xy - intended_xy) > max_dist:
        return None, scene
    return b, scene


def verify_target(obs, intended_xy, phase: str, opening_m: float = 0.0) -> Residual:
    """Where is the thing we aimed at, relative to where we meant?"""
    blob, _ = nearest_blob_xy(obs, intended_xy)
    if blob is None:
        return Residual(phase, intended_xy, None, None, opening_m,
                        "no blob near intended xy — shoved or occluded")
    res = np.append(blob.xy - intended_xy, blob.top_z - obs.eef_pos[2])
    return Residual(
        phase, intended_xy, blob.xy, res, opening_m,
        f"blob{blob.idx} residual {100*np.linalg.norm(res[:2]):.1f} cm",
    )


def append_lesson(skill: str, residual: Residual, success: bool, extra: dict = None):
    if os.environ.get("HOI_DISABLE_MEMORY_WRITE") == "1":
        return None
    os.makedirs(os.path.dirname(MEMORY), exist_ok=True)
    if os.path.exists(MEMORY):
        data = json.load(open(MEMORY))
    else:
        data = {"updated": str(date.today()), "lessons": []}
    entry = {
        "skill": skill,
        "phase": residual.phase,
        "residual_m": None if residual.residual_m is None else residual.residual_m.tolist(),
        "opening_m": residual.opening_m,
        "success": success,
        "note": residual.note,
        "date": str(date.today()),
    }
    if extra:
        entry.update(extra)
    data["lessons"].append(entry)
    data["updated"] = str(date.today())
    with open(MEMORY, "w") as f:
        json.dump(data, f, indent=2)
    return entry
