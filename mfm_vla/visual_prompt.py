"""Visual prompts for π0.5: marks drawn on the third-person policy image.

A green ring marks the object to pick up or act on; a magenta ring marks where it
goes. Coordinates are in the policy image (the 256x256 agentview frame exactly as
sent to the policy server, i.e. after the 180-degree rotation openpi's LIBERO
checkpoints expect). Training data and inference draw through this one function.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

PICK_RGB = (0, 255, 0)
PLACE_RGB = (255, 0, 255)


def policy_uv(obs_uv, size: int) -> np.ndarray:
    """Pixel in ``Observation`` image order (top-down) to policy-image pixel; the
    policy image is that frame mirrored left-right."""
    u, v = float(obs_uv[0]), float(obs_uv[1])
    return np.array([size - 1 - u, v], dtype=np.float32)


def goal_targets(goal) -> tuple[str | None, str | None]:
    """(pick/act target, place target) named by one BDDL goal predicate."""
    if str(goal[0]).lower() in ("in", "on"):
        return goal[1], goal[2]
    return goal[1], None


def project_target(inner, name: str | None, obs, cam: str, size: int) -> np.ndarray:
    """Policy-image pixel of a simulator object, fixture or site (NaN if unnamed)."""
    if name is None or name not in inner.object_states_dict:
        return np.full(2, np.nan, dtype=np.float32)
    pos = np.asarray(inner.object_states_dict[name].get_geom_state()["pos"], dtype=np.float64)
    return policy_uv(obs.project(cam, pos), size)


def draw_marks(image: np.ndarray, pick_uv=None, place_uv=None, radius: float = 9.0,
               width: int = 2) -> np.ndarray:
    """Copy of ``image`` with the marks; a mark that is None or NaN is omitted."""
    canvas = Image.fromarray(np.ascontiguousarray(image))
    draw = ImageDraw.Draw(canvas)
    for uv, rgb in ((pick_uv, PICK_RGB), (place_uv, PLACE_RGB)):
        if uv is None or not np.all(np.isfinite(uv)):
            continue
        u, v = float(uv[0]), float(uv[1])
        draw.ellipse([u - radius, v - radius, u + radius, v + radius], outline=rgb, width=width)
    return np.asarray(canvas)
