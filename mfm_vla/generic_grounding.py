"""Coarse-to-fine grounding of the object to grasp, from one calibrated RGB view.

Pass 1 sees the whole frame and proposes candidate boxes. Pass 2 sees a zoomed
crop around those candidates (numbered boxes drawn in) and commits to one
object and the pixel where its base meets the support surface. The crop is
derived from pass 1, so nothing is tuned to a particular camera or scene.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generic_config import GroundingConfig

COARSE_PROMPT = """You are the perception module of a robot arm.
The image is {width}x{height} pixels, origin at the top-left, u to the right, v downward.

Instruction: "{instruction}"

{question}
Objects may have been moved away from where they usually are, so identify the object by
its appearance (shape, colour, printed text), not by its location.
List up to {k} candidates that could be that object, most likely first.
Reply with JSON only:
{{"candidates": [{{"name": "<short name>", "visual_evidence": "<what identifies it>",
  "bbox": [u_min, v_min, u_max, v_max], "confidence": <0-1>}}]}}"""

FINE_PROMPT = """You are the perception module of a robot arm.
This is a zoomed crop, {width}x{height} pixels, origin at the top-left, u to the right, v downward.
Numbered yellow boxes mark candidates proposed from the full view: {candidates}.

Instruction: "{instruction}"

Decide which object is {target}; it may also be an unboxed object in the crop.
Identify it by its appearance, not by its location.
Return the pixel where that object's base centre touches the surface it stands on
(the centre of its footprint, not its top or label), and its bounding box, both in this crop.
Reply with JSON only:
{{"chosen": <candidate number, or 0 if unboxed>, "name": "<short name>",
  "visual_evidence": "<what identifies it>",
  "pixel_base_center": [u, v], "bbox": [u_min, v_min, u_max, v_max], "confidence": <0-1>}}"""


ROLES = {
    "grasp": {
        "question": "Which object must the robot grasp first to carry out this instruction?",
        "target": "the object the robot must grasp first",
    },
    "place": {
        "question": "Into or onto which object must the robot put the object it picks up, "
                    "to carry out this instruction?",
        "target": "the object into or onto which the picked-up object must be put",
    },
}


def ray_plane(obs, cam: str, uv, plane_z: float) -> np.ndarray:
    K = obs.intrinsics[cam]
    direction = obs.cam_to_world[cam][:3, :3] @ np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    origin = obs.cam_to_world[cam][:3, 3]
    if abs(direction[2]) < 1e-6:
        raise ValueError("pixel ray is parallel to the support plane")
    scale = (plane_z - origin[2]) / direction[2]
    if scale <= 0:
        raise ValueError("pixel ray does not hit the support plane")
    return origin + scale * direction


def _resize(image: np.ndarray, longest: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = longest / max(h, w)
    out = Image.fromarray(image).resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return np.asarray(out), scale


def crop_window(boxes: list[list[float]], size: int, cfg: GroundingConfig) -> tuple[int, int, int, int]:
    """Square window around the union of boxes, enlarged by the margin, clipped to the frame."""
    boxes = np.asarray(boxes, dtype=np.float64)
    u0, v0 = boxes[:, 0].min(), boxes[:, 1].min()
    u1, v1 = boxes[:, 2].max(), boxes[:, 3].max()
    side = max(u1 - u0, v1 - v0) * (1.0 + 2.0 * cfg.crop_margin)
    side = int(min(size, max(side, cfg.min_crop_frac * size)))
    cu, cv = (u0 + u1) / 2.0, (v0 + v1) / 2.0
    left = int(np.clip(cu - side / 2.0, 0, size - side))
    top = int(np.clip(cv - side / 2.0, 0, size - side))
    return left, top, left + side, top + side


def _draw(image: np.ndarray, boxes, labels, point=None) -> Image.Image:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for box, label in zip(boxes, labels):
        draw.rectangle([float(v) for v in box], outline=(255, 210, 0), width=2)
        draw.text((float(box[0]) + 3, float(box[1]) + 2), str(label), fill=(255, 210, 0))
    if point is not None:
        u, v = point
        draw.ellipse([u - 7, v - 7, u + 7, v + 7], outline=(0, 255, 0), width=3)
    return canvas


def ground_grasp_target(adapter, session, cfg: GroundingConfig, out_dir: Path, cam: str = "agentview",
                        role: str = "grasp") -> dict:
    """Ground the object playing ``role`` (first to grasp, or where to put it)."""
    words = ROLES[role]
    prefix = "" if role == "grasp" else f"{role}_"
    obs = adapter.observe([cam], cfg.render_px)
    frame = obs.rgb[cam]
    size = frame.shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = []

    coarse, coarse_scale = _resize(frame, cfg.coarse_px)
    asked = time.monotonic()
    reply1 = session.decide_json(
        COARSE_PROMPT.format(
            width=coarse.shape[1], height=coarse.shape[0],
            instruction=adapter.instruction, k=cfg.max_candidates, question=words["question"],
        ),
        {cam: coarse},
    )
    calls.append(round(time.monotonic() - asked, 2))
    candidates = [c for c in reply1.get("candidates", []) if len(c.get("bbox") or []) == 4]
    candidates = candidates[: cfg.max_candidates]
    if not candidates:
        raise ValueError("coarse grounding returned no candidate boxes")
    boxes = [np.asarray(c["bbox"], dtype=np.float64) / coarse_scale for c in candidates]
    _draw(coarse, [b * coarse_scale for b in boxes], range(1, len(boxes) + 1)).save(out_dir / f"{prefix}ground_coarse.png")

    left, top, right, bottom = crop_window(boxes, size, cfg)
    crop, fine_scale = _resize(frame[top:bottom, left:right], cfg.fine_px)
    crop_boxes = [(b - np.array([left, top, left, top])) * fine_scale for b in boxes]
    listed = ", ".join(f"{i}: {c.get('name', '?')}" for i, c in enumerate(candidates, 1))
    asked = time.monotonic()
    reply2 = session.decide_json(
        FINE_PROMPT.format(
            width=crop.shape[1], height=crop.shape[0],
            candidates=listed, instruction=adapter.instruction, target=words["target"],
        ),
        {cam: np.asarray(_draw(crop, crop_boxes, range(1, len(boxes) + 1)))},
    )
    calls.append(round(time.monotonic() - asked, 2))
    local = np.asarray(reply2["pixel_base_center"], dtype=np.float64)

    def crop_to_frame(crop_uv: np.ndarray) -> np.ndarray:
        return np.asarray(crop_uv, dtype=np.float64) / fine_scale + np.array([left, top], dtype=np.float64)

    uv = crop_to_frame(local)
    _draw(crop, crop_boxes, range(1, len(boxes) + 1), point=local).save(out_dir / f"{prefix}ground_fine.png")

    calibration = adapter.calibration
    return {
        "world": None if calibration is None else ray_plane(obs, cam, uv, calibration.support_eef_z),
        "pixel": uv.round(1).tolist(),
        "crop": [left, top, right, bottom],
        "coarse": reply1,
        "fine": reply2,
        "latency_s": calls,
        "_obs": obs,
        "_crop_image": crop,
        "_crop_to_frame": crop_to_frame,
    }
