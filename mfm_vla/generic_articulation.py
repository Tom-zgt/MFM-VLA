"""Drawer handles: measurement in two calibrated views, and the opening motion.

GPT marks the handle (centre and the two ends of its bar) in the fixed view, then
the same handle in the wrist view; the calibrated rays give its 3D centre and the
direction of the bar. The pull direction is horizontal, across the bar, towards the
fixed camera: a handle the camera can see is on the side of the fixture facing it.

The motion is the VLA's own, profiled on its training tasks: turn the hand, come in
above the handle, press down behind it until blocked, pull along the pull direction
until the drawer stops.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from generic_grounding import _resize, crop_window
from generic_multiview import pixel_ray, triangulate

COARSE_PROMPT = """You are the perception module of a robot arm.
The image is {width}x{height} pixels, origin at the top-left, u to the right, v downward.

Instruction: "{instruction}"

The robot must operate a fixture's moving part (for example open or close a drawer).
Which handle must the robot use to do this? If the instruction names one drawer among
several (top, middle, bottom...), pick that drawer's handle. Objects may have been moved
away from where they usually are, so identify it by its appearance.
Reply with JSON only:
{{"fixture": "<short name>", "part": "<which drawer or door>", "motion": "pull_open" or "push_close" or "other",
  "handle_bbox": [u_min, v_min, u_max, v_max], "fixture_bbox": [u_min, v_min, u_max, v_max],
  "confidence": <0-1>}}"""

FINE_PROMPT = """You are the perception module of a robot arm.
This is a zoomed crop, {width}x{height} pixels, origin at the top-left, u to the right, v downward.
The yellow box is a rough guess for the handle of {part} of the {fixture}.

Instruction: "{instruction}"

Find that handle exactly (it may be slightly outside the box). Give the pixel at the centre
of the handle bar and the pixels at the two ends of the bar.
Reply with JSON only:
{{"center": [u, v], "end_a": [u, v], "end_b": [u, v], "bbox": [u_min, v_min, u_max, v_max],
  "evidence": "<what identifies it>", "confidence": <0-1>}}"""

WRIST_PROMPT = """You are the perception module of a robot arm.
Image "A_fixed" is a zoomed crop of a fixed camera; the green dot marks the centre of a handle
bar and the two red dots its ends ({part} of the {fixture}).
Image "B_wrist" ({wb}x{hb} pixels) is from a camera on the robot's wrist, looking from a
different angle; it may appear rotated. Pixel coordinates: origin at the top-left, u to the
right, v downward.

Find the same handle in B_wrist. Give the pixel at the centre of its bar and at its two ends.
Reply with JSON only:
{{"visible_in_b": true or false, "center": [u, v], "end_a": [u, v], "end_b": [u, v],
  "b_bbox": [u_min, v_min, u_max, v_max], "evidence": "<what identifies it>", "confidence": <0-1>}}"""

OPEN_CHECK_PROMPT = """You are the verification module of a robot arm.
Instruction: "{instruction}"
The first image is a fixed camera now; the second is the camera on the robot's wrist.
Has the robot done what the instruction asks: is {part} of the {fixture} pulled out far,
clearly open? A drawer moved only a little is not open.
Reply with JSON only:
{{"done": true or false, "evidence": "<one short sentence>"}}"""


def _uv(value) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(v is None for v in value):
        return None
    out = np.asarray(value, dtype=np.float64)
    return out if np.isfinite(out).all() else None


def ground_handle(adapter, session, gcfg, render_px: int, out_dir, cam: str = "agentview") -> dict:
    """Fixed view only, no motion: the handle's centre and bar ends in frame pixels.
    ``gcfg``: the GroundingConfig (coarse and fine image sizes, crop window)."""
    obs = adapter.observe([cam], render_px)
    frame = obs.rgb[cam]
    size = frame.shape[0]
    coarse, scale = _resize(frame, gcfg.coarse_px)
    asked = time.monotonic()
    first = session.decide_json(
        COARSE_PROMPT.format(width=coarse.shape[1], height=coarse.shape[0], instruction=adapter.instruction),
        {cam: coarse})
    latency = [round(time.monotonic() - asked, 2)]
    box = first.get("handle_bbox")
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
        return {"coarse": first, "accepted": False, "latency_s": latency}
    handle_box = np.asarray(box, dtype=np.float64) / scale
    left, top, right, bottom = crop_window([handle_box], size, gcfg)
    crop, fine_scale = _resize(frame[top:bottom, left:right], gcfg.fine_px)
    guess = (handle_box - np.array([left, top, left, top])) * fine_scale
    canvas = Image.fromarray(crop)
    ImageDraw.Draw(canvas).rectangle([float(v) for v in guess], outline=(255, 210, 0), width=2)
    asked = time.monotonic()
    fine = session.decide_json(
        FINE_PROMPT.format(width=crop.shape[1], height=crop.shape[0], part=first.get("part", "the drawer"),
                           fixture=first.get("fixture", "fixture"), instruction=adapter.instruction),
        {cam: np.asarray(canvas)})
    latency.append(round(time.monotonic() - asked, 2))
    points = {k: _uv(fine.get(k)) for k in ("center", "end_a", "end_b")}
    out = {"coarse": first, "fine": fine, "crop": [left, top, right, bottom], "latency_s": latency,
           "accepted": all(p is not None for p in points.values())}
    if not out["accepted"]:
        return out
    to_frame = lambda uv: uv / fine_scale + np.array([left, top], dtype=np.float64)  # noqa: E731
    marked = Image.fromarray(crop)
    draw = ImageDraw.Draw(marked)
    for key, colour in (("center", (0, 255, 0)), ("end_a", (255, 0, 0)), ("end_b", (255, 0, 0))):
        u, v = points[key]
        draw.ellipse([u - 6, v - 6, u + 6, v + 6], outline=colour, width=3)
    marked.save(out_dir / "handle_fixed.png")
    out.update(
        pixels={k: to_frame(p) for k, p in points.items()},
        _obs=obs, _marked=np.asarray(marked),
        _rays=[pixel_ray(obs, cam, to_frame(np.array(c))) for c in
               ((guess[0], guess[1]), (guess[2], guess[1]), (guess[0], guess[3]), (guess[2], guess[3]))],
        fixture=first.get("fixture", "fixture"), part=first.get("part", "the drawer"),
        motion=first.get("motion", "other"),
    )
    return out


def locate_in_wrist(adapter, session, handle: dict, render_px: int, out_dir, tag: str,
                    wrist: str = "robot0_eye_in_hand") -> dict:
    """Mark the same handle in the wrist view; triangulate with the fixed view."""
    obs = adapter.observe([wrist], render_px)
    image = obs.rgb[wrist]
    asked = time.monotonic()
    reply = session.decide_json(
        WRIST_PROMPT.format(part=handle["part"], fixture=handle["fixture"], wb=image.shape[1], hb=image.shape[0]),
        {"A_fixed": handle["_marked"], "B_wrist": image})
    out = {"reply": reply, "latency_s": round(time.monotonic() - asked, 2), "accepted": False}
    wuv = {k: _uv(reply.get(k)) for k in ("center", "end_a", "end_b")}
    if not reply.get("visible_in_b") or any(v is None for v in wuv.values()):
        return out
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for key in wuv:
        u, v = wuv[key]
        draw.ellipse([u - 7, v - 7, u + 7, v + 7], outline=(0, 255, 0) if key == "center" else (255, 0, 0), width=3)
    canvas.save(out_dir / f"handle_wrist_{tag}.png")
    out.update(handle_geometry(handle["_obs"], handle["pixels"], adapter.static_cameras[0], obs, wuv, wrist))
    return out


def handle_geometry(fixed_obs, fixed_px: dict, fixed: str, wrist_obs, wrist_px: dict, wrist: str) -> dict:
    """Centre, ray gap, and pull direction from the handle's centre and bar ends in two views.
    The ends may be named in either order in the second view; the pairing whose rays meet
    more closely wins."""
    centre, gap = triangulate(pixel_ray(fixed_obs, fixed, fixed_px["center"]),
                              pixel_ray(wrist_obs, wrist, wrist_px["center"]))
    pairings = []
    for a, b in (("end_a", "end_b"), ("end_b", "end_a")):
        pa, ga = triangulate(pixel_ray(fixed_obs, fixed, fixed_px["end_a"]), pixel_ray(wrist_obs, wrist, wrist_px[a]))
        pb, gb = triangulate(pixel_ray(fixed_obs, fixed, fixed_px["end_b"]), pixel_ray(wrist_obs, wrist, wrist_px[b]))
        pairings.append((ga + gb, pa, pb))
    _, end_a, end_b = min(pairings, key=lambda p: p[0])
    bar = (end_b - end_a)[:2]
    out = {"point": centre, "ray_gap_m": round(gap, 4), "bar_length_m": round(float(np.linalg.norm(bar)), 4),
           "accepted": False}
    if np.linalg.norm(bar) < 0.02:
        return out
    normal = np.array([-bar[1], bar[0]]) / np.linalg.norm(bar)
    camera = fixed_obs.cam_to_world[fixed][:2, 3]
    if normal @ (camera - centre[:2]) < 0:
        normal = -normal
    out.update(outward_xy=normal, accepted=True)
    return out


def drawer_poses(profile: dict, point: np.ndarray, outward: np.ndarray, front_clearance: float) -> dict:
    """Hand poses in the frame whose +y is the pull direction, from the VLA's profile."""
    yaw = Rotation.from_euler("z", np.arctan2(outward[1], outward[0]) - np.pi / 2)
    out = np.array([outward[0], outward[1], 0.0])
    above = point + yaw.apply(np.asarray(profile["hook_above_m"]))
    return {
        "quat": (yaw * Rotation.from_quat(profile["pre_pull_quat_xyzw"])).as_quat(),
        "front": above + out * front_clearance,
        "above": above,
        "pressed": point + yaw.apply(np.asarray(profile["hook_press_m"])),
        "out": out,
    }


def check_open(adapter, session, handle: dict, size: int, fixed: str = "agentview",
               wrist: str = "robot0_eye_in_hand") -> dict:
    obs = adapter.observe([fixed, wrist], size)
    asked = time.monotonic()
    reply = session.decide_json(
        OPEN_CHECK_PROMPT.format(instruction=adapter.instruction, part=handle["part"], fixture=handle["fixture"]),
        {fixed: obs.rgb[fixed], wrist: obs.rgb[wrist]})
    return {"t": adapter.t, "done": bool(reply.get("done")), "evidence": reply.get("evidence"),
            "latency_s": round(time.monotonic() - asked, 2)}
