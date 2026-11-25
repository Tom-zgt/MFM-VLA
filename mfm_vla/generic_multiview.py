"""Second-view measurement: triangulate an object's top and estimate its support height.

GPT marks the centre of the object's top surface in the fixed view and in the
wrist view; the two calibrated rays give the top point. The fixed-view ray
through the base contact pixel, met with the vertical line under that top
point, gives the height of the surface the object stands on. No depth, no
pre-measured table height.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageDraw

SECOND_VIEW_PROMPT = """You are the perception module of a robot arm.
Image "A_fixed" is a zoomed crop ({wa}x{ha} pixels) of a fixed camera; the yellow box marks the
object "{name}" ({evidence}).
Image "B_wrist" ({wb}x{hb} pixels) is from a camera on the robot's wrist, looking from a different
angle; it may appear rotated. Pixel coordinates: origin at the top-left, u to the right, v downward.

Find the same physical object in B_wrist. In both images, give the pixel at the centre of the
object's top surface as it stands (for example the centre of a lid, cap, or top face{top_hint}).
Reply with JSON only:
{{"visible_in_b": true or false, "b_evidence": "<what identifies it in B>",
  "a_top_center": [u, v], "b_top_center": [u, v], "b_bbox": [u_min, v_min, u_max, v_max],
  "confidence": <0-1>}}"""


TOP_HINTS = {
    "grasp": "",
    "place": "; for an open container, the centre of its opening at the height of its rim",
}


def pixel_ray(obs, cam: str, uv) -> tuple[np.ndarray, np.ndarray]:
    K = obs.intrinsics[cam]
    direction = obs.cam_to_world[cam][:3, :3] @ np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    return obs.cam_to_world[cam][:3, 3].copy(), direction / np.linalg.norm(direction)


def triangulate(ray_a, ray_b) -> tuple[np.ndarray, float]:
    """Midpoint of the shortest segment between two rays, and that segment's length."""
    (oa, da), (ob, db) = ray_a, ray_b
    w = oa - ob
    a, b, c = da @ da, da @ db, db @ db
    d, e = da @ w, db @ w
    denom = a * c - b * b
    if denom < 1e-9:
        raise ValueError("rays are parallel")
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    pa, pb = oa + s * da, ob + t * db
    return (pa + pb) / 2.0, float(np.linalg.norm(pa - pb))


def nearest_on_first(ray_a, ray_b) -> np.ndarray:
    """Point of ``ray_a`` closest to ``ray_b``."""
    (oa, da), (ob, db) = ray_a, ray_b
    w = oa - ob
    a, b, c = da @ da, da @ db, db @ db
    d, e = da @ w, db @ w
    return oa + (b * e - c * d) / (a * c - b * b) * da


def height_under(ray, xy) -> float:
    """z of the ray point horizontally closest to the vertical line through ``xy``."""
    origin, direction = ray
    horizontal = direction[:2]
    s = float(horizontal @ (np.asarray(xy) - origin[:2]) / (horizontal @ horizontal))
    return float(origin[2] + s * direction[2])


def _point(value) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(v is None for v in value):
        return None
    return np.asarray(value, dtype=np.float64)


def _box(value) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(v is None for v in value):
        return None
    return np.asarray(value, dtype=np.float64)


def view_status(reply: dict, size: int, edge_px: float) -> str:
    """full: whole object and its top point inside the frame; partial: clipped; none: not seen."""
    if not reply.get("visible_in_b"):
        return "none"
    top, box = _point(reply.get("b_top_center")), _box(reply.get("b_bbox"))
    if top is None or box is None:
        return "partial"
    inside = (top >= 0).all() and (top < size).all()
    clear = box[0] > edge_px and box[1] > edge_px and box[2] < size - edge_px and box[3] < size - edge_px
    return "full" if inside and clear else "partial"


def survey_goal(adapter, wrist_obs, wrist: str, rays: list, depth_range: tuple[float, float],
                size: int, margin: float = 0.12, reach_xy_m: float = 0.25, rise_m: float = 0.2,
                grid_m: float = 0.025, keep: float = 0.95) -> np.ndarray | None:
    """End-effector goal, same orientation, from which the wrist camera sees as much
    as possible of every fixed-view ray over the heights ``depth_range`` below the
    start hand height.

    Each target is known only up to its fixed-view ray, so the goal maximises the
    share of ray samples inside the wrist image (with ``margin``); among goals within
    ``keep`` of the best share, the one closest to the hand wins, since a raised or
    far-out hand nears the arm's reach limits. Never below the hand. None if no grid
    goal sees any sample.
    """
    cam = wrist_obs.cam_to_world[wrist]
    K = wrist_obs.intrinsics[wrist]
    eef = adapter.proprio().eef_pos
    zs = eef[2] - np.linspace(depth_range[0], depth_range[1], 6)
    points = []
    for origin, direction in rays:
        if abs(direction[2]) < 1e-6:
            continue
        for z in zs:
            points.append(origin + (z - origin[2]) / direction[2] * direction)
    if not points:
        return None
    points = np.asarray(points)
    lo, hi = margin * size, (1.0 - margin) * size
    rotation = cam[:3, :3]
    steps = np.arange(-reach_xy_m, reach_xy_m + 1e-9, grid_m)
    scored = []
    for dz in np.arange(0.0, rise_m + 1e-9, 0.05):
        for dx in steps:
            for dy in steps:
                shift = np.array([dx, dy, dz])
                local = (points - (cam[:3, 3] + shift)) @ rotation
                ahead = local[:, 2] > 0.02
                uv = (local @ K.T)[:, :2] / np.where(ahead, local[:, 2], 1.0)[:, None]
                seen = ahead & (uv >= lo).all(1) & (uv <= hi).all(1)
                scored.append((float(seen.mean()), float(np.linalg.norm(shift)), shift))
    top = max(s[0] for s in scored)
    if top <= 0:
        return None
    _, _, shift = min((s for s in scored if s[0] >= keep * top), key=lambda s: s[1])
    return eef + shift


def recentre_goal(adapter, wrist_obs, wrist: str, point: np.ndarray) -> np.ndarray:
    """End-effector goal at the same height that puts ``point`` on the wrist camera's optical axis."""
    cam = wrist_obs.cam_to_world[wrist]
    axis = cam[:3, 2] / np.linalg.norm(cam[:3, 2])
    eef = adapter.proprio().eef_pos
    if axis[2] > -0.2:
        # Wrist camera is tilted horizontally rather than straight down: shift horizontally toward point
        goal = eef.copy()
        goal[:2] = point[:2] - 0.12 * (axis[:2] / max(float(np.linalg.norm(axis[:2])), 1e-6))
        return goal
    s = (cam[2, 3] - point[2]) / -axis[2]
    camera_goal = point - axis * s
    goal = eef + (camera_goal - cam[:3, 3])
    goal[2] = eef[2]
    return goal


def measure_in_second_view(adapter, session, fine: dict, crop_image: np.ndarray, crop_to_frame,
                           frame_obs, render_px: int, out_dir, *, max_recentre: int = 2,
                           raise_m: float = 0.15, edge_px: float = 3.0, role: str = "grasp",
                           max_gap_m: float | None = None, relook_on: str = "midpoint",
                           use_best: bool = False,
                           wrist: str = "robot0_eye_in_hand", fixed: str = "agentview") -> dict:
    """``fine`` is the fine-pass reply in crop pixels; ``crop_to_frame`` maps crop uv to frame uv.

    If the object is clipped, the wrist camera is moved over a rough triangulation of
    its visible part; if it is not seen at all, the camera is raised once. If the two
    rays miss each other by more than ``max_gap_m``, the wrist camera is recentred on
    the triangulated midpoint (``relook_on="midpoint"``, G3a) or on the fixed-view ray,
    which stays on the target even when the wrist view matched another object
    (``relook_on="fixed_ray"``). Motion uses the contact-free ``move_to`` only.
    ``use_best``: the result comes from the full view whose rays met most closely,
    not from the last view.
    """
    marked = Image.fromarray(crop_image)
    ImageDraw.Draw(marked).rectangle([float(v) for v in fine["bbox"]], outline=(255, 210, 0), width=3)
    base_a = crop_to_frame(np.asarray(fine["pixel_base_center"], dtype=np.float64))
    base_ray = pixel_ray(frame_obs, fixed, base_a)
    views, moves, latencies = [], [], []
    raised = False
    best = None
    for attempt in range(max_recentre + 3):
        wrist_obs = adapter.observe([wrist], render_px)
        wrist_image = wrist_obs.rgb[wrist]
        asked = time.monotonic()
        reply = session.decide_json(
            SECOND_VIEW_PROMPT.format(
                wa=crop_image.shape[1], ha=crop_image.shape[0], name=fine.get("name", "object"),
                evidence=fine.get("visual_evidence", ""), wb=wrist_image.shape[1], hb=wrist_image.shape[0],
                top_hint=TOP_HINTS[role],
            ),
            {"A_fixed": np.asarray(marked), "B_wrist": wrist_image},
        )
        latencies.append(round(time.monotonic() - asked, 2))
        status = view_status(reply, wrist_image.shape[0], edge_px)
        view = {"status": status, "reply": reply}
        views.append(view)
        if status == "full" and max_gap_m is not None:
            top_a = crop_to_frame(np.asarray(reply["a_top_center"], dtype=np.float64))
            ray_a = pixel_ray(frame_obs, fixed, top_a)
            ray_b = pixel_ray(wrist_obs, wrist, _point(reply["b_top_center"]))
            top, gap = triangulate(ray_a, ray_b)
            view["ray_gap_m"] = round(gap, 4)
            if best is None or gap < best[0]:
                best = (gap, reply, wrist_obs, wrist_image)
            if gap > max_gap_m and len(moves) < max_recentre:
                aim = top if relook_on == "midpoint" else nearest_on_first(ray_a, ray_b)
                moves.append(adapter.move_to(recentre_goal(adapter, wrist_obs, wrist, aim),
                                             phase="look_inconsistent"))
                continue
        if status == "full" or attempt >= max_recentre + 1:
            break
        if status == "partial":
            box = _box(reply.get("b_bbox"))
            if box is None:
                break
            seen = np.clip((box[:2] + box[2:]) / 2.0, 0, wrist_image.shape[0] - 1)
            rough, _ = triangulate(base_ray, pixel_ray(wrist_obs, wrist, seen))
            goal = recentre_goal(adapter, wrist_obs, wrist, rough)
        elif not raised:
            goal = adapter.proprio().eef_pos + np.array([0.0, 0.0, raise_m])
            raised = True
        else:
            break
        moves.append(adapter.move_to(goal, phase=f"look_{status}"))

    result = {"views": views, "moves": moves, "latency_s": latencies,
              "status": views[-1]["status"], "visible": views[-1]["status"] == "full"}
    if use_best and best is not None:
        _, reply, wrist_obs, wrist_image = best
        result.update(status="full", visible=True, chosen_view=next(
            i for i, v in enumerate(views) if v["reply"] is reply))
    if not result["visible"]:
        return result
    top_a = crop_to_frame(np.asarray(reply["a_top_center"], dtype=np.float64))
    top_b = _point(reply["b_top_center"])
    top, gap = triangulate(pixel_ray(frame_obs, fixed, top_a), pixel_ray(wrist_obs, wrist, top_b))
    support_z = height_under(base_ray, top[:2])

    canvas = Image.fromarray(wrist_image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([float(v) for v in reply["b_bbox"]], outline=(255, 210, 0), width=2)
    u, v = top_b
    draw.ellipse([u - 7, v - 7, u + 7, v + 7], outline=(0, 255, 0), width=3)
    canvas.save(out_dir / ("ground_wrist.png" if role == "grasp" else f"{role}_ground_wrist.png"))
    result.update({
        "top": top, "ray_gap_m": round(gap, 4), "support_z": round(support_z, 4),
        "height_m": round(float(top[2] - support_z), 4),
    })
    return result
