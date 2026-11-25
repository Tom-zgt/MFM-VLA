"""RGB-only tabletop parsing. No depth, no point cloud.

A calibrated camera plus the known table plane is enough: each object pixel
is a ray, the ray hits the plane, and those hits cluster into blobs. Height
is not measured — bowls vs plates are told apart by radius and colour, the
same cues a person uses in a single photo.

`plane_z` can be the table (workspace look) or a plane just under the
fingers (in-hand look). Projecting an airborne bowl onto the table would
put it in the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

WORKSPACE_X = (-0.45, 0.40)
WORKSPACE_Y = (-0.45, 0.45)

# Only used if a caller forgets to pass the contact-calibrated plane.
TABLE_Z = 0.90


@dataclass
class Blob:
    """One connected lump of stuff sitting on the table."""

    idx: int
    xy: np.ndarray
    top_z: float
    mid_z: float
    table_z: float
    n_points: int
    extent: np.ndarray
    radius: float
    mean_rgb: np.ndarray
    dark_frac: float
    bright_frac: float
    sat: float
    points: np.ndarray = field(repr=False, default=None)
    colors: np.ndarray = field(repr=False, default=None)

    @property
    def height(self) -> float:
        return self.top_z - self.table_z

    @property
    def height_mid(self) -> float:
        return self.mid_z - self.table_z

    @property
    def aspect(self) -> float:
        a, b = np.sort(self.extent)[::-1]
        return float(a / max(b, 1e-6))

    @property
    def grasp_center(self) -> np.ndarray:
        return np.array([self.xy[0], self.xy[1], self.top_z])

    def describe(self) -> str:
        return (f"blob{self.idx}: xy=({self.xy[0]:+.3f},{self.xy[1]:+.3f}) "
                f"h={self.height*100:4.1f}cm mid={self.height_mid*100:4.1f}cm "
                f"r={self.radius*100:4.1f}cm "
                f"asp={self.aspect:.2f} rgb=({self.mean_rgb[0]:3.0f},"
                f"{self.mean_rgb[1]:3.0f},{self.mean_rgb[2]:3.0f}) "
                f"dark={self.dark_frac:.2f} sat={self.sat:.2f} n={self.n_points}")


@dataclass
class Scene:
    table_z: float
    blobs: List[Blob]

    def describe(self) -> str:
        lines = [f"table at z={self.table_z:.3f} m, {len(self.blobs)} objects on it"]
        lines += ["  " + b.describe() for b in self.blobs]
        return "\n".join(lines)


def _saturation(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def _object_masks(rgb: np.ndarray):
    """Foreground vs the warm wooden table, split by appearance class.

    Clustering plate and bowl pixels together would glue a bowl sitting next
    to the plate into one blob. The table is beige (R well above B). The
    plate is a neutral white, bowls/ramekin are neutral grey, the cookie
    box is saturated orange.
    """
    x = rgb.astype(np.float32)
    lum = x.mean(axis=-1)
    sat = _saturation(x)
    warm = (x[..., 0] - x[..., 2]) > 16.0
    cookie = (sat > 0.35) & (lum > 40.0)
    plate = (lum > 150.0) & (sat < 0.12) & ~warm
    vessel = (lum > 78.0) & (lum < 125.0) & (sat < 0.12) & ~warm
    return {"cookie": cookie, "plate": plate, "vessel": vessel}


def estimate_plane_z(obs, cam: str = "agentview", bin_m: float = 0.005) -> float:
    """Support height from depth alone: the most common z in the workspace.

    LIBERO's living-room scenes sit ~0.9 m below the kitchen ones, so no
    constant works across suites. One camera frame sees the whole tabletop,
    where the touch probe only ever samples the single point it descends on.
    """
    pts = obs.point_cloud(cam, stride=2)
    ok = (
        np.isfinite(pts).all(axis=1)
        & (pts[:, 0] > WORKSPACE_X[0]) & (pts[:, 0] < WORKSPACE_X[1])
        & (pts[:, 1] > WORKSPACE_Y[0]) & (pts[:, 1] < WORKSPACE_Y[1])
    )
    z = pts[ok, 2]
    if len(z) < 50:
        return 0.90
    lo, hi = np.quantile(z, 0.02), np.quantile(z, 0.98)
    if hi - lo < bin_m:
        return float(np.median(z))
    bins = np.arange(lo, hi + bin_m, bin_m)
    hist, edges = np.histogram(z, bins=bins)
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def _rays_on_plane(obs, cam: str, plane_z: float) -> np.ndarray:
    """World xyz of every pixel's ray hitting `z = plane_z`. OpenCV cam frame."""
    rgb = obs.rgb[cam]
    H, W = rgb.shape[:2]
    K = obs.intrinsics[cam]
    T = obs.cam_to_world[cam]
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    dirs = np.stack([
        (uu - K[0, 2]) / K[0, 0],
        (vv - K[1, 2]) / K[1, 1],
        np.ones_like(uu),
    ], axis=-1)
    R = T[:3, :3]
    t = T[:3, 3]
    d_world = dirs @ R.T
    denom = d_world[:, :, 2]
    s = np.divide(plane_z - t[2], denom, out=np.full_like(denom, np.nan),
                  where=np.abs(denom) > 1e-8)
    s = np.where(s > 0.0, s, np.nan)
    return t + s[..., None] * d_world


def _grid_index(pts: np.ndarray, cell: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
    gx = ((pts[:, 0] - WORKSPACE_X[0]) / cell).astype(int)
    gy = ((pts[:, 1] - WORKSPACE_Y[0]) / cell).astype(int)
    nx = int((WORKSPACE_X[1] - WORKSPACE_X[0]) / cell) + 1
    ny = int((WORKSPACE_Y[1] - WORKSPACE_Y[0]) / cell) + 1
    return gx, gy, nx, ny


def _components(pts: np.ndarray, cell: float, min_points: int,
                close: int = 3, fill: bool = True) -> List[np.ndarray]:
    if len(pts) == 0:
        return []
    gx, gy, nx, ny = _grid_index(pts, cell)
    ok = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    if not ok.any():
        return []
    gx, gy = gx[ok], gy[ok]
    pts_ok = ok
    grid = np.zeros((nx, ny), dtype=bool)
    grid[gx, gy] = True
    if close > 1:
        grid = ndimage.binary_closing(grid, structure=np.ones((close, close)))
    if fill:
        grid = ndimage.binary_fill_holes(grid)
    labels, n = ndimage.label(grid, structure=np.ones((3, 3)))
    per_point = np.zeros(len(pts), dtype=int)
    per_point[pts_ok] = labels[gx, gy]
    return [per_point == lab for lab in range(1, n + 1)
            if (per_point == lab).sum() >= min_points]


def _appear_z(radius: float, sat: float, aspect: float, lum: float,
              table_z: float) -> float:
    """Stand-in height from appearance. RGB cannot measure metres of height."""
    if sat > 0.25 and aspect > 1.15:
        return table_z + 0.019
    if lum > 155.0:
        return table_z + 0.015
    return table_z + 0.045


def _make_blob(p, c, table_z, idx) -> Blob:
    c = c.astype(np.float32)
    xy = p[:, :2].mean(axis=0)
    d = np.linalg.norm(p[:, :2] - xy, axis=1)
    lum = c.mean(axis=1)
    sat = float(_saturation(c).mean())
    extent = p[:, :2].max(axis=0) - p[:, :2].min(axis=0)
    radius = float(np.median(d))
    a, b = np.sort(extent)[::-1]
    aspect = float(a / max(b, 1e-6))
    z = _appear_z(radius, sat, aspect, float(lum.mean()), table_z)
    return Blob(
        idx=idx,
        xy=xy,
        top_z=z,
        mid_z=z,
        table_z=table_z,
        n_points=len(p),
        extent=extent,
        radius=radius,
        mean_rgb=c.mean(axis=0),
        dark_frac=float((lum < 90).mean()),
        bright_frac=float((lum > 190).mean()),
        sat=sat,
        points=p,
        colors=c,
    )


def segment_scene(
    obs,
    cam: str = "agentview",
    stride: int = 1,
    cell: float = 0.006,
    min_points: int = 20,
    self_radius: float = 0.13,
    plane_z: Optional[float] = None,
    flat_clearance: float = 0.010,
    tall_clearance: float = 0.035,
) -> Scene:
    """Segment tabletop objects from RGB and a supporting plane.

    `flat_clearance` / `tall_clearance` are unused; kept so old call sites
    do not break. Height cuts need a depth image we no longer read.
    """
    del flat_clearance, tall_clearance
    if cam not in obs.rgb:
        return Scene(table_z=TABLE_Z, blobs=[])

    z0 = TABLE_Z if plane_z is None else float(plane_z)
    world = _rays_on_plane(obs, cam, z0)
    rgb = obs.rgb[cam]
    classes = _object_masks(rgb)
    pts_all = world[::stride, ::stride].reshape(-1, 3)
    cols_all = rgb[::stride, ::stride].reshape(-1, 3)
    workspace = (
        (pts_all[:, 0] > WORKSPACE_X[0]) & (pts_all[:, 0] < WORKSPACE_X[1])
        & (pts_all[:, 1] > WORKSPACE_Y[0]) & (pts_all[:, 1] < WORKSPACE_Y[1])
        & np.isfinite(pts_all).all(axis=1)
    )
    not_self = np.linalg.norm(pts_all[:, :2] - obs.eef_pos[:2], axis=1) > self_radius

    blobs: List[Blob] = []
    for name, fg in classes.items():
        mask = fg[::stride, ::stride].reshape(-1) & workspace & not_self
        pts, cols = pts_all[mask], cols_all[mask]
        if len(pts) < min_points:
            continue
        fill = name != "vessel"
        close = 5 if name == "plate" else 3
        for m in _components(pts, cell, min_points, close=close, fill=fill):
            blobs.append(_make_blob(pts[m], cols[m], z0, len(blobs)))

    blobs = _merge_close(blobs, 0.048)
    blobs = _split_fused(blobs, z0)
    blobs.sort(key=lambda b: -b.n_points)
    for i, b in enumerate(blobs):
        b.idx = i
    return Scene(table_z=z0, blobs=blobs)


def agentview_place_align(obs, table_z: float, plate_anchor, eef_xy):
    """Plate disk vs held bowl, from the workspace cam — no re-grounding.

    A person watching agentview can see whether the bowl is over the plate
    centre. `self_radius=0.13` was hiding both. We only punch out the
    fingers (~2 cm) so the white disk and the grey bowl stay in the scene.
    Returns (plate_xy, bowl_xy); either may be None.
    """
    scene = segment_scene(
        obs, cam="agentview", self_radius=0.02, plane_z=float(table_z),
    )
    plate_anchor = np.asarray(plate_anchor, dtype=np.float64)[:2]
    eef_xy = np.asarray(eef_xy, dtype=np.float64)[:2]
    plates = [
        b for b in scene.blobs
        if float(b.mean_rgb.mean()) > 155
        and b.sat < 0.12
        and np.linalg.norm(b.xy - plate_anchor) < 0.06
    ]
    bowls = [
        b for b in scene.blobs
        if float(b.mean_rgb.mean()) < 140
        and b.sat < 0.15
        and np.linalg.norm(b.xy - eef_xy) < 0.055
    ]
    plate = (min(plates, key=lambda b: np.linalg.norm(b.xy - plate_anchor))
             if plates else None)
    bowl = (min(bowls, key=lambda b: np.linalg.norm(b.xy - eef_xy))
            if bowls else None)
    return (None if plate is None else plate.xy.copy(),
            None if bowl is None else bowl.xy.copy())


def wrist_vessel_xy(obs, table_z: float, near_xy=None,
                    max_shift: float = 0.11) -> Optional[np.ndarray]:
    """Grey bowl on the table, from the wrist, last few centimetres.

    Agentview at 128² shards the akita bowl and glues it to the ramekin.
    The wrist is close enough that the clay is one disk. Rays through those
    pixels hit the table at the bowl. `near_xy` is the first-look pick —
    refuse a jump onto the other bowl. None if the disk is gone.
    """
    cam = "robot0_eye_in_hand"
    if cam not in getattr(obs, "rgb", {}):
        return None
    rgb = obs.rgb[cam]
    vessel = _object_masks(rgb)["vessel"]
    if int(vessel.sum()) < 20:
        lum = rgb.astype(np.float32).mean(axis=-1)
        sat = _saturation(rgb)
        vessel = (lum > 70.0) & (lum < 145.0) & (sat < 0.20)
    if int(vessel.sum()) < 20:
        return None
    world = _rays_on_plane(obs, cam, float(table_z))
    pts = world[vessel]
    ok = np.isfinite(pts).all(axis=1)
    pts = pts[ok]
    if near_xy is not None and len(pts):
        near_xy = np.asarray(near_xy, dtype=np.float64)[:2]
        pts = pts[np.linalg.norm(pts[:, :2] - near_xy, axis=1) < 0.13]
    if len(pts) < 15:
        return None
    xy = pts[:, :2].mean(axis=0)
    if near_xy is not None and np.linalg.norm(xy - near_xy) > max_shift:
        return None
    return xy


def wrist_plate_xy(obs, table_z: float, near_xy=None,
                   max_shift: float = 0.14) -> Optional[np.ndarray]:
    """White plate on the table, from the wrist — not from a workspace re-name.

    Agentview above the plate while holding the bowl names the bowl as the
    plate. The wrist does not have that problem: the held bowl sits in the
    image centre (grey / fingers), and the plate is the pale disk around it.
    Rays through those pale pixels hit the table at the plate. None if the
    disk is gone or has jumped too far from the last trusted spot.
    """
    cam = "robot0_eye_in_hand"
    if cam not in getattr(obs, "rgb", {}):
        return None
    rgb = obs.rgb[cam]
    H, W = rgb.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    # Fingers + in-hand bowl. Their rays hit the table under the gripper,
    # not the plate; counting them would report "already centred".
    hand = (xx - cx) ** 2 + (yy - cy) ** 2 < (0.30 * min(H, W)) ** 2
    plate = _object_masks(rgb)["plate"] & ~hand
    if int(plate.sum()) < 30:
        return None
    world = _rays_on_plane(obs, cam, float(table_z))
    pts = world[plate]
    ok = np.isfinite(pts).all(axis=1)
    pts = pts[ok]
    if near_xy is not None and len(pts):
        near_xy = np.asarray(near_xy, dtype=np.float64)[:2]
        pts = pts[np.linalg.norm(pts[:, :2] - near_xy, axis=1) < 0.16]
    if len(pts) < 25:
        return None
    xy = pts[:, :2].mean(axis=0)
    if near_xy is not None and np.linalg.norm(xy - near_xy) > max_shift:
        return None
    return xy


def _split_fused(blobs: List[Blob], table_z: float) -> List[Blob]:
    """Bowl + ramekin often glue into one elongated grey lump at 128².

    A pinch at that centroid misses both rims. Split along the long axis
    when the lump is wide and stretched; keep cookies and plates whole.
    """
    out: List[Blob] = []
    for b in blobs:
        if (b.sat > 0.25 or float(b.mean_rgb.mean()) > 155
                or b.aspect < 1.38 or b.radius < 0.038
                or b.points is None or len(b.points) < 40):
            out.append(b)
            continue
        pts = b.points[:, :2]
        cols = b.colors
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        proj = (pts - c) @ vt[0]
        left, right = proj <= 0.0, proj > 0.0
        if int(left.sum()) < 15 or int(right.sum()) < 15:
            out.append(b)
            continue
        out.append(_make_blob(b.points[left], cols[left], table_z, len(out)))
        out.append(_make_blob(b.points[right], cols[right], table_z, len(out)))
    return out


def _merge_close(blobs: List[Blob], limit: float) -> List[Blob]:
    """Grey bowls break into rim shards at 128²; glue neighbours of one class."""
    if len(blobs) < 2:
        return blobs
    used = [False] * len(blobs)
    out: List[Blob] = []
    for i, a in enumerate(blobs):
        if used[i]:
            continue
        used[i] = True
        group = [a]
        for j, b in enumerate(blobs):
            if used[j]:
                continue
            if np.linalg.norm(a.xy - b.xy) > limit:
                continue
            if (float(a.mean_rgb.mean()) > 150) != (float(b.mean_rgb.mean()) > 150):
                continue
            if (a.sat > 0.25) != (b.sat > 0.25):
                continue
            used[j] = True
            group.append(b)
        if len(group) == 1:
            out.append(a)
            continue
        pts = np.concatenate([g.points for g in group])
        cols = np.concatenate([g.colors for g in group])
        out.append(_make_blob(pts, cols, a.table_z, len(out)))
    return out
