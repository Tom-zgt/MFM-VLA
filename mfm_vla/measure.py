"""Object size from the depth image, so the grasp can be shaped to the object.

`segment_scene` deliberately does not measure height — it intersects rays with
a known plane, so every blob comes back lying flat on it, and `_appear_z`
guesses a height from colour and aspect. That guess is fine for telling a
plate from a bowl and useless for deciding where to close the fingers on a
salad-dressing bottle.

The depth buffer already carries the real geometry. Taking the points that
clear the support plane near a grounded xy gives the object's true top height
and the width the gripper has to span, which is everything the grasp needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ObjectMetrics:
    """What the depth image says about one object standing on a plane."""

    top_z: float
    height: float
    width: float
    n_points: int
    offset: float = 0.0
    center_xy: Optional[np.ndarray] = None

    @property
    def squatness(self) -> float:
        """Width over height. Bowls are wide, bottles are tall."""
        return self.width / max(self.height, 1e-6)

    # What a Panda finger pair opens to. No margin is added and none is
    # needed: measured after re-centring, LIBERO-Object's groceries span
    # 3.2–7.6 cm and LIBERO-Spatial's bowls 8.5–10.6 cm, so the physical limit
    # falls in the gap between them rather than through either population.
    _MAX_SPAN = 0.08

    @property
    def graspable(self) -> bool:
        """Can the fingers close around this, or must they pinch its edge?

        This replaced a tall-vs-squat test, which asked the wrong question.
        Squatness sorts bowls (2.2) from bottles (0.25) correctly but puts a
        chocolate pudding box (2.8) and a stick of butter (4.3) on the bowl
        side, and they were being grasped 3 cm off centre at rim height —
        the fingers closed on air every time.

        The real distinction is mechanical. A bowl is wider than the gripper
        can open, so the only way to hold it is to trap its rim. A pudding box
        is 7.7 cm across and can simply be picked up.
        """
        return self.width < self._MAX_SPAN

    def describe(self) -> str:
        kind = "graspable" if self.graspable else "too wide, pinch rim"
        return (f"top={self.top_z:.3f} h={100 * self.height:4.1f}cm "
                f"w={100 * self.width:4.1f}cm w/h={self.squatness:.2f} "
                f"{kind} n={self.n_points}")

    def holds(self, opening: float) -> bool:
        """Is this gripper opening consistent with holding *this* object?

        The bowl-rim gate (0.5–5.5 cm) is a fact about bowls. A soup can
        closes at 6.3 cm, above its ceiling, so a perfect grasp was being
        thrown away. Measured width runs 0.3–0.5 cm under the opening the
        object actually produces, so the window is built around it.
        """
        op = float(opening)
        return 0.45 * self.width < op < min(self.width + 0.020, 0.072)


def asymmetric_grasp_direction(
    obs,
    cam: str,
    xy,
    plane_z: float,
    radius: float = 0.095,
    robot_mask=None,
) -> Optional[np.ndarray]:
    """Direction from an object's body toward its handle / protrusion.

    Successful mug and moka-pot demonstrations do not treat the handle as
    noise in the measured width.  They grasp at a repeatable angle relative
    to it.  The outer fifth of the local depth silhouette identifies that
    direction without reading object pose from the simulator.  A weighted
    circular mean is used instead of PCA so the 180-degree sign is retained.
    """
    xy = np.asarray(xy, dtype=np.float64)[:2]
    cloud = obs.point_cloud(cam)
    keep = np.isfinite(cloud).all(axis=1)
    if robot_mask is not None:
        keep &= ~np.asarray(robot_mask).ravel()
    cloud = cloud[keep]
    delta = cloud[:, :2] - xy
    radial = np.linalg.norm(delta, axis=1)
    local = (
        (radial < radius)
        & (cloud[:, 2] > float(plane_z) + 0.025)
        & (cloud[:, 2] < float(plane_z) + 0.14)
    )
    if np.count_nonzero(local) < 80:
        return None
    delta = delta[local]
    radial = radial[local]
    cutoff = float(np.percentile(radial, 80.0))
    outer = radial >= cutoff
    weights = np.maximum(radial[outer] - cutoff, 0.001)
    direction = np.average(delta[outer], axis=0, weights=weights)
    norm = float(np.linalg.norm(direction))
    # A round bowl has no directed protrusion.  Refuse to manufacture a yaw
    # from depth noise; callers retain their ordinary rim-pinch hypothesis.
    if norm < 0.18 * float(np.percentile(radial, 90.0)):
        return None
    return direction / norm


def estimate_top_normal(
    obs,
    cam: str,
    xy,
    plane_z: float,
    radius: float = 0.06,
    robot_mask=None,
) -> Optional[np.ndarray]:
    """Estimate the upward object / rim normal from its upper depth points.

    A bowl on a ramekin starts about 16 degrees from horizontal. Closing a
    horizontal gripper at its highest rim point puts both fingers on the same
    side and leaves the bowl behind. A least-squares plane through the upper
    third of the local cloud recovers 9.8 degrees from agentview depth (6.3
    degrees from the simulator normal), while reporting under one degree on
    ordinary table bowls.
    """
    xy = np.asarray(xy, dtype=np.float64)[:2]
    cloud = obs.point_cloud(cam)
    keep = np.isfinite(cloud).all(axis=1)
    if robot_mask is not None:
        keep &= ~np.asarray(robot_mask).ravel()
    cloud = cloud[keep]
    local = (
        (np.linalg.norm(cloud[:, :2] - xy, axis=1) < radius)
        & (cloud[:, 2] > float(plane_z) + 0.010)
    )
    points = cloud[local]
    if len(points) < 30:
        return None
    upper = points[points[:, 2] > np.percentile(points[:, 2], 65.0)]
    if len(upper) < 20:
        return None
    design = np.c_[
        upper[:, 0] - xy[0],
        upper[:, 1] - xy[1],
        np.ones(len(upper)),
    ]
    slope = np.linalg.lstsq(design, upper[:, 2], rcond=None)[0]
    normal = np.array([-slope[0], -slope[1], 1.0], dtype=np.float64)
    return normal / max(float(np.linalg.norm(normal)), 1e-12)


def local_plane_z(
    obs,
    cam: str,
    xy,
    r_in: float = 0.06,
    r_out: float = 0.11,
    bin_m: float = 0.005,
    support_frac: float = 0.20,
) -> Optional[float]:
    """The surface an object at `xy` is standing on, from the depth ring
    around it.

    Callers tend to pass a support height derived from somewhere else, and
    those are calibrated to one object class: the oracle grounder subtracts a
    constant measured on bowl origins, which lands 4.7 cm too high on a soup
    can and hides the whole can below the measurement floor.

    Takes the highest well-populated plane in the ring, not the most common
    one. A bowl standing on a cabinet has both the cabinet top and the distant
    tabletop in its neighbourhood, and the tabletop wins a vote on area — that
    reads the bowl as 25 cm tall and grasps it like a bottle.
    """
    xy = np.asarray(xy, dtype=np.float64)[:2]
    cloud = obs.point_cloud(cam, stride=2)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    d = np.linalg.norm(cloud[:, :2] - xy, axis=1)
    z = cloud[(d > r_in) & (d < r_out), 2]
    if len(z) < 40:
        return None
    lo, hi = np.quantile(z, 0.02), np.quantile(z, 0.98)
    if hi - lo < bin_m:
        return float(np.median(z))
    hist, edges = np.histogram(z, bins=np.arange(lo, hi + bin_m, bin_m))
    dense = np.flatnonzero(hist >= support_frac * hist.max())
    k = int(dense[-1])
    return float(0.5 * (edges[k] + edges[k + 1]))


def surface_top_z(
    obs,
    cam: str,
    xy,
    plane_z: float,
    radius: float = 0.09,
    clearance: float = 0.008,
    bin_m: float = 0.005,
    min_frac: float = 0.25,
    robot_mask=None,
) -> Optional[float]:
    """Height of the highest horizontal surface near `xy` — where to let go.

    Deliberately not the highest point. The arm is parked over the workspace
    during the first look, and a percentile of z answers "how high is the
    forearm": every LIBERO-Spatial plate measured 35 cm tall that way, which
    is exactly the hover height of the hand that had just probed the table.

    A surface puts many points in one 5 mm layer; a robot link smears a few
    points over twenty. Taking the highest *well-populated* layer reads the
    plate at 1.5 cm and the cabinet top at 21.8 cm, and ignores the arm.
    """
    xy = np.asarray(xy, dtype=np.float64)[:2]
    stride = 2
    cloud = obs.point_cloud(cam, stride=stride)
    keep = np.isfinite(cloud).all(axis=1)
    if robot_mask is not None:
        keep &= ~np.asarray(robot_mask)[::stride, ::stride].ravel()
    cloud = cloud[keep]
    near = np.linalg.norm(cloud[:, :2] - xy, axis=1) < radius
    z = cloud[near & (cloud[:, 2] > plane_z + clearance), 2]
    if len(z) < 30:
        return None
    lo, hi = np.quantile(z, 0.02), np.quantile(z, 0.98)
    if hi - lo < bin_m:
        return float(np.median(z))
    hist, edges = np.histogram(z, bins=np.arange(lo, hi + bin_m, bin_m))
    dense = np.flatnonzero(hist >= min_frac * hist.max())
    k = int(dense[-1])
    return float(0.5 * (edges[k] + edges[k + 1]))


def placed_object_xy(
    obs,
    cam: str,
    target_xy,
    table_z: float,
    radius: float = 0.11,
    robot_mask=None,
) -> Optional[np.ndarray]:
    """Metric centre of a bowl lying near its destination after release.

    At this phase the hand has moved aside and the upper bowl / rim points are
    separated from the plate by height. Their xy mean estimates ground-truth
    bowl centre within 0.3–1.1 cm on the two terminal failures where RGB
    ray-plane projection erred by over 2 cm.

    This intentionally is not used for an in-hand object: a camera sees only
    one hanging wall there and its point centroid is biased.
    """
    target_xy = np.asarray(target_xy, dtype=np.float64)[:2]
    cloud = obs.point_cloud(cam)
    keep = np.isfinite(cloud).all(axis=1)
    if robot_mask is not None:
        keep &= ~np.asarray(robot_mask).ravel()
    cloud = cloud[keep]
    local = (
        (np.linalg.norm(cloud[:, :2] - target_xy, axis=1) < radius)
        & (cloud[:, 2] > float(table_z) + 0.025)
        & (cloud[:, 2] < float(table_z) + 0.10)
    )
    points = cloud[local]
    if len(points) < 20:
        return None
    upper = points[points[:, 2] > np.percentile(points[:, 2], 55.0)]
    if len(upper) < 15:
        return None
    return upper[:, :2].mean(axis=0)


def measure_object(
    obs,
    cam: str,
    xy,
    plane_z: Optional[float] = None,
    radius: float = 0.055,
    clearance: float = 0.008,
    grasp_frac: float = 0.30,
    self_radius: float = 0.09,
    min_points: int = 25,
    robot_mask=None,
) -> Optional[ObjectMetrics]:
    """Height and graspable width of whatever stands near `xy` on `plane_z`.

    Width is measured in a thin slice at the height the fingers will close,
    not across the whole object. A dressing bottle is 4.4 cm at the base and
    2.8 cm at the neck, and only one of those numbers tells you whether the
    grasp will hold.

    Points near the end-effector are dropped. `segment_scene` punches the
    hand out for the same reason: parked over the table it is the tallest
    thing in the scene, and it reads as a 50 cm object.
    """
    xy = np.asarray(xy, dtype=np.float64)[:2]
    if plane_z is None:
        plane_z = local_plane_z(obs, cam, xy, r_in=radius + 0.005)
        if plane_z is None:
            return None
    cloud = obs.point_cloud(cam)
    finite = np.isfinite(cloud).all(axis=1)
    if robot_mask is not None:
        finite &= ~np.asarray(robot_mask).ravel()
    cloud = cloud[finite]
    above = (cloud[:, 2] > plane_z + clearance) & (
        np.linalg.norm(cloud[:, :2] - obs.eef_pos[:2], axis=1) > self_radius
    )

    def gather(centre):
        pts = cloud[above & (np.linalg.norm(cloud[:, :2] - centre, axis=1)
                             < radius)]
        if len(pts) < min_points:
            return None, None
        # A few stray pixels on a bright edge deproject far above the object.
        top = float(np.percentile(pts[:, 2], 95.0))
        grasp_z = top - grasp_frac * (top - plane_z)
        sl = pts[np.abs(pts[:, 2] - grasp_z) < 0.015]
        return pts, (sl if len(sl) >= 8 else pts)

    pts, slice_ = gather(xy)
    if pts is None:
        return None
    # One depth camera sees only the near face, so the points it returns are
    # an arc whose centroid sits toward the camera, and the disc that gathered
    # them is barely wider than a bowl. Off-centre by 2 cm it clips the far
    # rim and the bowl measures 9.0 cm instead of 10.4 — straddling the line
    # between "close the fingers around it" and "it does not fit".
    #
    # Re-gathering about the centroid removes the clipping, and matters most
    # exactly where it is most dangerous: when the grounding was already
    # imprecise, which is when the measurement gets used to overrule it.
    centred = slice_[:, :2].mean(axis=0)
    if np.linalg.norm(centred - xy) > 0.005:
        pts2, slice2 = gather(centred)
        if pts2 is not None:
            pts, slice_ = pts2, slice2

    top_z = float(np.percentile(pts[:, 2], 95.0))
    height = top_z - plane_z
    span = slice_[:, :2].max(axis=0) - slice_[:, :2].min(axis=0)
    # How far the points sit from the xy they were asked about. A disc centred
    # on the object collects it symmetrically; a disc centred 5 cm away clips
    # one side and everything derived from it is a measurement of a fragment.
    center_xy = slice_[:, :2].mean(axis=0)
    offset = float(np.linalg.norm(center_xy - xy))
    # One depth camera only ever returns the near surface, so a can's point
    # cloud is an arc, not a ring: across the viewing direction it spans the
    # full diameter, along it only half. Taking the smaller of the two axes
    # therefore halves every round object (6.6 cm soup can measured as 3.2).
    return ObjectMetrics(
        top_z, height, float(span.max()), len(pts), offset, center_xy.copy(),
    )
