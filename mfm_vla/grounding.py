"""Map a LIBERO-spatial instruction onto segmented tabletop blobs.

LIBERO-Spatial tasks are all the same sentence with a different spatial clause:
    pick up the black bowl <clause> and place it on the plate

The bowls are not black (white clay, dark floral print), and the ramekin is
almost the same colour, so colour is a weak cue. Size, height and the clause
do the real work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from perception import Blob, Scene


@dataclass
class Grounding:
    pick: Optional[Blob]
    place: Optional[Blob]
    landmarks: dict
    reason: str

    def ok(self) -> bool:
        return self.pick is not None and self.place is not None


def _is_fixture(b: Blob) -> bool:
    """Panda mount / cabinet face: dark, or down in the -x-y corner.

    The cookie box is also dark brown — keep it. sat > 0.35 is the only
    saturated object on these tables.
    """
    if _is_cookie(b):
        return False
    if b.dark_frac > 0.42:
        return True
    # The stove-side bowl sits at ~(-0.26, −0.13). Only drop the corner
    # if it is also dark (Panda mount), not a grey vessel.
    if b.dark_frac > 0.30 and b.xy[1] < -0.04 and b.xy[0] < -0.18:
        return True
    return False


def _tableware(scene: Scene) -> List[Blob]:
    """Drop the robot mount, cabinet and other fixtures."""
    out = []
    for b in scene.blobs:
        if b.height > 0.11:
            continue
        if b.xy[0] < -0.38:  # wooden cabinet lives on the far -x side
            continue
        if b.radius > 0.12:
            continue
        if _is_fixture(b):
            continue
        out.append(b)
    return out


def _is_cookie(b: Blob) -> bool:
    # At 128² the cookie box is often a round saturated patch, not elongated.
    return b.sat > 0.35


def _is_plate(b: Blob) -> bool:
    # The plate is the only very white disk. Radius alone is weak in RGB:
    # a merged pair of grey vessels can look wider than the real plate.
    if _is_cookie(b):
        return False
    lum = float(b.mean_rgb.mean())
    return lum > 155.0 and 0.025 < b.radius < 0.12 and b.aspect < 1.70


def _is_bowlish(b: Blob) -> bool:
    # Both the akita bowl and the ramekin: circular, a few cm tall.
    if _is_cookie(b) or _is_plate(b):
        return False
    return 0.025 < b.height < 0.09 and b.aspect < 1.55 and 0.015 < b.radius < 0.07


def _pick_plate(cands: List[Blob]) -> Optional[Blob]:
    plates = [b for b in cands if _is_plate(b)]
    if plates:
        return max(plates, key=lambda b: b.radius)
    # Fallback: the widest thing whose median height is still dish-like.
    low = [b for b in cands if b.height_mid < 0.035 and not _is_cookie(b)]
    return max(low, key=lambda b: b.radius) if low else None


def _pick_cookie(cands: List[Blob]) -> Optional[Blob]:
    cookies = [b for b in cands if _is_cookie(b)]
    return max(cookies, key=lambda b: b.sat) if cookies else None


def _pick_ramekin(cands: List[Blob], bowls: List[Blob]) -> Optional[Blob]:
    vessels = _grey_disks(cands)
    if not vessels:
        return None
    roundish = [b for b in vessels if b.aspect < 1.50]
    pool = roundish or vessels
    return min(pool, key=lambda b: b.radius)


def _grey_disks(cands: List[Blob]) -> List[Blob]:
    """Akita shards are often not circular. Colour + size, not aspect."""
    return [
        b for b in cands
        if not _is_cookie(b) and not _is_plate(b) and not _is_fixture(b)
        and 0.015 < b.radius < 0.08
    ]


def _bowls(cands: List[Blob]) -> List[Blob]:
    bowls = _grey_disks(cands)
    bowls.sort(key=lambda b: -b.radius)
    return bowls


def _nearest(blobs: List[Blob], xy: np.ndarray) -> Optional[Blob]:
    if not blobs:
        return None
    return min(blobs, key=lambda b: np.linalg.norm(b.xy - xy))


def _between_plate_and_other(vessels: List[Blob], plate: Blob):
    """The bowl that actually sits on the plate–other segment.

    Naming the smallest grey disk 'ramekin' fails at 128²: the target bowl
    often shards smaller than the ramekin and gets dropped from the pool.
    Read the clause geometrically instead.
    """
    best = None
    best_cost = 1e9
    other_used = None
    for other in vessels:
        ab = other.xy - plate.xy
        n2 = float(np.dot(ab, ab))
        if n2 < 0.04 ** 2:
            continue
        for b in vessels:
            if b.idx == other.idx:
                continue
            t = float(np.dot(b.xy - plate.xy, ab) / n2)
            lat = float(np.linalg.norm((b.xy - plate.xy) - t * ab))
            if not (0.12 < t < 0.88 and lat < 0.09):
                continue
            cost = abs(t - 0.5) + lat
            if cost < best_cost:
                best, best_cost, other_used = b, cost, other
    return best, other_used


def ground(scene: Scene, instruction: str) -> Grounding:
    cands = _tableware(scene)
    plate = _pick_plate(cands)
    cookie = _pick_cookie(cands)
    bowls = _bowls(cands)
    ramekin = _pick_ramekin(cands, bowls)
    # Do not pick the ramekin when we mean a bowl, if we have a larger one.
    bowl_pool = [b for b in bowls if ramekin is None or b.idx != ramekin.idx]
    if not bowl_pool:
        bowl_pool = bowls

    landmarks = {
        "plate": plate,
        "cookie": cookie,
        "ramekin": ramekin,
        "bowls": bowl_pool,
    }

    if plate is None:
        return Grounding(None, None, landmarks, "no plate found")
    if not bowl_pool:
        return Grounding(None, plate, landmarks, "no bowl found")

    clause = instruction.lower()
    m = re.search(r"black bowl (.+?) and place", clause)
    rel = m.group(1).strip() if m else ""

    pick = None
    reason = rel or "default: largest bowl"

    if "between" in rel and plate is not None:
        # Shards of the akita bowl often have aspect > 1.55 and fail
        # `_is_bowlish`; the clause only needs grey disks, not a circle prior.
        vessels = _grey_disks(cands)
        pick, other = _between_plate_and_other(vessels, plate)
        if pick is None and ramekin is not None:
            mid = 0.5 * (plate.xy + ramekin.xy)
            pick = _nearest(bowl_pool, mid)
            reason = f"between fallback midpoint {mid}"
        else:
            reason = (
                f"between plate and blob{getattr(other, 'idx', '?')}, "
                f"pick blob{getattr(pick, 'idx', '?')}"
            )
    elif "next to the ramekin" in rel:
        greys = _grey_disks(cands)
        # In this suite the ramekin sits at y≈0.20; the bowl next to it
        # is the back-left grey (y>0.25). Nearest-to-smallest-grey often
        # returns the ramekin itself or a mount shard.
        back = [b for b in greys if b.xy[0] < 0.05 and b.xy[1] > 0.25]
        if back:
            pick = max(back, key=lambda b: b.n_points)
            reason = "next to ramekin (back-left)"
        elif ramekin is not None:
            pick = _nearest([b for b in greys if b.idx != ramekin.idx], ramekin.xy)
            reason = "next to ramekin"
    elif "next to the plate" in rel:
        greys = _grey_disks(cands)
        near = [b for b in greys if np.linalg.norm(b.xy - plate.xy) < 0.22]
        pick = _nearest(near or greys, plate.xy)
        reason = "next to plate"
    elif "next to the cookie" in rel and cookie is not None:
        greys = _grey_disks(cands)
        pick = _nearest(greys, cookie.xy)
        reason = "next to cookie box"
    elif "from table center" in rel:
        center = np.array([0.0, 0.0])
        # Back bowls sit at y≈0.31; "table center" is near the origin.
        central = [b for b in bowl_pool if abs(b.xy[0]) < 0.18 and abs(b.xy[1]) < 0.18]
        pick = _nearest(central or bowl_pool, center)
        reason = "table center"
    elif "on the cookie" in rel and cookie is not None:
        greys = _grey_disks(cands)
        pick = _nearest(greys, cookie.xy)
        reason = "on cookie box"
    elif "on the ramekin" in rel:
        # Stacked bowl+ramekin is one grey disk. Do not name it 'ramekin'
        # and then pick the other bowl.
        greys = _grey_disks(cands)
        zone = [b for b in greys if b.xy[0] < -0.05 and 0.12 < b.xy[1] < 0.28]
        if zone:
            pick = max(zone, key=lambda b: b.n_points)
            reason = "on ramekin (stack)"
        elif ramekin is not None:
            pick = _nearest(greys, ramekin.xy)
            reason = "on ramekin"
    elif "on the stove" in rel:
        greys = _grey_disks(cands)
        zone = [b for b in greys if -0.38 < b.xy[0] < -0.12 and -0.22 < b.xy[1] < 0.02]
        if zone:
            pick = max(zone, key=lambda b: b.n_points)
            reason = "on stove"
        else:
            pick = bowl_pool[0]
            reason = "on stove fallback"
    elif "on the wooden cabinet" in rel:
        greys = _grey_disks(cands)
        zone = [b for b in greys if b.xy[1] < -0.12]
        if zone:
            pick = max(zone, key=lambda b: b.n_points)
            reason = "on wooden cabinet"
        else:
            pick = bowl_pool[0]
            reason = "on cabinet fallback"
    elif "drawer" in rel:
        greys = _grey_disks(cands)
        zone = [b for b in greys if b.xy[0] > -0.05 and b.xy[1] < 0.0]
        if zone:
            pick = max(zone, key=lambda b: b.n_points)
            reason = "in top drawer"
        else:
            pick = bowl_pool[0]
            reason = "drawer fallback"
    else:
        # stove / cabinet / drawer: take the bowl that is *not* the obvious
        # table-center one if we can tell them apart, else the largest.
        pick = bowl_pool[0]
        reason = f"fallback ({rel or 'no clause'})"

    return Grounding(pick, plate, landmarks, reason)
