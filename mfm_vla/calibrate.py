"""One-time table height from depth, expressed in the controller's frame.

A fixed touch point is not guaranteed to be bare in open-vocabulary scenes.
It pushed a Goal-suite bowl roughly 20 cm before planning even began. Depth
estimates the global support layer without contacting the scene.
"""

from __future__ import annotations

import numpy as np

from control import OPEN, move_to
from harness import AGENT_CAM
from perception import estimate_plane_z

# Reachable tabletop, away from the cookie box and the +y object cluster.
_PROBE_XY = (-0.08, 0.10)
# First look and the start of every approach. Home (-0.21, 0) hides the
# between-plate bowl; the probe hover clips some right-side layouts.
_LOOK_XY = (0.00, 0.06)
_LOOK_DZ = 0.26
# `estimate_plane_z` is fitted to the visible top table layer and empirically
# already matches the old open-hand stop within a few millimetres. Adding a
# second standoff put the plate pusher 6 mm above contact.
_CONTACT_STANDOFF = 0.0


def _descend_until_contact(env, xy, hover_z: float) -> float:
    """Lower the open gripper; contact = z has dropped a lot, then stalls."""
    xy = np.asarray(xy, dtype=np.float64)
    move_to(env, np.array([xy[0], xy[1], hover_z]), OPEN, max_steps=100, look_every=0)
    start_z = float(env.proprio().eef_pos[2])
    stalled = 0
    last_z = start_z
    lowest = start_z
    scale = env.action_pos_scale
    for _ in range(160):
        act = np.zeros(7)
        act[2] = -1.0
        act[6] = OPEN
        env.step(act)
        tr = getattr(env, "trace", None)
        if tr is not None:
            tr.act(env.t, env.proprio().eef_pos, act, note="probe down")
        if env.t % 3 == 0:
            env.grab_frame()
        new_z = float(env.proprio().eef_pos[2])
        lowest = min(lowest, new_z)
        dropped = start_z - new_z
        if dropped > 0.12 and last_z - new_z < 0.0015:
            stalled += 1
        else:
            stalled = 0
        last_z = new_z
        if stalled >= 6:
            break
        # Stop before driving through the floor. This used to be an absolute
        # `new_z < 0.70`, which is a fact about libero_spatial's table, not
        # about probing: libero_object starts the hand at 0.248 m and
        # libero_10 at 0.703 m, so the probe broke on its first step and
        # reported the hand's own starting height as the table — 25 cm too
        # high, and every grasp in both suites then missed by that much.
        if start_z - new_z > 0.50:
            break
    return lowest


def calibrate_table_z(env, force: bool = False) -> float:
    """Touch the table once and store `env.table_z`.

    Later episodes on the same env reuse the height but must still park at
    the probe hover. Skipping that left the hand at the reset home pose;
    first look and the approach path then disagreed with the first episode
    (task 0 init 1: 5/5 cold, 0/5 after another reset).
    """
    cached = getattr(env, "table_z", None)
    if cached is not None and getattr(env, "_table_calibrated", False) and not force:
        move_to(
            env,
            np.array([_LOOK_XY[0], _LOOK_XY[1], float(cached) + _LOOK_DZ]),
            OPEN,
            max_steps=80,
            look_every=0,
        )
        return float(cached)

    # The fitted visible table layer matches the controller's old open-hand
    # stop closely enough to retain the existing grasp constants.
    seen = estimate_plane_z(env.observe(), AGENT_CAM)
    contact = float(seen) + _CONTACT_STANDOFF
    env.table_z = contact
    env._table_calibrated = True
    move_to(
        env,
        np.array([_LOOK_XY[0], _LOOK_XY[1], contact + _LOOK_DZ]),
        OPEN,
        max_steps=80,
        look_every=0,
    )
    return contact
