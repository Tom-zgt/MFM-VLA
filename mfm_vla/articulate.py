"""Articulated primitives: pulling a drawer open, pushing one shut.

The pick-and-place routine cannot express these, and not for the reason it
looked like. A drawer handle is reachable and its joint is a plain slide; what
stops the hand is that `move_to` writes only `act[:3]` and `act[6]`, leaving
the three orientation channels at zero for the whole run. The wrist therefore
points down from the moment the episode starts, and a downward wrist cannot
close around a handle on a vertical face — driven at one, the hand's own body
hits the drawer above before the fingers arrive. Every task the controller
solves is one a top-down grasp happens to suit.

So this module does the one thing the rest of the stack never does: it turns
the wrist. Nothing here reads object state; `handle_xyz` and `axis` come from
the grounder on the same terms as a pick target.
"""

from __future__ import annotations

import numpy as np

from control import (
    CLOSE,
    OPEN,
    eef_orientation,
    hold,
    move_to,
    nominal_orientation,
    orientation_for_approach,
    rotvec_to_mat,
)


def approach_axis(env) -> np.ndarray:
    """Unit vector the gripper points along, in world coordinates."""
    return eef_orientation(env)[:, 2]


def face(env, direction, gripper: float = OPEN, max_steps: int = 70,
         tol: float = 0.06) -> bool:
    """Turn to a complete orientation with its approach along `direction`."""
    target_ori = orientation_for_approach(env, direction)
    return move_to(
        env, env.proprio().eef_pos, gripper, max_steps=max_steps,
        target_ori=target_ori, ori_tol=tol,
    )


def push_object(
    env,
    source_xy,
    target_xy,
    table_z: float,
    verbose: bool = False,
    use_goal_oracle: bool = True,
) -> bool:
    """Push a flat object along the table instead of trying to grasp it.

    LIBERO encodes "push the plate to the front of the stove" as an ordinary
    On(plate, region) predicate, so predicate shape alone cannot select this
    contact mode. The caller uses the language verb; this primitive guides the
    plate between open fingers and re-acquires it if it slips out.
    """
    source = np.asarray(source_xy, dtype=np.float64)[:2]
    target = np.asarray(target_xy, dtype=np.float64)[:2]
    # `table_z` is a touch-probe reading about 5–6 mm above the geometric
    # tabletop. Expert eef-plate-origin z is 1.50 cm, while the plate origin is
    # 0.75 cm above the table, so the matching command is probe + ~0.9 cm.
    # Using probe + 1.5 cm left both fingertips above the thin plate.
    contact_z = float(table_z) + 0.009

    current = source.copy()
    for attempt in range(5):
        delta = target - current
        distance = float(np.linalg.norm(delta))
        if distance < 0.015 or env.success:
            break
        direction = delta / distance
        lateral = np.array([-direction[1], direction[0]])
        z_axis = nominal_orientation(env)[:, 2]
        close_axis = np.r_[-direction, 0.0]
        close_axis -= z_axis * float(close_axis @ z_axis)
        close_axis /= np.linalg.norm(close_axis)
        x_axis = np.cross(close_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        push_ori = np.column_stack([x_axis, close_axis, z_axis])
        # Across 50 demonstrations, motion starts at eef-plate = +1.97 cm
        # along the path and -0.74 cm laterally.
        contact_xy = current + 0.020 * direction - 0.007 * lateral
        contact = np.r_[contact_xy, contact_z]
        hover = contact + np.array([0.0, 0.0, 0.12])

        hold(env, OPEN, steps=8)
        if not move_to(env, hover, OPEN, max_steps=75, target_ori=push_ori):
            return False
        if not move_to(env, contact, OPEN, tol=0.005, max_steps=65,
                       target_ori=push_ori):
            return False
        steps = max(3, int(np.ceil(distance / 0.045)))
        for k in range(1, steps + 1):
            goal = contact.copy()
            goal[:2] += direction * (distance * k / steps)
            move_to(env, goal, OPEN, tol=0.008, max_steps=40,
                    target_ori=push_ori)
            if env.success:
                break

        move_to(env, env.proprio().eef_pos + np.array([0.0, 0.0, 0.10]),
                OPEN, max_steps=45, target_ori=push_ori)
        if use_goal_oracle and env.success:
            break

        # The plate can lag the fingers or escape laterally. Re-acquire the
        # pale disk from a clear overhead view, then form a fresh contact
        # instead of continuing along a path where no object remains.
        from perception import segment_scene
        obs = env.observe(cameras=("agentview",))
        scene = segment_scene(
            obs, cam="agentview", self_radius=0.02, plane_z=float(table_z),
        )
        candidates = []
        for blob in scene.blobs:
            if float(blob.mean_rgb.mean()) <= 150 or blob.sat >= 0.15:
                continue
            along = float((blob.xy - source) @ direction)
            cross = abs(float((blob.xy - source) @ lateral))
            if -0.03 < along < np.linalg.norm(target - source) + 0.08 \
                    and cross < 0.10:
                candidates.append(blob)
        if not candidates:
            break
        current = min(
            candidates, key=lambda b: np.linalg.norm(b.xy - target),
        ).xy.copy()
        if verbose:
            print(f"    push re-acquire {attempt + 1}: "
                  f"remaining={100 * np.linalg.norm(target-current):.1f} cm")
    if verbose:
        print(f"    push finished; success={env.success}")
    return bool(env.success) if use_goal_oracle else True if use_goal_oracle else True


def turn_knob(
    env,
    anchor_xyz,
    axis,
    angle: float,
    verbose: bool = False,
    use_goal_oracle: bool = True,
) -> bool:
    """Close around a stove knob and follow its positive-hinge arc."""
    anchor = np.asarray(anchor_xyz, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    if angle <= 0.0:
        return bool(env.success) if use_goal_oracle else True

    # Demonstrations settle about 1–2 cm from the hinge and 2.42 cm above it.
    # More importantly, their wrist yaw then increases by 0.665 rad while the
    # eef stays near the hinge: they twist the grasp, rather than pushing the
    # knob around with a fixed-orientation hand.
    base_ori = nominal_orientation(env)
    radial = -base_ori[:, 1]
    radial -= axis * float(radial @ axis)
    radial = radial / np.linalg.norm(radial) * 0.012
    height = axis * 0.0242
    contact = anchor + radial + height
    hover = contact + axis * 0.12

    if not move_to(env, hover, OPEN, max_steps=90, target_ori=base_ori):
        return False
    if not move_to(env, contact, OPEN, tol=0.004, max_steps=90,
                   target_ori=base_ori):
        return False
    hold(env, CLOSE, steps=25, target_ori=base_ori)

    # Rodrigues rotation follows the knob's positive hinge direction. Small
    # angular increments keep both fingers engaged with the short handle.
    steps = max(5, int(np.ceil(angle / 0.10)))
    for k in range(1, steps + 1):
        theta = angle * k / steps
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        turn = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
        r = turn @ radial
        move_to(
            env, anchor + r + height, CLOSE, tol=0.004, max_steps=35,
            target_ori=turn @ base_ori, ori_tol=0.02,
        )
        if use_goal_oracle and env.success:
            break
    hold(env, OPEN, steps=12, target_ori=turn @ base_ori)
    move_to(env, hover, OPEN, max_steps=60, target_ori=turn @ base_ori)
    if verbose:
        print(f"    rotated stove knob {angle:.2f} rad; success={env.success}")
    return bool(env.success)


def slide_handle(
    env,
    handle_xyz,
    axis,
    distance: float = 0.16,
    standoff: float = 0.16,
    verbose: bool = False,
    approach_axis=None,
) -> bool:
    """Grasp a handle and travel along `axis` by `distance`.

    `axis` is the unit vector the handle has to move along; opening a drawer
    and shutting it are the same motion with the sign flipped.
    """
    handle = np.asarray(handle_xyz, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    approach = (
        axis.copy() if approach_axis is None
        else np.asarray(approach_axis, dtype=np.float64)
    )
    approach /= max(np.linalg.norm(approach), 1e-9)

    # Get out in front of the handle before dropping to its height. Coming
    # down any closer runs the fingers into the drawer above this one.
    stand = handle + approach * standoff
    move_to(env, stand + np.array([0.0, 0.0, 0.16]), OPEN, max_steps=90)
    if not move_to(env, stand, OPEN, tol=0.01, max_steps=110):
        if verbose:
            print("    could not get in front of the handle")
        return False

    side_ori = orientation_for_approach(env, -approach)
    if not move_to(
        env, stand, OPEN, max_steps=100, target_ori=side_ori, ori_tol=0.015,
    ):
        if verbose:
            print("    could not turn the wrist toward the drawer")
        return False

    # Straight in along the axis, so the fingers stay either side of the bar.
    contact = handle + approach * 0.006
    if not move_to(
        env, contact, OPEN, tol=0.003, max_steps=140,
        target_ori=side_ori, ori_tol=0.015,
    ):
        if verbose:
            print("    could not reach the handle")
        return False
    hold(env, CLOSE, steps=25, target_ori=side_ori)
    if verbose:
        print(f"    closed on handle: opening={env.proprio().gripper_opening:.4f}")

    # Travel in short hops. One long command lets the controller cut the
    # corner, and the handle is a 1.6 cm bar — a centimetre of drift across
    # the axis pulls the fingers off it.
    start = contact
    steps = max(2, int(round(distance / 0.04)))
    for k in range(1, steps + 1):
        move_to(env, start + axis * (distance * k / steps), CLOSE,
                tol=0.01, max_steps=50, target_ori=side_ori)
    hold(env, OPEN, steps=12, target_ori=side_ori)
    # Put the wrist back where the rest of the stack assumes it is: everything
    # downstream was written for a hand pointing straight down, and a bowl
    # fetched with the wrist still turned sideways is missed every time.
    # Climb well clear first — turning the hand while it is still level with
    # the drawer front sweeps the drawer shut again.
    clear = start + axis * distance + approach * 0.10
    move_to(env, clear, OPEN, max_steps=60, target_ori=side_ori)
    move_to(
        env, clear + np.array([0.0, 0.0, 0.18]), OPEN, max_steps=70,
        target_ori=side_ori,
    )
    move_to(env, clear + np.array([0.0, 0.0, 0.18]), OPEN, max_steps=70)
    return True


def push_drawer_closed(
    env,
    handle_xyz,
    open_axis,
    distance: float = 0.18,
    verbose: bool = False,
    use_goal_oracle: bool = True,
) -> bool:
    """Close a slide drawer with the expert open-hand terminal push."""
    handle = np.asarray(handle_xyz, dtype=np.float64)
    close_axis = -np.asarray(open_axis, dtype=np.float64)
    close_axis /= max(np.linalg.norm(close_axis), 1e-9)
    lateral = np.array([close_axis[1], -close_axis[0], 0.0])
    lateral /= max(np.linalg.norm(lateral), 1e-9)
    push_ori = rotvec_to_mat(np.array([2.304, -2.021, -1.029]))

    # Matched demonstration: eef-handle=(+3.0,-0.2,+0.5) cm while pushing.
    contact = (
        handle + 0.030 * lateral - 0.002 * close_axis
        + np.array([0.0, 0.0, 0.005])
    )
    hover = contact + np.array([0.0, 0.0, 0.12])
    # Reach the free-space point before applying the large wrist rotation;
    # asking OSC to translate and rotate there in one command stalls.
    move_to(env, hover, OPEN, max_steps=90)
    move_to(
        env, hover, OPEN, max_steps=120, target_ori=push_ori, ori_tol=0.08,
    )
    contact_ok = move_to(
        env, contact, OPEN, tol=0.006, max_steps=120,
        target_ori=push_ori, ori_tol=0.08,
    )
    contact_near = np.linalg.norm(env.proprio().eef_pos - contact) < 0.018
    if not contact_ok and not contact_near:
        if verbose:
            print("    could not reach drawer-push contact")
        return False

    # Continue beyond the nominal 16 cm joint stroke so contact force carries
    # qpos slightly positive, as required by WhiteCabinet.is_close().
    for alpha in np.linspace(0.0, 1.0, 7)[1:]:
        move_to(
            env, contact + close_axis * (distance * alpha), OPEN,
            tol=0.008, max_steps=40, target_ori=push_ori, ori_tol=0.08,
        )
        if env.success:
            break
    # Retreat out of the cabinet face as well as upward. A vertical-only
    # retreat left the wrist filling the agentview handle region, so the next
    # visual verification could not tell whether the drawer was closed.
    retreat = (
        env.proprio().eef_pos
        + np.asarray(open_axis, dtype=np.float64) * 0.12
        + np.array([0.0, 0.0, 0.12])
    )
    move_to(env, retreat, OPEN, max_steps=70, target_ori=push_ori)
    move_to(env, retreat, OPEN, max_steps=70)
    if verbose:
        print(f"    open-hand drawer push; success={env.success}")
    return bool(env.success)
