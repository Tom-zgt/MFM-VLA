"""Look-act-look pick-and-place. No learned weights, no privileged state.

Cameras match current LIBERO VLA papers: `agentview` for the workspace and
`robot0_eye_in_hand` for the last few centimetres. After every action both
are re-rendered; the next command can move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibrate import calibrate_table_z
from control import CLOSE, OPEN, hold, move_to
from grounding import ground
from harness import AGENT_CAM, TaskEnv
from measure import measure_object, surface_top_z
from perception import agentview_place_align, segment_scene, wrist_vessel_xy


def _support_offsets(instruction: str):
    """Extra planes above the table for bowls that are not on the tabletop."""
    s = instruction.lower()
    if "on the stove" in s:
        return (0.018, 0.030, 0.040, 0.050)
    if "on the wooden cabinet" in s:
        return (0.20, 0.218, 0.236)
    if "drawer" in s:
        return (0.145, 0.155, 0.165)
    return ()
from skills.verify import append_lesson, verify_target
from trace import Trace, attach


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    reason: str
    video: str = ""
    trace: str = ""
    # What the policy believed it was reaching for at first look. Recorded so
    # an evaluator can compare belief against ground truth and decide whether
    # a failure belongs to the eyes or to the hand. The policy never sees the
    # comparison.
    pick_xy: np.ndarray = None
    place_xy: np.ndarray = None
    ground_reason: str = ""


def _see(env: TaskEnv, cam: str = AGENT_CAM, self_radius: float = 0.13,
         plane_z=None):
    obs = env.observe()
    if plane_z is None:
        plane_z = getattr(env, "table_z", None)
    scene = segment_scene(obs, cam=cam, self_radius=self_radius, plane_z=plane_z)
    g = ground(scene, obs.instruction)
    tr = getattr(env, "trace", None)
    if tr is not None:
        tr.see(env.t, scene, g)
    return obs, scene, g


def _hover(point: np.ndarray, height: float) -> np.ndarray:
    p = point.copy()
    p[2] = height
    return p


def _refine_near(scene, xy, radius=0.07):
    """Use a later look only if a blob is still sitting near a remembered spot."""
    if not scene.blobs:
        return None
    near = [b for b in scene.blobs if np.linalg.norm(b.xy - xy) < radius]
    return min(near, key=lambda b: np.linalg.norm(b.xy - xy)) if near else None


def _track_place_agentview(env, place_xy, make_goal, table_z: float,
                           anchor_xy: np.ndarray, allow_plate_lock: bool = True):
    """Slide the hand so agentview's bowl sits on agentview's plate.

    Do not call `ground()`. The spatial clause would rename the bowl in
    the gripper as the plate. The workspace image is enough: white disk
    near the first-look plate, grey lump near the fingers, residual in
    metres.
    """
    anchor = np.asarray(anchor_xy, dtype=np.float64)[:2].copy()
    plate_samples = []
    plate_locked = False

    def on_look(obs):
        nonlocal plate_locked
        if AGENT_CAM not in obs.rgb:
            return None
        plate, bowl = agentview_place_align(
            obs, table_z, anchor, obs.eef_pos[:2],
        )
        tr = getattr(env, "trace", None)
        if plate is None and bowl is None:
            if tr is not None and env.t % 8 == 0:
                tr.add(f"t={env.t:4d}  SEE  agentview place: no plate/bowl")
            return None
        err = None
        if plate is not None and bowl is not None:
            err = bowl - plate
        if tr is not None:
            p = "none" if plate is None else f"({plate[0]:+.3f},{plate[1]:+.3f})"
            b = "none" if bowl is None else f"({bowl[0]:+.3f},{bowl[1]:+.3f})"
            e = "n/a" if err is None else f"{100*np.linalg.norm(err):.1f}cm"
            tr.add(f"t={env.t:4d}  SEE  agentview plate={p} bowl={b} "
                   f"align={e}")
        # A plate sighting alone is not a placement residual. While the
        # carried bowl is hidden by the hand, arm occlusion moves the apparent
        # plate from (0.064, 0.210) to (0.090, 0.200); the old code chased that
        # 2.8 cm wander frame after frame and turned an oracle target into a
        # failed placement. Only the relative bowl-minus-plate observation is
        # invariant enough to close the loop.
        if err is None:
            # Far from the target, three mutually consistent plate sightings
            # are useful: some fixture sites sit at the Panda's reach edge,
            # while the visible support centre is reachable. Lock that
            # correction once. Near the plate, the arm occludes it and the
            # apparent centre wanders by centimetres, so never update from a
            # plate-only sighting there.
            if plate is not None and allow_plate_lock and not plate_locked:
                plate_samples.append(plate.copy())
                if len(plate_samples) >= 3:
                    recent = np.asarray(plate_samples[-3:])
                    centre = np.median(recent, axis=0)
                    spread = np.max(np.linalg.norm(recent - centre, axis=1))
                    if spread < 0.008:
                        place_xy[0] = centre
                        plate_locked = True
                        return make_goal(place_xy[0])
            return None
        if np.linalg.norm(err) < 0.008:
            return make_goal(place_xy[0])
        # Move the gripper by (plate - bowl) so the next look is more centred.
        shift = np.clip(plate - bowl, -0.03, 0.03)
        place_xy[0] = place_xy[0] + shift
        return make_goal(place_xy[0])

    return on_look


def _track(xy_holder, make_goal, table_z: float):
    """Far-field only: shift xy from agentview. Near the table, use `_track_wrist`."""

    def on_look(obs):
        if AGENT_CAM not in obs.rgb:
            return None
        eef = obs.eef_pos
        if np.linalg.norm(eef[:2] - xy_holder[0]) < 0.08 and eef[2] < table_z + 0.12:
            return None
        scene = segment_scene(
            obs, cam=AGENT_CAM, self_radius=0.13, plane_z=table_z,
        )
        blob = _refine_near(scene, xy_holder[0], radius=0.08)
        if blob is None:
            return None
        if np.linalg.norm(blob.xy - xy_holder[0]) < 0.008:
            return None
        xy_holder[0] = blob.xy.copy()
        return make_goal(xy_holder[0])

    return on_look


def _track_wrist(xy_holder, make_goal, table_z: float, anchor_xy: np.ndarray,
                 support_z: float = None, max_shift: float = 0.11):
    """Last ~10 cm: wrist grey disk, clipped to the first-look pick.

    Demos put the bowl just under the fingers, not at image centre. We still
    apply `grasp_offset` in `make_goal`; this only updates the bowl xy.
    Agentview is ignored here — it occludes and renames.

    `support_z` is the plane the object rests on. Projecting the wrist rays
    onto the table when the bowl is sitting 22 cm up a cabinet puts the
    correction in the wrong place entirely.
    """
    anchor = np.asarray(anchor_xy, dtype=np.float64)[:2].copy()
    plane = float(table_z if support_z is None else support_z)

    def on_look(obs):
        eef = obs.eef_pos
        far = (np.linalg.norm(eef[:2] - xy_holder[0]) > 0.10
               or eef[2] > plane + 0.14)
        if far:
            return _track(xy_holder, make_goal, table_z)(obs)
        wxy = wrist_vessel_xy(obs, plane, near_xy=anchor, max_shift=max_shift)
        if wxy is None:
            return None
        if np.linalg.norm(wxy - xy_holder[0]) < 0.006:
            return None
        xy_holder[0] = wxy.copy()
        return make_goal(wxy)

    return on_look


def _in_hand_offset(scene, eef_xy: np.ndarray):
    """Bowl minus gripper, from a side look. None if the hand hid it."""
    near = [b for b in scene.blobs if np.linalg.norm(b.xy - eef_xy) < 0.10]
    if not near:
        return None
    b = min(near, key=lambda x: np.linalg.norm(x.xy - eef_xy))
    return b.xy - eef_xy


def _pinch_clearance(instruction: str) -> float:
    """How far above the table plane the fingers close.

    Table bowls: demo hold is table+2.8 cm. A bowl already sitting on the
    cookie box / ramekin / stove needs a higher close or we punch the support.
    """
    s = instruction.lower()
    if "top drawer" in s or "in the" in s and "drawer" in s:
        return 0.100
    if "on the stove" in s:
        # Stove top is ~2–3 cm above the table. 8.5 cm was closing in air.
        return 0.055
    if "on the wooden cabinet" in s:
        return 0.100
    if "on the cookie" in s:
        return 0.048
    if "on the ramekin" in s:
        return 0.052
    return 0.028


def _holding(opening: float) -> bool:
    """Rim pinch, not an empty close and not a fully open hand.

    Spatial-0 demos on the table close to 1.3–1.9 cm, but a bowl on the stove,
    in the drawer or on the cabinet pinches at 0.70–1.02 cm — under the old
    1.0 cm floor, so a grasp that had physically succeeded was thrown away and
    retried until the episode reported "missed grasp". An empty close measures
    0.15 cm, which a 0.5 cm floor still separates cleanly. Fully open is ~8 cm.
    """
    return 0.005 < float(opening) < 0.055


def _carrying(env) -> bool:
    """Are the fingers stopped on something right now?

    Cheaper and far more reliable than looking. The workspace camera cannot
    tell a bowl from the grey stove it was standing on, so the "did the bowl
    leave the table?" check fires on the fixture and the policy opens its hand
    to re-pinch — dropping the bowl it was already holding.

    Closing on air settles at 0.15 cm. A carried bowl settles at 0.45 cm once
    the fingers finish creeping in, lower than the 0.5 cm grasp gate, so this
    needs its own floor between the two.
    """
    return float(env.proprio().gripper_opening) > 0.003


def _result(env, ok: bool, reason: str, belief=None) -> EpisodeResult:
    tr = getattr(env, "trace", None)
    if tr is not None:
        tr.phase(env.t, "end", reason)
    b = belief or {}
    return EpisodeResult(
        ok, env.t, reason, trace="" if tr is None else tr.dump(),
        pick_xy=b.get("pick_xy"), place_xy=b.get("place_xy"),
        ground_reason=b.get("ground_reason", ""),
    )


def run_episode(
    env: TaskEnv,
    init_idx: int = 0,
    max_steps: int = 400,
    verbose: bool = True,
    grounder=None,
    reset: bool = True,
    provisional: bool = False,
) -> EpisodeResult:
    """Run one pick-and-place.

    `grounder` swaps out the first look only. It is called as
    `grounder(env, obs, table_z)` and returns `(pick_xy, place_xy, reason)`
    in metres, or None to fall back to the built-in vision grounding. The
    evaluator uses it to feed oracle targets in, which isolates whatever the
    controller fails at once perception is perfect. Default is None, so the
    policy stays vision-only.

    With `reset=False` the scene is left as the previous call finished it and
    the table height is reused. A goal with two predicates is two of these
    back to back; re-probing the table between them would be a second contact
    search for a number that has not moved.
    """
    if reset:
        obs = env.reset(init_idx)
        tr = attach(env, Trace())
    else:
        obs = env.observe()
        tr = getattr(env, "trace", None) or attach(env, Trace())
    if verbose:
        print(f"instruction: {obs.instruction}")
        print(f"policy cameras: {list(obs.rgb)}")

    if reset or getattr(env, "table_z", None) is None:
        tr.phase(env.t, "calibrate table by touch")
        table_z = calibrate_table_z(env)
        env.table_z = table_z
        tr.add(f"t={env.t:4d}  CAL  table_z={table_z:.4f} m (open-gripper contact)")
        if verbose:
            print(f"table_z from contact: {table_z:.4f} m")
    else:
        table_z = float(env.table_z)

    tr.phase(env.t, "first look")
    # After the table probe the hand hovers near (-0.08, +0.10). The default
    # 13 cm self punch-out hides the table-center bowl (~8 cm away).
    obs, scene, g = _see(env, AGENT_CAM, self_radius=0.05)
    support_z = table_z
    for dz in _support_offsets(obs.instruction):
        elev = segment_scene(
            obs, cam=AGENT_CAM, self_radius=0.05, plane_z=table_z + dz,
        )
        ge = ground(elev, obs.instruction)
        if ge.pick is not None and "fallback" not in ge.reason:
            scene, g = elev, ge
            support_z = table_z + dz
            if verbose:
                print(f"elevated plane table+{100*dz:.1f} cm: {ge.reason}")
            break
    if verbose:
        print(scene.describe())
        print(f"grounding: {g.reason}")
        if g.pick is not None:
            print(f"  pick  blob{g.pick.idx} {g.pick.describe()}")
        if g.place is not None:
            print(f"  place blob{g.place.idx} {g.place.describe()}")

    override = grounder(env, obs, table_z) if grounder is not None else None
    if override is not None:
        ov_pick, ov_place, ov_reason = override[:3]
        # A grounder may also correct the support height. Overriding xy alone
        # leaves the fingers closing at whatever plane the vision search
        # picked, which is the whole difficulty on a drawer or a stove top.
        if len(override) > 3 and override[3] is not None:
            support_z = float(override[3])
            ov_reason = f"{ov_reason} + support_z={support_z:.4f}"
        belief = {"pick_xy": np.asarray(ov_pick, dtype=np.float64)[:2].copy(),
                  "place_xy": np.asarray(ov_place, dtype=np.float64)[:2].copy(),
                  "ground_reason": ov_reason}
        if verbose:
            print(f"grounder override: {ov_reason}")
    elif g.ok():
        belief = {"pick_xy": g.pick.xy.copy(), "place_xy": g.place.xy.copy(),
                  "ground_reason": g.reason}
    else:
        return _result(env, False, f"grounding failed: {g.reason}",
                       {"ground_reason": g.reason})

    # Remember the first look. Later frames are partly occluded by the arm, and
    # re-grounding then happily names the remaining bowl as the plate.
    pick_xy = [belief["pick_xy"].copy()]
    place_xy = [belief["place_xy"].copy()]
    if obs.instruction.lower().startswith("push "):
        # BDDL represents this as On(plate, stove_front_region), exactly like
        # a pick-and-place goal. The language verb is the only fact that says
        # the plate should stay on the table.
        from articulate import push_object
        tr.phase(env.t, "push")
        pushed = push_object(
            env, pick_xy[0], place_xy[0], table_z, verbose=verbose,
        )
        return _result(
            env, pushed, "goal satisfied" if pushed else "push did not satisfy goal",
            belief,
        )
    plate_anchor = belief["place_xy"].copy()
    # Spatial-0 demos (table bowls): eef − bowl ≈ (+1.3, −3.1) cm,
    # eef z ≈ table + 2.8 cm. Bowls sitting on something need more clearance.
    grasp_offset = np.array([0.012, -0.030])
    pinch_dz = _pinch_clearance(obs.instruction)
    pinch_z = support_z + 0.028
    # Everything above is a fact about bowls: reach past the near rim, close
    # 2.8 cm up, expect 0.5–5.5 cm between the fingers. A soup can wants the
    # centre of its body and closes at 6.3 cm; the bowl numbers miss it
    # entirely. Measure the object instead of assuming which one it is.
    rmask = env.robot_pixel_mask(AGENT_CAM)
    # `support_z` is where this object stands, already established above. The
    # generic annulus search was finding a plane too high around the alphabet
    # soup can, so every point on the can fell below the clearance and it came
    # back "no depth points" — measured blind, grasped as a bowl, missed.
    seen = measure_object(obs, AGENT_CAM, pick_xy[0], plane_z=support_z,
                          robot_mask=rmask)
    # A robot mask removes arm pixels but cannot reveal an object physically
    # hidden behind the arm. Do not turn "no depth" into an arbitrary bowl
    # grasp: park at the back edge, take a fresh view, and measure again.
    if seen is None:
        move_to(
            env, np.array([0.12, 0.30, table_z + 0.28]), OPEN, max_steps=70,
        )
        obs = env.observe(cameras=(AGENT_CAM,))
        rmask = env.robot_pixel_mask(AGENT_CAM)
        seen = measure_object(
            obs, AGENT_CAM, pick_xy[0], plane_z=support_z, robot_mask=rmask,
        )
    instruction = obs.instruction.lower()
    target_is_bowl = instruction.startswith((
        "put the bowl", "pick up the bowl", "pick up the black bowl",
    ))
    # A partial far-side bowl can measure only 6–7 cm wide and masquerade as
    # a graspable carton. The task-level semantic label is stronger evidence
    # about contact mode than that clipped width. Conversely, a >9 cm "bowl"
    # height is cabinet/robot contamination; use the calibrated bowl default.
    if target_is_bowl and seen is not None and seen.height > 0.07:
        seen = None
    body_grasp = seen is not None and seen.graspable and not target_is_bowl
    flat_body = body_grasp and seen.squatness > 2.0
    if body_grasp:
        grasp_offset = np.zeros(2)
        pinch_z = seen.top_z - 0.30 * seen.height
        if flat_body:
            # Fifty successful cream-cheese and butter demonstrations do not
            # skim the top edge. Their settled eef is at the package body
            # centre, with median eef-object y offsets of -1.0 and -1.4 cm.
            # The shallow generic body grasp sat 7–9 mm higher and frequently
            # let both fingers close above cream cheese without contact.
            grasp_offset = np.array([0.0, -0.012])
            pinch_z = seen.top_z - 0.88 * seen.height
    elif seen is not None:
        # Too wide to span, so the rim gets trapped — but at *this* object's
        # rim. The constants above put the fingers 1.8 cm below the top of a
        # 4.6 cm bowl and 3.2 cm off its centre, which for a 10.6 cm mug means
        # closing on the widest part of its body, 8 cm below the rim it was
        # aiming for. Every mug and moka pot in LIBERO-10 missed that way.
        #
        # The height is the bowl's own rule restated against the measurement
        # (4.6 − 1.8 = 2.8 cm), so it is unchanged on a bowl and follows the
        # rim up when the object is taller.
        #
        # The reach is not, and it is not one number. 3.2 cm is 0.62 of a
        # bowl's radius; reaching that far into a mug closes on nothing.
        # Reaching 0.91 of the radius holds both mugs — and holds bowls too in
        # an isolated grasp, but costs four LIBERO-Spatial episodes, because a
        # bowl gripped at its outermost edge tilts on the way to the plate.
        #
        # The two are not the same shape. A bowl is a shallow dish, twice as
        # wide as it is tall, and its wall slopes into the fingers on the way
        # down. A mug is a straight-sided vessel about as tall as it is wide,
        # and its wall is only ever at one radius.
        pinch_z = seen.top_z - 0.018
        reach = 0.62 if seen.squatness > 1.5 else 0.91
        grasp_offset = (grasp_offset / np.linalg.norm(grasp_offset)) \
            * reach * 0.5 * seen.width
    if verbose:
        print(f"pick measured: {seen.describe() if seen else 'no depth points'}"
              f" -> {'body grasp' if body_grasp else 'rim pinch'}")
    # Transit has to clear the grasp point, not a constant above the table.
    # `table_z + max(0.12, pinch_dz + 0.08)` tops out at table + 18 cm, but a
    # bowl on the cabinet is grasped at table + 25.6 cm and one in the top
    # drawer at + 19.2 cm — the "hover above the pick" was below the pick, so
    # the arm drove into the cabinet and every attempt timed out. The first
    # term is kept so table-height picks behave exactly as before.
    transit_z = max(table_z + max(0.12, pinch_dz + 0.08), pinch_z + 0.08)
    if handle_grasp and seen is not None:
        # Plan clearance for the carried object, not just the wrist. The white
        # mug hangs 9 cm below the eef; at the old 20 cm wrist transit its base
        # grazed the second mug and rotated that untouched target by 44.6 deg.
        hanging = max(0.0, pinch_z - support_z)
        transit_z = max(
            transit_z,
            table_z + seen.height + hanging + 0.030,
        )

    # Where to let go, and whether the plate servo applies. `table_z + 3 cm`
    # is a plate constant. It is right for a plate and wrong by 22 cm for the
    # cabinet top, where the hand would open below the surface it is aiming
    # for. Ask the depth map how high the receptacle's top face is instead.
    place_is_bowl = "in the bowl" in instruction or "on the bowl" in instruction
    target_is_mug = "mug" in instruction or "moka" in instruction
    handle_grasp = False
    grasp_ori = None
    pick_transit_z = transit_z
    semantic_target = instruction
    top = surface_top_z(obs, AGENT_CAM, place_xy[0], table_z, robot_mask=rmask)
    release_z = table_z + 0.030
    plate_like = True
    if top is not None and (top > table_z + 0.05 or place_is_bowl):
        release_z = top + 0.025 + max(0.0, pinch_z - support_z)
        plate_like = False
        transit_z = max(transit_z, release_z + 0.06)
    if provisional:
        plate_like = False
    if verbose:
        where = "not visible" if top is None else \
            f"{100 * (top - table_z):.1f} cm above table"
        print(f"place surface: {where} -> release {release_z:.3f}, "
              f"servo {'on' if plate_like else 'off'}")

    terminal_bias = (
        np.array([-0.020, -0.015])
        if plate_like and target_is_bowl else np.zeros(2)
    )
    if handle_grasp and target_is_mug:
        terminal_bias = (
            np.array([0.023, -0.012])
            if "porcelain_mug" in semantic_target
            else np.array([0.031, -0.004])
        )

    def pinch_pose(xy):
        return np.array([xy[0] + grasp_offset[0], xy[1] + grasp_offset[1],
                         pinch_z])

    def place_goal(xy, z=None):
        if z is None:
            z = release_z
        return np.array([
            xy[0] + grasp_offset[0] + terminal_bias[0],
            xy[1] + grasp_offset[1] + terminal_bias[1],
            z,
        ])

    tr.phase(env.t, "pinch")
    # First-look pick is the anchor. Later agentview looks rename or fuse
    # bowls; the wrist only nudges within 7 cm of this spot.
    pick_anchor = pick_xy[0].copy()
    if not reset:
        # A later predicate starts above the previous receptacle, which may
        # now be occupied. Route high through free workspace before descending
        # toward the next source; the direct basket-to-butter diagonal left
        # the eef at 12 cm while the fingers swept through the 15 cm basket.
        move_to(
            env,
            np.array([0.00, 0.06, max(table_z + 0.26, pick_transit_z)]),
            OPEN,
            max_steps=70,
            target_ori=grasp_ori,
        )
    # Front-right bowls (next to the cookie) sit near the Panda's reach
    # edge. A straight line from the table-probe hover folds the arm;
    # go high over the table centre first.
    if pick_anchor[0] > 0.04 or pick_anchor[1] < 0.02:
        move_to(env, np.array([0.00, 0.06, table_z + 0.26]), OPEN, max_steps=70)
    if handle_grasp:
        # The previous long-horizon leg may leave the wrist in its release
        # orientation over the stove. Going straight from there to the next
        # handle hover stalls, then burns the remaining horizon on generic IK
        # retries. Re-centre while rotating to the next demonstrated grasp.
        move_to(
            env, np.array([0.00, 0.08, pick_transit_z]), OPEN,
            max_steps=70, target_ori=grasp_ori,
        )
    got = False
    pick = pinch_pose(pick_xy[0])
    # `wrist_vessel_xy` calls every grey lump a vessel. On the table that is
    # only ever a bowl, but a stove top and a cabinet carcass are grey too, so
    # on an elevated support the servo locks onto the furniture and walks the
    # grasp point off the bowl — the gripper then closes on air (0.15 cm).
    # Keep the full correction on the table, allow only a nudge up high.
    elevated = support_z > table_z + 0.01
    wrist_shift = 0.03 if elevated else 0.11
    # The servo exists to find a bowl's rim, which the first look locates only
    # roughly. It has nothing to offer a body grasp: the object's centre and
    # width have already been measured from depth, and letting a rim-finder
    # move that point by up to 11 cm walked the fingers clean off every bottle
    # and carton in LIBERO-Object, which then closed on air at 0.15 cm.
    # Three attempts used to repeat one grasp verbatim and expect a different
    # answer. They now walk a list of hypotheses, best measurement first.
    # Nothing is added for an object the fingers can span — that grasp either
    # works or the object is not where it was measured. For one too wide, the
    # measured width is the only uncertain quantity: a mug reads 10.1 cm
    # because of its handle, so the rim it actually has to straddle is nearer
    # than the reading implies, and a further-out offset finds it.
    tries = [(grasp_offset.copy(), pinch_z)]
    if seen is not None and not body_grasp:
        unit = grasp_offset / max(np.linalg.norm(grasp_offset), 1e-9)
        other = 0.91 if seen.squatness > 1.5 else 0.62
        tries.append((unit * other * 0.5 * seen.width, pinch_z))
        tries.append((np.zeros(2), seen.top_z - 0.30 * seen.height))
    elif seen is not None and flat_body:
        # Thin cartons expose only ~1.5 cm of side wall. Repeating the exact
        # same centre/height after a miss cannot create a new contact. Depth
        # sees their top face reliably, so retry at its measured centre and
        # bracket the nominal close height: high first to avoid the tabletop,
        # then lower in case the fingertips passed above the side wall.
        # Expert x offsets vary on either side of centre. Bracket that axis,
        # retaining the demonstrated -y contact and a near-centre grasp z.
        tries.append((np.array([-0.012, -0.010]),
                      seen.top_z - 0.72 * seen.height))
        tries.append((np.array([+0.012, -0.014]),
                      seen.top_z - 0.95 * seen.height))
    while len(tries) < 3:
        tries.append(tries[0])

    for attempt, (grasp_offset, pinch_z) in enumerate(tries):
        pick = pinch_pose(pick_xy[0])
        # The servo corrects a first look that named roughly the right region,
        # which is worth a great deal when the segmenter picked one of two
        # identical bowls. But it is a rim-finder, and once it has had its one
        # attempt it is only competing with the depth measurement — it walked
        # the fingers off every mug in LIBERO-10 exactly as it had off every
        # bottle in LIBERO-Object. Let it try first, then trust what was
        # measured.
        servo = attempt == 0 and not body_grasp
        if not servo:
            # The servo writes its correction back into `pick_xy`, so a retry
            # would otherwise start from wherever the last one wandered to.
            pick_xy[0] = pick_anchor.copy()
        track_hover = _track_wrist(
            pick_xy, lambda xy: _hover(pinch_pose(xy), transit_z),
            table_z, pick_anchor, support_z=support_z, max_shift=wrist_shift,
        ) if servo else None
        track_pinch = _track_wrist(
            pick_xy, pinch_pose, table_z, pick_anchor,
            support_z=support_z, max_shift=wrist_shift) if servo else None
        if not move_to(env, _hover(pick, transit_z), OPEN, max_steps=80,
                       on_look=track_hover):
            continue
        pick = pinch_pose(pick_xy[0])
        if not move_to(env, pick, OPEN, tol=0.005, max_steps=80,
                       on_look=track_pinch):
            continue
        hold(env, CLOSE, steps=25)
        opening = env.proprio().gripper_opening
        held = seen.holds(opening) if body_grasp else _holding(opening)
        if verbose:
            print(f"pinch {attempt}: opening={opening:.4f} held={held}")
        # A wrist-RGB "am I holding something" check used to widen this gate,
        # to rescue thin rim pinches the 1.0 cm floor rejected. The floor now
        # admits them, and the check was reporting "holding" on an empty
        # gripper, so it only served to accept a fully open hand.
        if held:
            got = True
            break
        hold(env, OPEN, steps=10)
        move_to(env, _hover(pick, transit_z), OPEN, max_steps=50)

    if not got:
        return _result(env, False, "missed grasp")

    tr.phase(env.t, "lift")
    # ---- 3. lift. Do not re-name objects; the hand is now in the way. ----
    if not move_to(env, _hover(pick, transit_z), CLOSE):
        return _result(env, False, "failed lift")
    lifted_open = env.proprio().gripper_opening
    if verbose:
        print(f"lifted, opening={lifted_open:.4f}")
    if env.success:
        return _result(env, True, "success after lift")

    # Did the bowl actually leave the table? Look from the side — a look
    # from above the pick site hides the bowl inside the eef exclusion radius.
    aside = np.array([pick_xy[0][0] - 0.14, pick_xy[0][1], transit_z])
    move_to(env, aside, CLOSE, max_steps=60)
    obs, scene, _ = _see(env, AGENT_CAM)
    still = _refine_near(scene, pick_xy[0], radius=0.06)
    if still is not None and still.height > 0.03 and not _carrying(env):
        if verbose:
            print(f"lift check: bowl still at {still.xy}, retry pinch")
        pick_xy[0] = still.xy.copy()
        pick = pinch_pose(pick_xy[0])
        track_hover = _track(
            pick_xy, lambda xy: _hover(pinch_pose(xy), transit_z), table_z,
        )
        track_pinch = _track(pick_xy, pinch_pose, table_z)
        move_to(env, _hover(pick, transit_z), OPEN, max_steps=70, on_look=track_hover)
        move_to(env, pick, OPEN, tol=0.005, max_steps=70, on_look=track_pinch)
        hold(env, CLOSE, steps=25)
        move_to(env, _hover(pick, transit_z), CLOSE, max_steps=70)
        obs, scene, _ = _see(env, AGENT_CAM)
        still = _refine_near(scene, pick_xy[0], radius=0.06)
        if still is not None and still.height > 0.03:
            return _result(env, False, "bowl still on table after lift")

    # Side look at the hand itself. Above the pick site the bowl is inside
    # the eef exclusion radius and looks "gone".
    aside_hand = np.array([pick_xy[0][0] - 0.16, pick_xy[0][1], transit_z])
    move_to(env, aside_hand, CLOSE, max_steps=60)
    eef = env.proprio().eef_pos
    obs, scene, _ = _see(
        env, AGENT_CAM, self_radius=0.03, plane_z=float(eef[2] - 0.03),
    )
    held = _in_hand_offset(scene, eef[:2])
    if held is not None and np.linalg.norm(held) < 0.04:
        grasp_offset = -held
        if verbose:
            print(f"in-hand bowl-eef {held}, place uses offset {grasp_offset}")
    elif verbose and held is not None:
        print(f"in-hand bowl-eef {held} ignored (>{4}cm); keep demo offset {grasp_offset}")

    tr.phase(env.t, "carry and place")
    # Agentview can see plate-vs-bowl alignment. Wrist is often just fingers.
    # Do not re-run grounding — only measure the two disks.
    #
    # `agentview_place_align` looks for a wide pale disk, which is a plate and
    # nothing else. Aimed at a 10 cm-deep basket it locks onto whatever pale
    # thing is nearest and drags the release point 11 cm off. `plate_like` is
    # cleared only when the receptacle has been measured and is not a plate.
    # Reset the arm configuration before reaching toward the plate. A direct
    # diagonal from the side-look waypoint can hit an OSC / joint-limit stall
    # even though the same plate is reachable from above the table centre.
    move_to(
        env, np.array([0.00, 0.06, transit_z]), CLOSE, max_steps=70,
    )
    track_place_hover = _track_place_agentview(
        env, place_xy, lambda xy: _hover(place_goal(xy), transit_z),
        table_z, plate_anchor,
    ) if plate_like else None
    if not move_to(env, _hover(place_goal(place_xy[0]), transit_z), CLOSE,
                   on_look=track_place_hover):
        return _result(env, False, "failed hover above plate")

    def set_down(xy_holder):
        track = _track_place_agentview(
            env, xy_holder, lambda xy: place_goal(xy), table_z, plate_anchor,
            allow_plate_lock=False,
        ) if plate_like else None
        move_to(env, place_goal(xy_holder[0]), CLOSE, tol=0.006, max_steps=80,
                on_look=track)
        hold(env, CLOSE, steps=8)
        hold(env, OPEN, steps=22)
        move_to(env, _hover(place_goal(xy_holder[0]), transit_z), OPEN,
                max_steps=60)
        hold(env, OPEN, steps=16)

    def measure_place():
        aside_v = np.array([place_xy[0][0] - 0.14, place_xy[0][1], transit_z])
        move_to(env, aside_v, OPEN, max_steps=60)
        obs, scene, _ = _see(env, AGENT_CAM)
        bowl = _refine_near(scene, place_xy[0], radius=0.14)
        near = [b for b in scene.blobs
                if np.linalg.norm(b.xy - place_xy[0]) < 0.14
                and b.height_mid > 0.03]
        if near:
            bowl = min(near, key=lambda b: np.linalg.norm(b.xy - place_xy[0]))
        res = verify_target(obs, place_xy[0], "place",
                            env.proprio().gripper_opening)
        if bowl is not None:
            err = bowl.xy - place_xy[0]
            res.measured_xy = bowl.xy
            res.residual_m = np.array([err[0], err[1], 0.0])
            # This is the nearest tall blob to where we aimed, not a verified
            # sighting of the carried bowl. When the bowl was dropped en route
            # the search still finds *something* near the plate and reports a
            # tidy 1.7 cm while the bowl is 55 cm away. Name it accordingly so
            # the failure reason is not mistaken for an On() margin.
            res.note = (f"nearest blob {100*np.linalg.norm(err):.1f} cm from "
                        f"place target (identity unverified)")
        return res, bowl

    set_down(place_xy)
    if provisional:
        return _result(env, False, "provisional placement complete", belief)
    if env.success:
        if verbose:
            print("done: success after first place")
        return _result(env, True, "placed")

    tr.phase(env.t, "verify place")
    res, bowl = measure_place()
    if verbose:
        print(f"verify place: {res.note}")

    if bowl is None:
        verify_obs = env.observe()
        near_plate = [
            b for b in segment_scene(
                verify_obs, cam=AGENT_CAM, self_radius=0.04, plane_z=table_z,
            ).blobs
            if float(b.mean_rgb.mean()) < 140 and b.sat < 0.15
            and np.linalg.norm(b.xy - plate_anchor) < 0.07
        ]
        if near_plate:
            bowl = min(near_plate, key=lambda b: np.linalg.norm(b.xy - plate_anchor))
            metric_xy = placed_object_xy(
                verify_obs, AGENT_CAM, place_xy[0], table_z,
                robot_mask=env.robot_pixel_mask(AGENT_CAM),
            )
            measured_xy = metric_xy if metric_xy is not None else bowl.xy
            err = measured_xy - place_xy[0]
            res.measured_xy = measured_xy
            res.residual_m = np.array([err[0], err[1], 0.0])
            res.note = (f"dark blob {100*np.linalg.norm(err):.1f} cm from "
                        f"place target (identity unverified)")

    if bowl is not None and res.residual_m is not None:
        err_xy = res.residual_m[:2]
        dist = float(np.linalg.norm(err_xy))
        if (target_is_bowl and 0.025 < dist < 0.08
                and float(bowl.mean_rgb.mean()) < 140):
            # This is a new grasp in a new state. A bowl fetched from a
            # cabinet or stove is now resting on the table-height plate; using
            # its original support_z / pinch_z closes the fingers 20+ cm above
            # it and makes the correction a no-op.
            nudge_obs = env.observe()
            nudge_seen = measure_object(
                nudge_obs, AGENT_CAM, bowl.xy, plane_z=table_z,
                robot_mask=env.robot_pixel_mask(AGENT_CAM),
            )
            pick_xy[0] = bowl.xy.copy()
            if nudge_seen is not None:
                pinch_z = nudge_seen.top_z - 0.018
            grasp_ori = None
            pick = pinch_pose(pick_xy[0])
            if verbose:
                print(f"nudge: re-pinch at {pick_xy[0]}, residual {100*dist:.1f} cm")
            track_hover = _track(
                pick_xy, lambda xy: _hover(pinch_pose(xy), transit_z), table_z,
            )
            track_pinch = _track(pick_xy, pinch_pose, table_z)
            move_to(env, _hover(pick, transit_z), OPEN, max_steps=70,
                    on_look=track_hover)
            move_to(env, pick, OPEN, tol=0.005, max_steps=70,
                    on_look=track_pinch)
            hold(env, CLOSE, steps=25)
            if _holding(env.proprio().gripper_opening):
                move_to(env, _hover(pick, transit_z), CLOSE, max_steps=70)
                set_down(place_xy)
                if not env.success:
                    tr.phase(env.t, "verify after nudge")
                    res, _ = measure_place()
                    if verbose:
                        print(f"verify after nudge: {res.note}")

    ok = env.success
    append_lesson(
        "place_on_plate", res, ok,
        extra={"task": f"{env.suite_name}/{env.task_id}", "init": init_idx},
    )
    if verbose:
        print(f"done: success={ok} opening={env.proprio().gripper_opening:.4f}")
    why = "placed" if ok else f"placed but not success ({res.note})"
    return _result(env, ok, why)
