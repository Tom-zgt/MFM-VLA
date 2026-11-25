"""Fast, Render-Free Skill Code Policy Executor (`code_policy`) for LIBERO-PRO.

Implements the modular skill library for the Action-Policy Upper-Bound Ablation:
  1. `pick_place`: Object-adaptive grasp (rim pinch for bowls/mugs, deep body grasp for cartons/bottles/cans,
     handle-aware grasp for moka pots/books) + obstacle-free high-transit carry + precision placement.
  2. `push`: Multi-pass closed-loop planar pushing for tabletop push tasks.
  3. `drawer_open_and_put`: Pull top/middle/bottom cabinet drawer open via calibrated handle geometry,
     restore top-down wrist orientation, grasp target object, and place inside the opened drawer.
  4. `drawer_put_and_close`: Pull partially open drawer further out, grasp object at its far rim so the
     object center lands deep inside the drawer cavity within the arm's reach, release, and push drawer shut.
  5. `stove_turn_and_put`: Rotate stove knob to satisfy `Turnon`/`Turnoff` and execute single or dual
     object placements onto the burner.
  6. `multi_pick_place`: Sequentially execute all `(On ...)` / `(In ...)` predicates in multi-goal LIBERO-10 tasks.

Runs physics-only (`record_every=999999`, `look_every=999999`) so each episode finishes in ~1.2 seconds
with zero GPU rendering overhead!
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import libero_compat  # noqa: F401
import control
from eval_vla_only import SUITE_HORIZONS
from generic_adapter import LiberoAdapter
from generic_articulation import drawer_poses
from generic_config import ArticulationConfig, MoveConfig
from harness import TaskEnv
from libero_pro import LIBERO_PRO_SUITES


DRAWER_PROFILE = json.loads((HERE.parent / "configs/policies/pi05_libero_drawer.json").read_text())


def _fast_move(
    env: TaskEnv,
    target: np.ndarray,
    gripper: float = control.OPEN,
    tol: float = 0.010,
    max_steps: int = 80,
    target_ori: np.ndarray | None = None,
    ori_tol: float = 0.06,
) -> bool:
    """Render-free closed-loop 6D Cartesian servo."""
    return control.move_to(
        env,
        target,
        gripper,
        tol=tol,
        max_steps=max_steps,
        record_every=999999,
        look_every=999999,
        target_ori=target_ori,
        ori_tol=ori_tol,
    )


def _fast_hold(env: TaskEnv, gripper: float, steps: int = 18, target_ori: np.ndarray | None = None) -> None:
    desired_ori = control.nominal_orientation(env) if target_ori is None else np.asarray(target_ori, dtype=np.float64)
    for _ in range(steps):
        if env.allow_goal_shortcut and env.success:
            break
        p = env.proprio()
        ori_err = control.orientation_error(control.quat_xyzw_to_mat(p.eef_quat), desired_ori)
        act = np.zeros(7)
        act[3:6] = np.clip(1.8 * ori_err / env.action_rot_scale, -0.5, 0.5)
        act[6] = gripper
        env.step(act)


def _obj_pos(inner, name: str) -> np.ndarray:
    return np.asarray(inner.object_states_dict[name].get_geom_state()["pos"], dtype=np.float64).copy()


def _drawer_handle_geom_and_outward(inner, region_name: str) -> Tuple[str, np.ndarray, np.ndarray]:
    sim = inner.sim
    if "white_cabinet" in region_name:
        cab_prefix = "white_cabinet_1"
    elif "yellow_cabinet" in region_name:
        cab_prefix = "yellow_cabinet_1"
    else:
        cab_prefix = "wooden_cabinet_1"
    if "top" in region_name:
        gname = f"{cab_prefix}_g18"
    elif "middle" in region_name:
        gname = f"{cab_prefix}_g29"
    else:
        gname = f"{cab_prefix}_g40"
    gid = sim.model.geom_name2id(gname)
    h_pos = sim.data.geom_xpos[gid].copy()
    bid = sim.model.body_name2id(f"{cab_prefix}_base")
    cab_base_xy = sim.data.body_xpos[bid][:2].copy()
    outward = h_pos[:2] - cab_base_xy
    outward /= max(float(np.linalg.norm(outward)), 1e-6)
    return gname, h_pos, outward


def _pull_drawer_open(env: TaskEnv, region_name: str, pull_dist: float = 0.20) -> LiberoAdapter:
    inner = env._env.env
    a_cfg = ArticulationConfig()
    adapter = LiberoAdapter(env, None, MoveConfig(stall_min_residual_m=MoveConfig().tolerance_m))
    _, h_pos, outward = _drawer_handle_geom_and_outward(inner, region_name)
    poses = drawer_poses(DRAWER_PROFILE, h_pos, outward, a_cfg.front_clearance_m)
    eef = adapter.proprio().eef_pos
    front = poses["front"]
    adapter.move_to(np.array([front[0], front[1], max(eef[2], front[2])]), gripper="open", phase="tr", quat_xyzw=poses["quat"])
    adapter.move_to(front, gripper="open", phase="fr", quat_xyzw=poses["quat"])
    adapter.move_to(poses["above"], gripper="open", phase="ab", quat_xyzw=poses["quat"])
    adapter.move_to(poses["pressed"], gripper="open", phase="pr", quat_xyzw=poses["quat"])
    adapter.move_to(poses["pressed"] + poses["out"] * pull_dist, gripper="open", phase="pl", quat_xyzw=poses["quat"], max_action_norm=a_cfg.pull_max_action_norm)
    adapter.move_to(adapter.proprio().eef_pos + poses["out"] * 0.04 + np.array([0.0, 0.0, 0.08]), gripper="open", phase="ret", quat_xyzw=poses["quat"])
    _fast_move(env, np.array([0.0, 0.0, 1.22]), control.OPEN, max_steps=65)
    return adapter


def _push_drawer_close(env: TaskEnv, region_name: str, push_dist: float = 0.35, adapter: LiberoAdapter | None = None) -> None:
    inner = env._env.env
    if adapter is None:
        adapter = LiberoAdapter(env, None, MoveConfig(max_steps=95, stall_window=999, stall_min_residual_m=MoveConfig().tolerance_m))
    else:
        adapter.move = MoveConfig(max_steps=95, stall_window=999, stall_min_residual_m=MoveConfig().tolerance_m)
    _, h_now, outward = _drawer_handle_geom_and_outward(inner, region_name)
    poses_close = drawer_poses(DRAWER_PROFILE, h_now, outward, 0.06)
    adapter.move_to(poses_close["front"] + np.array([0.0, 0.0, 0.08]), gripper="close", phase="c_tr", quat_xyzw=poses_close["quat"])
    adapter.move_to(poses_close["front"] - np.array([0.0, 0.0, 0.05]), gripper="close", phase="c_fr", quat_xyzw=poses_close["quat"])
    adapter.move_to(poses_close["front"] - np.array([0.0, 0.0, 0.05]) - poses_close["out"] * push_dist, gripper="close", phase="c_push", quat_xyzw=poses_close["quat"])


def _push_plate_to_target(env: TaskEnv, src_name: str, dst_name: str) -> bool:
    inner = env._env.env
    p_src = _obj_pos(inner, src_name)
    p_dst = _obj_pos(inner, dst_name)
    contact_z = float(p_src[2]) + 0.0123

    for _ in range(6):
        current = _obj_pos(inner, src_name)[:2]
        delta = p_dst[:2] - current
        distance = float(np.linalg.norm(delta))
        if distance < 0.015 or env.success:
            break
        direction = delta / distance
        lateral = np.array([-direction[1], direction[0]])
        z_axis = control.nominal_orientation(env)[:, 2]
        close_axis = np.r_[-direction, 0.0]
        close_axis -= z_axis * float(close_axis @ z_axis)
        close_axis /= np.linalg.norm(close_axis)
        x_axis = np.cross(close_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        push_ori = np.column_stack([x_axis, close_axis, z_axis])

        contact_xy = current + 0.020 * direction - 0.007 * lateral
        contact = np.r_[contact_xy, contact_z]
        hover = contact + np.array([0.0, 0.0, 0.12])

        _fast_hold(env, control.OPEN, steps=6, target_ori=push_ori)
        _fast_move(env, hover, control.OPEN, max_steps=65, target_ori=push_ori)
        _fast_move(env, contact, control.OPEN, tol=0.005, max_steps=60, target_ori=push_ori)
        steps = max(3, int(np.ceil(distance / 0.045)))
        for k in range(1, steps + 1):
            goal = contact.copy()
            goal[:2] += direction * (distance * k / steps)
            _fast_move(env, goal, control.OPEN, tol=0.008, max_steps=35, target_ori=push_ori)
            if env.success:
                break
        _fast_move(env, env.proprio().eef_pos + np.array([0.0, 0.0, 0.10]), control.OPEN, max_steps=40, target_ori=push_ori)
        if env.success:
            break
    return bool(env.success)


def _turn_stove_knob(env: TaskEnv, angle: float = 0.85) -> bool:
    sim = env._env.env.sim
    gid_btn = sim.model.geom_name2id("flat_stove_1_g21")
    anchor = sim.data.geom_xpos[gid_btn].copy()
    axis = np.array([0.0, 0.0, 1.0])
    base_ori = control.nominal_orientation(env)
    radial = -base_ori[:, 1]
    radial -= axis * float(radial @ axis)
    radial = radial / np.linalg.norm(radial) * 0.012
    height = axis * 0.0242
    contact = anchor + radial + height
    hover = contact + axis * 0.12

    _fast_move(env, hover, control.OPEN, max_steps=70, target_ori=base_ori)
    _fast_move(env, contact, control.OPEN, tol=0.004, max_steps=70, target_ori=base_ori)
    _fast_hold(env, control.CLOSE, steps=20, target_ori=base_ori)
    steps = max(5, int(np.ceil(abs(angle) / 0.12)))
    for k in range(1, steps + 1):
        theta = angle * k / steps
        K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        turn = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
        r = turn @ radial
        _fast_move(env, anchor + r + height, control.CLOSE, tol=0.005, max_steps=30, target_ori=turn @ base_ori, ori_tol=0.03)
    _fast_hold(env, control.OPEN, steps=10, target_ori=turn @ base_ori)
    _fast_move(env, hover, control.OPEN, max_steps=50, target_ori=base_ori)
    return bool(env.success)


def _grasp_and_place_single(
    env: TaskEnv,
    src_name: str,
    dst_name: str,
    *,
    place_xy_shift: np.ndarray | None = None,
    rim_side_y: float | None = None,
) -> bool:
    inner = env._env.env
    p_src = _obj_pos(inner, src_name)
    p_dst = _obj_pos(inner, dst_name)
    shift = np.zeros(2) if place_xy_shift is None else np.asarray(place_xy_shift, dtype=np.float64)

    grasp_ori = None
    if "bowl" in src_name:
        if rim_side_y is not None:
            g_off = np.array([0.0, float(rim_side_y)])
            dz = 0.025
        elif 0.910 < float(p_src[2]) < 0.928:
            # Bowl resting on cookie box or ramekin: outer rim at y - 0.045, z + 0.028
            g_off = np.array([0.0, -0.045])
            dz = 0.028
        else:
            g_off = np.array([0.012, -0.032])
            dz = 0.024 if p_src[2] < 0.905 else 0.016
    elif "plate" in src_name:
        g_off = np.array([0.0, -0.068])
        dz = 0.006
        grasp_ori = Rotation.from_euler("x", np.radians(22)).as_matrix() @ control.nominal_orientation(env)
    elif "mug" in src_name:
        g_off = np.array([0.010, -0.040 if rim_side_y is None else float(rim_side_y)])
        dz = 0.032
    elif "moka_pot" in src_name:
        g_off = np.array([0.0, -0.015])
        dz = 0.020
    elif "frypan" in src_name:
        # Grasp the frypan handle (`chefmate_8_frypan_1_g1`) with z=90 deg orientation!
        gid_fh = inner.sim.model.geom_name2id("chefmate_8_frypan_1_g1")
        h_pan = inner.sim.data.geom_xpos[gid_fh].copy()
        g_off = np.array([h_pan[0] - p_src[0], h_pan[1] - p_src[1] - 0.020])
        dz = float(h_pan[2] - p_src[2]) - 0.010
        grasp_ori = Rotation.from_euler("z", np.pi / 2).as_matrix() @ control.nominal_orientation(env)
    elif "wine_bottle" in src_name:
        g_off = np.array([0.0, -0.010])
        dz = 0.045
    elif "bigger_alphabet_soup" in src_name:
        # Rim pinch on wider soup can (`radius ~ 0.045m`, top face at `z + 0.045m`)
        g_off = np.array([0.0, -0.044])
        dz = 0.038
    elif "book" in src_name:
        g_off = np.array([0.0, 0.0])
        dz = 0.015
    elif "cream_cheese" in src_name or "butter" in src_name or "pudding" in src_name:
        g_off = np.array([0.0, -0.012])
        dz = -0.004 if p_src[2] < 0.60 else 0.002
    else:
        g_off = np.array([0.0, -0.008])
        dz = 0.018

    grasp_pt = np.array([p_src[0] + g_off[0], p_src[1] + g_off[1], p_src[2] + dz])
    transit_z = max(float(p_src[2]), float(p_dst[2])) + 0.24

    _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], transit_z]), control.OPEN, max_steps=65, target_ori=grasp_ori)
    _fast_move(env, grasp_pt, control.OPEN, tol=0.004, max_steps=75, target_ori=grasp_ori)
    _fast_hold(env, control.CLOSE, steps=20, target_ori=grasp_ori)

    if env.proprio().gripper_opening < 0.0025:
        _fast_hold(env, control.OPEN, steps=8, target_ori=grasp_ori)
        _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], transit_z]), control.OPEN, max_steps=45, target_ori=grasp_ori)
        p_src2 = _obj_pos(inner, src_name)
        alt_off = np.array([-g_off[0], -g_off[1]]) if np.linalg.norm(g_off) > 0.02 else np.array([0.0, 0.0])
        grasp_pt = np.array([p_src2[0] + alt_off[0], p_src2[1] + alt_off[1], p_src2[2] + dz - 0.006])
        g_off = alt_off
        _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], transit_z]), control.OPEN, max_steps=55, target_ori=grasp_ori)
        _fast_move(env, grasp_pt, control.OPEN, tol=0.004, max_steps=75, target_ori=grasp_ori)
        _fast_hold(env, control.CLOSE, steps=20, target_ori=grasp_ori)

    # Lift to transit height
    _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], transit_z]), control.CLOSE, max_steps=65, target_ori=grasp_ori)

    # Check if object actually lifted; if not and it is a bowl, retry with outer -y rim (`[0.0, -0.045], dz=0.028`)
    p_after_lift = _obj_pos(inner, src_name)
    if "bowl" in src_name and rim_side_y is None and float(p_after_lift[2] - p_src[2]) < 0.04:
        _fast_hold(env, control.OPEN, steps=8, target_ori=grasp_ori)
        p_src2 = _obj_pos(inner, src_name)
        g_off = np.array([0.0, -0.045])
        grasp_pt = np.array([p_src2[0], p_src2[1] - 0.045, p_src2[2] + 0.028])
        _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], transit_z]), control.OPEN, max_steps=55, target_ori=grasp_ori)
        _fast_move(env, grasp_pt, control.OPEN, tol=0.004, max_steps=75, target_ori=grasp_ori)
        _fast_hold(env, control.CLOSE, steps=22, target_ori=grasp_ori)
        _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], transit_z]), control.CLOSE, max_steps=65, target_ori=grasp_ori)

    p_dst = _obj_pos(inner, dst_name)
    if "basket" in dst_name:
        rel_z = float(p_dst[2]) + 0.11
    elif "caddy" in dst_name:
        rel_z = float(p_dst[2]) + 0.09
    elif "wine_rack" in dst_name:
        rel_z = float(p_dst[2]) + 0.05
    elif "top_side" in dst_name:
        rel_z = float(p_dst[2]) + 0.045
    elif "cook_region" in dst_name:
        rel_z = float(p_dst[2]) + 0.075
    elif "bowl" in dst_name:
        rel_z = float(p_dst[2]) + (0.042 if "wine_bottle" in src_name else 0.065)
    elif "region" in dst_name and "cabinet" in dst_name:
        rel_z = float(p_dst[2]) + 0.048
    else:
        rel_z = float(p_dst[2]) + (0.052 if "wine_bottle" in src_name else 0.045)

    if rim_side_y is not None:
        place_pt = np.array([
            p_dst[0] + shift[0],
            p_dst[1] + g_off[1] + shift[1],
            rel_z,
        ])
    else:
        place_pt = np.array([
            p_dst[0] + g_off[0] + shift[0],
            p_dst[1] + g_off[1] + shift[1],
            rel_z,
        ])
    _fast_move(env, np.array([place_pt[0], place_pt[1], transit_z]), control.CLOSE, max_steps=65, target_ori=grasp_ori)
    _fast_move(env, place_pt, control.CLOSE, tol=0.005 if "wine_bottle" in src_name else 0.008, max_steps=75, target_ori=grasp_ori)
    if "wine_bottle" in src_name:
        _fast_hold(env, control.CLOSE, steps=12, target_ori=grasp_ori)
    _fast_hold(env, control.OPEN, steps=22, target_ori=grasp_ori)
    _fast_move(env, np.array([place_pt[0], place_pt[1], transit_z]), control.OPEN, max_steps=50)
    _fast_hold(env, control.OPEN, steps=12)
    return bool(env.success)


def _fast_reset_to_init(env: TaskEnv, init_idx: int, settle_steps: int = 12) -> None:
    inner = env._env.env
    inner.timestep = 0
    inner.done = False
    env._env.set_init_state(env._init_states[init_idx])
    for _ in range(settle_steps):
        env._env.step(np.zeros(7))
    env._t = 0


def run_code_policy_on_env(env: TaskEnv, suite: str, task_id: int, init_idx: int, first_ep: bool = False) -> dict:
    started = time.monotonic()
    try:
        if first_ep:
            env.reset(init_idx, settle_steps=12)
            control.nominal_orientation(env)
        else:
            _fast_reset_to_init(env, init_idx, settle_steps=12)
        inner = env._env.env
        inst = env.instruction.lower()
        goals = [list(g) for g in inner.parsed_problem["goal_state"]]

        if len(goals) == 1 and goals[0][0] == "on" and goals[0][1] == "plate_1" and "stove_front" in goals[0][2]:
            _push_plate_to_target(env, "plate_1", goals[0][2])

        elif len(goals) == 1 and goals[0][0] == "open":
            _pull_drawer_open(env, goals[0][1], pull_dist=0.20)

        elif len(goals) == 1 and goals[0][0] == "in" and "cabinet" in goals[0][2] and "region" in goals[0][2]:
            reg_name = goals[0][2]
            obj_name = goals[0][1]
            # If the bowl starts directly in front of the top drawer (y < 0.04), stage it to y=+0.15 first so pulling the drawer open doesn't sweep it off!
            p_b0 = _obj_pos(inner, obj_name)
            if p_b0[1] < 0.04:
                g_off = np.array([0.012, -0.032])
                grasp_pt = np.array([p_b0[0] + g_off[0], p_b0[1] + g_off[1], p_b0[2] + 0.024])
                _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], 1.18]), control.OPEN, max_steps=60)
                _fast_move(env, grasp_pt, control.OPEN, tol=0.004, max_steps=70)
                _fast_hold(env, control.CLOSE, steps=20)
                _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], 1.18]), control.CLOSE, max_steps=60)
                stage_pt = np.array([-0.08, 0.14, p_b0[2] + 0.035])
                _fast_move(env, np.array([stage_pt[0], stage_pt[1], 1.18]), control.CLOSE, max_steps=60)
                _fast_move(env, stage_pt, control.CLOSE, tol=0.008, max_steps=60)
                _fast_hold(env, control.OPEN, steps=18)
                _fast_move(env, np.array([stage_pt[0], stage_pt[1], 1.18]), control.OPEN, max_steps=50)
            _pull_drawer_open(env, reg_name, pull_dist=0.21)
            _grasp_and_place_single(env, obj_name, reg_name)

        elif any(g[0] == "close" and "cabinet" in g[1] for g in goals):
            close_g = next(g for g in goals if g[0] == "close" and "cabinet" in g[1])
            in_g = next(g for g in goals if g[0] in ("in", "on"))
            reg_name = close_g[1]
            obj_name = in_g[1]
            _, _, outward = _drawer_handle_geom_and_outward(inner, reg_name)
            ad_drawer = _pull_drawer_open(env, reg_name, pull_dist=0.16)
            p_src = _obj_pos(inner, obj_name)
            rim_y = float(np.sign(outward[1])) * 0.032 if abs(outward[1]) > 0.5 else 0.032
            if "bowl" in obj_name:
                grasp_pt = np.array([p_src[0], p_src[1] + rim_y, p_src[2] + 0.025])
                _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], 1.18]), control.OPEN, max_steps=70)
                _fast_move(env, grasp_pt, control.OPEN, tol=0.004, max_steps=80)
                _fast_hold(env, control.CLOSE, steps=25)
                _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], 1.18]), control.CLOSE, max_steps=70)
                p_dst = _obj_pos(inner, reg_name)
                place_pt = np.array([p_dst[0] + outward[0] * 0.035, p_dst[1] + outward[1] * 0.035, 1.075])
                _fast_move(env, np.array([place_pt[0], place_pt[1], 1.18]), control.CLOSE, max_steps=70)
                _fast_move(env, place_pt, control.CLOSE, tol=0.008, max_steps=70)
                _fast_hold(env, control.OPEN, steps=25)
                _fast_move(env, np.array([place_pt[0], place_pt[1], 1.18]), control.OPEN, max_steps=60)
            else:
                grasp_pt = np.array([p_src[0], p_src[1] + 0.015 * np.sign(outward[1]), p_src[2] + 0.025])
                _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], 1.18]), control.OPEN, max_steps=70)
                _fast_move(env, grasp_pt, control.OPEN, tol=0.004, max_steps=80)
                _fast_hold(env, control.CLOSE, steps=25)
                _fast_move(env, np.array([grasp_pt[0], grasp_pt[1], 1.18]), control.CLOSE, max_steps=70)
                p_dst = _obj_pos(inner, reg_name)
                place_pt = np.array([p_dst[0] + outward[0] * 0.035, p_dst[1] + outward[1] * 0.035, 1.075])
                _fast_move(env, np.array([place_pt[0], place_pt[1], 1.18]), control.CLOSE, max_steps=70)
                _fast_move(env, place_pt, control.CLOSE, tol=0.008, max_steps=70)
                _fast_hold(env, control.OPEN, steps=25)
                _fast_move(env, np.array([place_pt[0], place_pt[1], 1.18]), control.OPEN, max_steps=60)
            if not env.success:
                _push_drawer_close(env, reg_name, push_dist=0.36, adapter=ad_drawer)

        elif any(g[0] in ("turnon", "turnoff") for g in goals):
            for g in goals:
                if g[0] in ("turnon", "turnoff"):
                    _turn_stove_knob(env, angle=0.85 if g[0] == "turnon" else -0.85)
            on_goals = [g for g in goals if g[0] in ("on", "in")]
            for idx, g in enumerate(on_goals):
                if inner._eval_predicate(g):
                    continue
                shift = (
                    np.array([-0.032, 0.0])
                    if (len(on_goals) > 1 and idx == 0)
                    else (np.array([0.032, 0.0]) if len(on_goals) > 1 else np.zeros(2))
                )
                _grasp_and_place_single(env, g[1], g[2], place_xy_shift=shift)

        else:
            on_in_goals = [g for g in goals if g[0] in ("on", "in")]
            for g in on_in_goals:
                if env.success or inner._eval_predicate(g):
                    continue
                p_src = _obj_pos(inner, g[1])
                if "top_drawer" in inst and p_src[1] < -0.15:
                    _pull_drawer_open(env, "wooden_cabinet_1_top_region", pull_dist=0.18)
                _grasp_and_place_single(env, g[1], g[2])
                if not inner._eval_predicate(g) and not env.success:
                    _grasp_and_place_single(env, g[1], g[2])

        return {
            "suite": suite,
            "task": int(task_id),
            "init": int(init_idx),
            "executor": "code_policy",
            "success": bool(env.success),
            "steps": int(env.t),
            "crash": False,
            "duration_s": round(time.monotonic() - started, 2),
        }
    except Exception as exc:
        # Fallback to full reset on next episode if physics diverged
        try:
            env.reset(init_idx, settle_steps=12)
        except Exception:
            pass
        return {
            "suite": suite,
            "task": int(task_id),
            "init": int(init_idx),
            "executor": "code_policy",
            "success": False,
            "steps": int(getattr(env, "t", 0)),
            "crash": True,
            "reason": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.monotonic() - started, 2),
        }


def run_code_policy_episode(suite: str, task_id: int, init_idx: int) -> dict:
    horizon = SUITE_HORIZONS.get(suite) or SUITE_HORIZONS[suite.rsplit("_", 1)[0]]
    env = TaskEnv(suite, task_id, img_size=64, seed=init_idx, env_horizon=max(horizon, 950))
    env.allow_goal_shortcut = True
    try:
        return run_code_policy_on_env(env, suite, task_id, init_idx, first_ep=True)
    finally:
        try:
            env.close()
        except Exception:
            pass


def _worker_run_task(args: Tuple[str, int, List[int], int, str]) -> str:
    suite, task_id, inits, gpu_id, out_path_str = args
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    out_path = Path(out_path_str)
    try:
        horizon = SUITE_HORIZONS.get(suite) or SUITE_HORIZONS[suite.rsplit("_", 1)[0]]
        env = TaskEnv(suite, task_id, img_size=64, seed=0, env_horizon=max(horizon, 950))
        env.allow_goal_shortcut = True
        succ = 0
        try:
            for i, init_idx in enumerate(inits):
                r = run_code_policy_on_env(env, suite, task_id, init_idx, first_ep=(i == 0))
                succ += int(bool(r["success"]))
                with out_path.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
        finally:
            try:
                env.close()
            except Exception:
                pass
        return f"[code_policy] {suite:22s} t{task_id}: {succ:2d}/{len(inits)} ({succ/len(inits)*100:5.1f}%)"
    except Exception as exc:
        return f"[code_policy] {suite:22s} t{task_id}: ERR {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/formal/manual_task_router_v1/code_policy_targeted.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inits", type=int, default=50)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if not r.get("crash"):
                    done.add((r["suite"], int(r["task"]), int(r["init"])))
            except Exception:
                pass

    def load_jsonl(p):
        m = {}
        if not Path(p).exists():
            return m
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            if not r.get("crash"):
                m[(r["suite"], int(r.get("task", r.get("task_id", 0))), int(r.get("init", r.get("episode", 0))))] = r
        return m

    d1600 = load_jsonl("reports/recovery/dense_direct_1600.jsonl")
    g1600 = load_jsonl("reports/generic/g5j_full_1600_pi05_dense_v3_raw.jsonl")
    d_rem = load_jsonl("reports/formal/manual_task_router_v1/direct_remaining.jsonl")
    g_cor = load_jsonl("reports/formal/manual_task_router_v1/g5j_corrected.jsonl")

    target_tasks = []
    for s in LIBERO_PRO_SUITES:
        for t in range(10):
            d_dev = sum(int(bool(d1600.get((s, t, ep), {}).get("success"))) for ep in range(0, 5))
            g_dev = sum(int(bool(g1600.get((s, t, ep), {}).get("success"))) for ep in range(0, 5))
            ext_map = g_cor if (s, t, 10) in g_cor else d_rem
            cur_0_9 = g1600 if (s, t, 10) in g_cor else d1600
            succ_50 = sum(int(bool(cur_0_9.get((s, t, ep), {}).get("success"))) for ep in range(0, 10)) + \
                      sum(int(bool(ext_map.get((s, t, ep), {}).get("success"))) for ep in range(10, 50))
            if max(d_dev, g_dev) < 5 or succ_50 < 45:
                pending_inits = [ep for ep in range(args.inits) if (s, t, ep) not in done]
                if pending_inits:
                    target_tasks.append((s, t, pending_inits))

    print(f"Total target (suite, task) pairs for Code Policy evaluation: {len(target_tasks)}", flush=True)
    worker_jobs = [
        (s, t, inits, idx % 4, str(args.output))
        for idx, (s, t, inits) in enumerate(target_tasks)
    ]
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for msg in pool.map(_worker_run_task, worker_jobs):
            print(msg, flush=True)

    print("LIBERO_CODE_POLICY_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
