"""RoboCasa365 Skill & Navigation Primitives (`G5j-v5/v6 + Code Policy`) for the 6th-Row
Action-Policy Upper-Bound Ablation on `Target-50` (`ep 0..9` = 500 episodes).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
import robosuite.utils.transform_utils as T
import robocasa  # noqa: F401
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FALLBACK_MAP = {
    "strawberry": "kiwi",
    "sausage": "hot_dog",
    "kebabs": "steak",
    "skewers": "steak",
    "bacon": "steak",
    "pork_chop": "steak",
    "pork_loin": "steak",
    "lamb_chop": "steak",
    "chicken_breast": "steak",
    "ham": "steak",
    "salami": "steak",
    "tofu": "cheese",
    "zucchini": "cucumber",
    "celery": "cucumber",
    "asparagus": "cucumber",
    "pickle": "cucumber",
    "chili_pepper": "cucumber",
    "radish": "kiwi",
    "beet": "onion",
    "cabbage": "apple",
    "cauliflower": "apple",
    "brussel_sprout": "kiwi",
    "artichoke": "avocado",
    "ginger": "potato",
    "cherry": "kiwi",
    "raspberry": "kiwi",
    "grapes": "kiwi",
    "dates": "kiwi",
    "apricot": "peach",
    "pomegranate": "apple",
    "cantaloupe": "apple",
    "watermelon": "apple",
    "pineapple": "apple",
    "coconut": "apple",
}
for _cat, _regs in OBJ_CATEGORIES.items():
    if sum(len(m.mjcf_paths) for m in _regs.values()) == 0:
        _source_paths = []
        for _source_cat in (
            FALLBACK_MAP.get(_cat, "apple"),
            "apple",
            "kiwi",
            "cucumber",
            "steak",
        ):
            for _source_meta in OBJ_CATEGORIES.get(_source_cat, {}).values():
                if len(_source_meta.mjcf_paths) > 0:
                    _source_paths = list(_source_meta.mjcf_paths)
                    break
            if _source_paths:
                break
        if _source_paths:
            for _m in _regs.values():
                _m.mjcf_paths = list(_source_paths)


def get_base_pose_2d(adapter) -> Tuple[np.ndarray, float, np.ndarray]:
    sim = adapter._sim
    base_id = sim.model.body_name2id("mobilebase0_base")
    pos = np.asarray(sim.data.body_xpos[base_id], dtype=np.float64)[:2].copy()
    R_b = np.asarray(sim.data.body_xmat[base_id], dtype=np.float64).reshape(3, 3).copy()
    yaw = float(T.mat2euler(R_b)[2])
    return pos, yaw, R_b


def navigate_base_to_pose(
    adapter,
    target_xy: np.ndarray,
    target_yaw: float,
    *,
    max_steps: int = 220,
    pos_tol_m: float = 0.03,
    cos_yaw_tol: float = 0.99,
    backoff_steps: int = 16,
    gripper: str = "hold",
    torso: float = -1.0,
    speed_limit: Optional[float] = None,
) -> dict:
    sim = adapter._sim
    tgt_xy = np.asarray(target_xy, dtype=np.float64)[:2].copy()
    tgt_yaw = float(target_yaw)
    grip_val = (
        (1.0 if adapter.gripper_command > 0 else -1.0)
        if gripper == "hold"
        else (-1.0 if gripper == "open" else 1.0)
    )
    base_limit = (
        float(speed_limit)
        if speed_limit is not None
        else (0.55 if gripper == "close" else 1.0)
    )
    base_accel = (
        min(0.16, 0.30 * base_limit)
        if gripper == "close"
        else 1.0
    )
    prev_base_cmd = np.zeros(3, dtype=np.float64)

    def _smooth_base_cmd(raw_cmd: np.ndarray) -> np.ndarray:
        nonlocal prev_base_cmd
        target = np.clip(np.asarray(raw_cmd, dtype=np.float64), -base_limit, base_limit)
        prev_base_cmd += np.clip(target - prev_base_cmd, -base_accel, base_accel)
        return prev_base_cmd.copy()

    start_t = adapter.t
    pos0, yaw0, _ = get_base_pose_2d(adapter)
    init_dist = float(np.linalg.norm(tgt_xy - pos0))
    need_backoff = init_dist > 0.35 or np.cos(tgt_yaw - yaw0) < 0.90
    b_steps = backoff_steps if need_backoff else 0

    hist: list[np.ndarray] = []
    esc_steps = 0
    esc_p1 = np.zeros(2, dtype=np.float64)
    esc_p2 = np.zeros(2, dtype=np.float64)

    for step in range(max_steps):
        if adapter.success:
            break
        pos, yaw, R_b = get_base_pose_2d(adapter)
        err_w = tgt_xy - pos
        dist = float(np.linalg.norm(err_w))
        cos_err = float(np.cos(tgt_yaw - yaw))
        if dist <= pos_tol_m and cos_err >= cos_yaw_tol:
            break

        if step < b_steps:
            act = np.zeros(12, dtype=np.float64)
            act[6] = grip_val
            act[7:10] = _smooth_base_cmd(np.array([-0.85, 0.0, 0.0]))
            act[10] = torso
            act[11] = 1.0
            adapter.step_12d(act, mode="nav_backoff")
            continue

        if esc_steps > 0:
            w_cmd = esc_p1 if esc_steps > 24 else esc_p2
            esc_steps -= 1
            vb = R_b[:2, :2].T @ w_cmd
            yaw_err = float(np.arctan2(np.sin(tgt_yaw - yaw), np.cos(tgt_yaw - yaw)))
            act = np.zeros(12, dtype=np.float64)
            act[6] = grip_val
            act[7:10] = _smooth_base_cmd(
                np.array([vb[0], vb[1], yaw_err * 2.5], dtype=np.float64)
            )
            act[10] = torso
            act[11] = 1.0
            adapter.step_12d(act, mode="nav_escape")
            continue

        err_b = R_b[:2, :2].T @ err_w
        hist.append(pos.copy())
        if backoff_steps == 0:
            if (
                len(hist) >= 8
                and float(np.linalg.norm(pos - hist[-8])) < 0.012
                and abs(float(err_b[1])) < 0.025
                and cos_err >= cos_yaw_tol
            ):
                break
        elif len(hist) >= 8 and float(np.linalg.norm(pos - hist[-8])) < 0.04 and dist > 0.15:
            push_away = np.zeros(2, dtype=np.float64)
            for i in range(sim.data.ncon):
                c = sim.data.contact[i]
                g1 = sim.model.geom_id2name(c.geom1) or ""
                g2 = sim.model.geom_id2name(c.geom2) or ""
                if ("mobile" in g1 or "robot" in g1 or "mobile" in g2 or "robot" in g2) and "floor" not in g1 and "floor" not in g2:
                    v = pos - c.pos[:2]
                    if np.linalg.norm(v) > 1e-4:
                        push_away += v / np.linalg.norm(v)
            if np.linalg.norm(push_away) > 1e-4:
                push_away = push_away / np.linalg.norm(push_away)
            else:
                push_away = -err_w / (dist + 1e-8)
            perp = np.array([-push_away[1], push_away[0]])
            if np.dot(perp, err_w) < 0:
                perp = -perp
            esc_p1 = 1.0 * push_away
            esc_p2 = 1.0 * perp
            esc_steps = 36
            hist.clear()
            continue
        yaw_err = float(np.arctan2(np.sin(tgt_yaw - yaw), np.cos(tgt_yaw - yaw)))
        act = np.zeros(12, dtype=np.float64)
        act[6] = grip_val
        act[7:10] = _smooth_base_cmd(
            np.array(
                [err_b[0] * 3.5, err_b[1] * 3.5, yaw_err * 2.8],
                dtype=np.float64,
            )
        )
        act[10] = torso
        act[11] = 1.0
        adapter.step_12d(act, mode="nav_drive")

    adapter.move_steps += adapter.t - start_t
    pf, yf, _ = get_base_pose_2d(adapter)
    return {
        "steps": adapter.t - start_t,
        "pos_err_m": round(float(np.linalg.norm(tgt_xy - pf)), 4),
        "cos_yaw_err": round(float(np.cos(tgt_yaw - yf)), 4),
    }


def navigate_base_to_fixture(
    adapter,
    fixture,
    *,
    ref_object: Optional[str] = None,
    gripper: str = "hold",
    torso: float = -1.0,
    speed_limit: Optional[float] = None,
) -> dict:
    inner = adapter.inner
    tgt_pos, tgt_ori = EnvUtils.compute_robot_base_placement_pose(inner, fixture, ref_object=ref_object)
    return navigate_base_to_pose(
        adapter,
        tgt_pos[:2],
        float(tgt_ori[2]),
        gripper=gripper,
        torso=torso,
        speed_limit=speed_limit,
    )


def fixture_region_center_world(
    fixture,
    inner,
    *,
    z_offset: float = 0.03,
    **region_kwargs,
) -> np.ndarray:
    regions = fixture.get_reset_regions(inner, **region_kwargs)
    if not regions:
        raise ValueError(f"No reset region available for fixture {fixture.name}")
    region = max(
        regions.values(),
        key=lambda reg: float(np.prod(np.asarray(reg.get("size", (0.0, 0.0))))),
    )
    local = np.asarray(region["offset"], dtype=np.float64).copy()
    R_fixture = T.euler2mat(np.array([0.0, 0.0, float(fixture.rot)]))
    target = np.asarray(fixture.pos, dtype=np.float64) + R_fixture @ local
    target[2] += z_offset
    return target


def poke_button_l_shaped(adapter, p_btn: np.ndarray) -> dict:
    sim = adapter._sim
    inner = adapter.inner
    torso_val = 1.0 if float(p_btn[2]) > 1.22 else -1.0
    base_pos, _, _ = get_base_pose_2d(adapter)
    base_yaw = float(
        np.arctan2(
            p_btn[1] - base_pos[1],
            p_btn[0] - base_pos[0],
        )
    )
    face = np.array([np.cos(base_yaw), np.sin(base_yaw)])
    navigate_base_to_pose(
        adapter,
        p_btn[:2] - 0.48 * face,
        base_yaw,
        max_steps=120,
        backoff_steps=0,
        pos_tol_m=0.02,
        gripper="close",
        torso=0.0,
    )
    _, _, R_b = get_base_pose_2d(adapter)
    fwd = R_b[:, 0]
    up = np.array([0.0, 0.0, 1.0])
    f1_id = sim.model.geom_name2id("gripper0_right_finger1_pad_collision")
    f2_id = sim.model.geom_name2id("gripper0_right_finger2_pad_collision")
    eef_site_id = sim.model.site_name2id("gripper0_right_grip_site")
    z_des = 0.85 * fwd - 0.52 * up
    z_des = z_des / np.linalg.norm(z_des)
    button_geom = None
    best_pad_distance = float("inf")
    best_pad_delta = np.full(3, np.nan, dtype=np.float64)
    best_blocking_contacts: list[str] = []
    contact_steps = 0
    gripper_geom_ids = {
        sim.model.geom_name2id(name)
        for name in inner.robots[0].gripper["right"].contact_geoms
    }
    for geom_id in range(sim.model.ngeom):
        if np.array_equal(sim.data.geom_xpos[geom_id], p_btn):
            button_geom = sim.model.geom_id2name(geom_id)
            break

    def _observe() -> None:
        nonlocal best_pad_distance, best_pad_delta
        nonlocal best_blocking_contacts, contact_steps
        pad_mid = 0.5 * (
            sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
        )
        delta = pad_mid - p_btn
        distance = float(np.linalg.norm(delta))
        if distance < best_pad_distance:
            best_pad_distance = distance
            best_pad_delta = delta.copy()
            contacts = set()
            for idx in range(sim.data.ncon):
                con = sim.data.contact[idx]
                if con.geom1 in gripper_geom_ids:
                    contacts.add(sim.model.geom_id2name(con.geom2) or "?")
                if con.geom2 in gripper_geom_ids:
                    contacts.add(sim.model.geom_id2name(con.geom1) or "?")
            best_blocking_contacts = sorted(contacts)
        if button_geom is not None and inner.check_contact(
            inner.robots[0].gripper["right"],
            button_geom,
        ):
            contact_steps += 1

    def _servo(target_pad: np.ndarray, steps: int = 40) -> None:
        for _ in range(steps):
            if adapter.success:
                break
            pad_mid = 0.5 * (sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id])
            err = target_pad - pad_mid
            if float(np.linalg.norm(err)) < 0.006:
                break
            R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
            f_vec = sim.data.geom_xpos[f2_id] - sim.data.geom_xpos[f1_id]
            f_dir = f_vec / (np.linalg.norm(f_vec) + 1e-8)
            w_world = 3.5 * np.cross(R_eef[:, 2], z_des) + 2.5 * np.cross(f_dir, up)
            act = np.zeros(12, dtype=np.float64)
            act[:3] = np.clip((R_b.T @ err) * 6.5, -1.0, 1.0)
            act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
            act[6] = 1.0
            act[10] = float(np.clip(6.0 * err[2], -1.0, 1.0))
            act[11] = -1.0
            adapter.step_12d(act, mode="poke")
            _observe()

    adapter.move_to(
        p_btn - 0.23 * fwd - 0.08 * up,
        gripper="close",
        phase="poke_reset",
        tolerance_m=0.03,
        max_steps=70,
        allow_base_assist=True,
    )
    _servo(p_btn - 0.18 * fwd, 50)
    contact_target = p_btn + 0.14 * fwd - 0.065 * up
    _servo(contact_target, 70)
    # High-mounted variants can leave the arm at a Cartesian joint limit. Keep
    # the arm pose fixed and use the holonomic base plus torso as a final
    # closed-loop contact stage; this is still executed through the 12D robot
    # controller and stops immediately on measured button contact.
    for _ in range(55):
        _observe()
        if contact_steps:
            break
        pad_mid = 0.5 * (
            sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
        )
        err = contact_target - pad_mid
        _, _, R_b = get_base_pose_2d(adapter)
        local_xy = R_b[:2, :2].T @ err[:2]
        act = np.zeros(12, dtype=np.float64)
        act[6] = 1.0
        act[7:9] = np.clip(2.5 * local_xy, -0.35, 0.35)
        act[10] = float(np.clip(5.0 * err[2], -1.0, 1.0))
        act[11] = 1.0
        adapter.step_12d(act, mode="poke_base_trim")
    if not contact_steps:
        # The button collision geom protrudes only about 1 mm from the panel on
        # several microwave assets. Reorient nearly normal to the face and make
        # a narrow vertical contact sweep so the finger tip does not stop on
        # the surrounding panel a few millimetres above or below the button.
        z_des = 0.98 * fwd - 0.20 * up
        z_des /= np.linalg.norm(z_des)
        for dz in (-0.035, -0.015, 0.005, 0.025):
            if contact_steps or adapter.success:
                break
            _servo(p_btn - 0.12 * fwd + dz * up, 28)
            _servo(p_btn + 0.16 * fwd + dz * up, 45)
    for _ in range(25):
        if adapter.success:
            break
        act = np.zeros(12, dtype=np.float64)
        act[0] = -1.0
        act[2] = 0.5
        act[6] = 1.0
        act[10] = torso_val
        act[11] = -1.0
        adapter.step_12d(act, mode="poke_retract")
        _observe()
    return {
        "min_pad_distance_m": round(best_pad_distance, 4),
        "best_pad_delta_xyz_m": np.round(best_pad_delta, 4).tolist(),
        "blocking_contacts_at_best": best_blocking_contacts,
        "contact_steps": int(contact_steps),
        "button_geom": button_geom,
    }


def turn_sink_water_on_and_align(
    adapter,
    sink,
    *,
    target_obj: Optional[str] = None,
) -> dict:
    """Physically aim the spout and push the faucet handle until water flows."""
    sim = adapter._sim
    inner = adapter.inner
    prefix = sink.naming_prefix
    handle_jid = sim.model.joint_name2id(f"{prefix}handle_joint")
    spout_jid = sim.model.joint_name2id(f"{prefix}spout_joint")
    handle_gid = sim.model.geom_name2id(f"{prefix}handle_main")
    spout_gid = sim.model.geom_name2id(f"{prefix}spout_main")
    water_sid = sim.model.site_name2id(f"{prefix}water")
    f1_id = sim.model.geom_name2id("gripper0_right_finger1_pad_collision")
    f2_id = sim.model.geom_name2id("gripper0_right_finger2_pad_collision")
    eef_site_id = sim.model.site_name2id("gripper0_right_grip_site")
    up = np.array([0.0, 0.0, 1.0])
    handle_min_distance = float("inf")
    handle_contact_steps = 0

    navigate_base_to_fixture(
        adapter,
        sink,
        gripper="close",
        torso=-1.0,
    )

    def _servo_pad(
        target: np.ndarray,
        *,
        steps: int,
        mode: str,
        grip: float = 1.0,
    ) -> None:
        nonlocal handle_min_distance, handle_contact_steps
        for _ in range(steps):
            _, _, R_b = get_base_pose_2d(adapter)
            fwd = R_b[:, 0]
            left = R_b[:, 1]
            pad_mid = 0.5 * (
                sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
            )
            err = np.asarray(target, dtype=np.float64) - pad_mid
            if float(np.linalg.norm(err)) < 0.006:
                break
            R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
            finger_vec = sim.data.geom_xpos[f2_id] - sim.data.geom_xpos[f1_id]
            finger_dir = finger_vec / (np.linalg.norm(finger_vec) + 1e-8)
            if np.dot(finger_dir, left) < 0:
                finger_dir = -finger_dir
            z_des = 0.90 * fwd - 0.43 * up
            z_des /= np.linalg.norm(z_des)
            w_world = (
                3.5 * np.cross(R_eef[:, 2], z_des)
                + 2.0 * np.cross(finger_dir, left)
            )
            act = np.zeros(12, dtype=np.float64)
            act[:3] = np.clip((R_b.T @ err) * 6.0, -0.75, 0.75)
            act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
            act[6] = grip
            act[10] = -1.0
            act[11] = -1.0
            adapter.step_12d(act, mode=mode)
            tip = sim.data.xanchor[handle_jid] + 1.65 * (
                sim.data.geom_xpos[handle_gid]
                - sim.data.xanchor[handle_jid]
            )
            handle_min_distance = min(
                handle_min_distance,
                float(
                    np.linalg.norm(
                        0.5
                        * (
                            sim.data.geom_xpos[f1_id]
                            + sim.data.geom_xpos[f2_id]
                        )
                        - tip
                    )
                ),
            )
            if inner.check_contact(
                inner.robots[0].gripper["right"],
                sim.model.geom_id2name(handle_gid),
            ):
                handle_contact_steps += 1

    p_handle = sim.data.geom_xpos[handle_gid].copy()
    base_pos, _, _ = get_base_pose_2d(adapter)
    base_yaw = float(
        np.arctan2(
            p_handle[1] - base_pos[1],
            p_handle[0] - base_pos[0],
        )
    )
    face_dir = np.array(
        [np.cos(base_yaw), np.sin(base_yaw)],
        dtype=np.float64,
    )
    navigate_base_to_pose(
        adapter,
        p_handle[:2] - 0.49 * face_dir,
        base_yaw,
        max_steps=40,
        backoff_steps=0,
        pos_tol_m=0.02,
        gripper="open",
        torso=-1.0,
    )

    for attempt in range(2):
        state = sink.get_handle_state(inner)
        if bool(state.get("water_on", False)):
            break
        p_handle = sim.data.geom_xpos[handle_gid].copy()
        anchor = sim.data.xanchor[handle_jid].copy()
        axis = sim.data.xaxis[handle_jid].copy()
        p_contact = anchor + 1.65 * (p_handle - anchor)
        sign_dir = 1.0 if attempt == 0 else -1.0
        tangent = sign_dir * np.cross(axis, p_contact - anchor)
        tangent /= np.linalg.norm(tangent) + 1e-8
        _, _, R_b = get_base_pose_2d(adapter)
        _servo_pad(
            p_contact - 0.08 * tangent + np.array([0.0, 0.0, 0.06]),
            steps=35,
            mode="faucet_reset",
            grip=1.0,
        )
        _servo_pad(
            p_contact - 0.06 * tangent + np.array([0.0, 0.0, 0.004]),
            steps=35,
            mode="faucet_pre",
            grip=1.0,
        )
        for _ in range(35):
            if bool(sink.get_handle_state(inner).get("water_on", False)):
                break
            p_handle = sim.data.geom_xpos[handle_gid].copy()
            anchor = sim.data.xanchor[handle_jid].copy()
            p_contact = anchor + 1.65 * (p_handle - anchor)
            tangent = sign_dir * np.cross(
                sim.data.xaxis[handle_jid],
                p_contact - anchor,
            )
            tangent /= np.linalg.norm(tangent) + 1e-8
            _servo_pad(
                p_contact + 0.05 * tangent,
                steps=2,
                mode="faucet_on",
                grip=1.0,
            )
        for _ in range(35):
            if bool(sink.get_handle_state(inner).get("water_on", False)):
                break
            p_handle = sim.data.geom_xpos[handle_gid].copy()
            anchor = sim.data.xanchor[handle_jid].copy()
            p_contact = anchor + 1.65 * (p_handle - anchor)
            tangent = sign_dir * np.cross(
                sim.data.xaxis[handle_jid],
                p_contact - anchor,
            )
            tangent /= np.linalg.norm(tangent) + 1e-8
            _, _, R_b = get_base_pose_2d(adapter)
            base_cmd = R_b[:2, :2].T @ tangent[:2]
            act = np.zeros(12, dtype=np.float64)
            act[6] = 1.0
            act[7:9] = np.clip(0.45 * base_cmd, -0.45, 0.45)
            act[10] = -1.0
            act[11] = 1.0
            adapter.step_12d(act, mode="faucet_on_base_push")

    # Align only after the valve is open. This preserves the proven handle
    # approach and prevents a long spout sweep from leaving the arm at a joint
    # limit before the higher-priority on/off contact.
    if target_obj is not None and target_obj in inner.obj_body_id:
        obj_radius = float(inner.objects[target_obj].horizontal_radius)
        align_tol = min(0.07, 0.65 * obj_radius)
        qadr = sim.model.jnt_qposadr[spout_jid]
        q_lo, q_hi = map(float, sim.model.jnt_range[spout_jid])
        for _ in range(8):
            target_xy = sim.data.body_xpos[
                inner.obj_body_id[target_obj]
            ][:2].copy()
            water_xy = sim.data.site_xpos[water_sid][:2].copy()
            before = float(np.linalg.norm(target_xy - water_xy))
            if before < align_tol:
                break

            # Privileged model-based direction estimate. The temporary
            # kinematic probes are restored before any scored controller step;
            # all actual motion still comes from PandaOmron 12D actions.
            q0 = float(sim.data.qpos[qadr])
            eps = 0.025
            qm = max(q_lo, q0 - eps)
            qp = min(q_hi, q0 + eps)
            sim.data.qpos[qadr] = qm
            sim.forward()
            wm = sim.data.site_xpos[water_sid][:2].copy()
            sim.data.qpos[qadr] = qp
            sim.forward()
            wp = sim.data.site_xpos[water_sid][:2].copy()
            sim.data.qpos[qadr] = q0
            sim.forward()
            dwater_dq = (wp - wm) / max(qp - qm, 1e-6)
            gradient = float(np.dot(water_xy - target_xy, dwater_dq))
            q_dir = -1.0 if gradient > 0 else 1.0

            p_spout = sim.data.geom_xpos[spout_gid].copy()
            tangent = np.cross(
                sim.data.xaxis[spout_jid],
                p_spout - sim.data.xanchor[spout_jid],
            )
            tangent *= q_dir / (np.linalg.norm(tangent) + 1e-8)
            _servo_pad(
                p_spout - 0.09 * tangent,
                steps=35,
                mode="spout_model_pre",
            )
            for _ in range(24):
                p_spout = sim.data.geom_xpos[spout_gid].copy()
                _servo_pad(
                    p_spout + 0.035 * tangent,
                    steps=2,
                    mode="spout_model_align",
                )
                target_xy = sim.data.body_xpos[
                    inner.obj_body_id[target_obj]
                ][:2]
                if float(
                    np.linalg.norm(
                        target_xy - sim.data.site_xpos[water_sid][:2]
                    )
                ) < align_tol:
                    break

    _, _, R_b = get_base_pose_2d(adapter)
    pad_mid = 0.5 * (
        sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
    )
    _servo_pad(
        pad_mid - 0.18 * R_b[:, 0] + 0.10 * up,
        steps=24,
        mode="faucet_retract",
        grip=-1.0,
    )
    state = sink.get_handle_state(inner)
    water_dist = None
    aligned = None
    if target_obj is not None and target_obj in inner.obj_body_id:
        water_dist = float(
            np.linalg.norm(
                sim.data.body_xpos[inner.obj_body_id[target_obj]][:2]
                - sim.data.site_xpos[water_sid][:2]
            )
        )
        aligned = bool(sink.check_obj_under_water(inner, target_obj))
    return {
        "water_on": bool(state.get("water_on", False)),
        "handle_q": round(float(state.get("handle_joint", 0.0)), 4),
        "spout_q": round(float(state.get("spout_joint", 0.0)), 4),
        "water_distance_m": (
            None if water_dist is None else round(water_dist, 4)
        ),
        "target_under_water": aligned,
        "handle_min_distance_m": round(handle_min_distance, 4),
        "handle_contact_steps": int(handle_contact_steps),
    }


def press_toaster_lever(adapter, toaster, slot_pair: int) -> None:
    sim = adapter._sim
    _, _, R_b = get_base_pose_2d(adapter)
    left = R_b[:, 1]
    up = np.array([0.0, 0.0, 1.0])
    f1_id = sim.model.geom_name2id("gripper0_right_finger1_pad_collision")
    f2_id = sim.model.geom_name2id("gripper0_right_finger2_pad_collision")
    eef_site_id = sim.model.site_name2id("gripper0_right_grip_site")
    joint_name = toaster._joint_names[f"lever_{slot_pair}"]
    jid = sim.model.joint_name2id(joint_name)
    body_id = sim.model.jnt_bodyid[jid]
    lever_geoms = [
        geom_id
        for geom_id in range(sim.model.ngeom)
        if sim.model.geom_bodyid[geom_id] == body_id
        and "lever_handle" in (sim.model.geom_id2name(geom_id) or "")
    ]
    if not lever_geoms:
        raise ValueError(f"No lever handle geom found for {joint_name}")
    lever_geom = lever_geoms[0]
    axis = sim.data.xaxis[jid].copy()
    z_front = 0.95 * R_b[:, 0] - 0.31 * up
    z_front = z_front / np.linalg.norm(z_front)

    def _servo(target: np.ndarray, steps: int) -> None:
        for _ in range(steps):
            pad_mid = 0.5 * (
                sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
            )
            err = target - pad_mid
            if float(np.linalg.norm(err)) < 0.005:
                break
            R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
            finger_vec = sim.data.geom_xpos[f2_id] - sim.data.geom_xpos[f1_id]
            finger_dir = finger_vec / (np.linalg.norm(finger_vec) + 1e-8)
            if np.dot(finger_dir, left) < 0:
                finger_dir = -finger_dir
            w_world = (
                3.5 * np.cross(R_eef[:, 2], z_front)
                + 2.0 * np.cross(finger_dir, left)
            )
            act = np.zeros(12, dtype=np.float64)
            act[:3] = np.clip((R_b.T @ err) * 6.5, -1.0, 1.0)
            act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
            act[6] = 1.0
            act[11] = -1.0
            adapter.step_12d(act, mode="toaster_lever")

    p_lever = sim.data.geom_xpos[lever_geom].copy()
    fwd = R_b[:, 0]
    _servo(p_lever - 0.14 * fwd, 45)
    _servo(p_lever + 0.008 * fwd, 35)
    for _ in range(70):
        if toaster.get_state(
            adapter.inner,
            slot_pair=slot_pair,
        )["turned_on"]:
            break
        p_lever = sim.data.geom_xpos[lever_geom].copy()
        _servo(p_lever + 0.025 * axis, 1)
    p_lever = sim.data.geom_xpos[lever_geom].copy()
    _servo(p_lever - 0.14 * fwd, 30)


def open_slide_drawer(
    adapter,
    drawer,
    pull_wide: bool = False,
    align_obj_key: Optional[str] = None,
) -> None:
    sim = adapter._sim
    inner = adapter.inner
    up = np.array([0.0, 0.0, 1.0])
    f1_id = sim.model.geom_name2id("gripper0_right_finger1_pad_collision")
    f2_id = sim.model.geom_name2id("gripper0_right_finger2_pad_collision")
    eef_site_id = sim.model.site_name2id("gripper0_right_grip_site")
    joint_name = drawer.door_joint_names[0]
    jid = sim.model.joint_name2id(joint_name)
    root_body = sim.model.jnt_bodyid[jid]

    def _is_descendant(body_id: int) -> bool:
        while body_id > 0:
            if body_id == root_body:
                return True
            body_id = sim.model.body_parentid[body_id]
        return False

    pos0, yaw0, R_b = get_base_pose_2d(adapter)
    fwd = R_b[:, 0]
    left = R_b[:, 1]
    door_val = float(list(drawer.get_door_state(env=inner).values())[0])

    # If drawer is closed (door_val < 0.45), grasp horizontal handle with vertical finger opening (`up`)
    if door_val < 0.45:
        handle_geoms = [
            geom_id
            for geom_id in range(sim.model.ngeom)
            if _is_descendant(sim.model.geom_bodyid[geom_id])
            and sim.model.geom_contype[geom_id] > 0
            and "door_handle" in (sim.model.geom_id2name(geom_id) or "")
        ]
        if not handle_geoms:
            handle_geoms = [
                geom_id
                for geom_id in range(sim.model.ngeom)
                if _is_descendant(sim.model.geom_bodyid[geom_id])
                and "door_handle" in (sim.model.geom_id2name(geom_id) or "")
            ]
        handle_geom = min(
            handle_geoms,
            key=lambda g: float(np.dot(sim.data.geom_xpos[g][:2] - pos0, fwd[:2])),
        )
        p_h = sim.data.geom_xpos[handle_geom].copy()
        d_fwd = float(np.dot(p_h[:2] - pos0, fwd[:2]))
        d_left = float(np.dot(p_h[:2] - pos0, left[:2]))
        if d_fwd < 0.64 and abs(d_left) > 0.03:
            navigate_base_to_pose(
                adapter,
                pos0 + (d_fwd - 0.64) * fwd[:2],
                yaw0,
                max_steps=25,
                backoff_steps=0,
                pos_tol_m=0.02,
                gripper="open",
            )
        pos1, yaw1, R_b = get_base_pose_2d(adapter)
        d_fwd = float(np.dot(p_h[:2] - pos1, fwd[:2]))
        d_left = float(np.dot(p_h[:2] - pos1, left[:2]))
        navigate_base_to_pose(
            adapter,
            pos1 + (d_fwd - 0.62) * fwd[:2] + d_left * left[:2],
            yaw1,
            max_steps=32,
            backoff_steps=0,
            pos_tol_m=0.02,
            gripper="open",
        )
        _, _, R_b = get_base_pose_2d(adapter)
        fwd = R_b[:, 0]
        left = R_b[:, 1]
        z_front = 0.96 * fwd - 0.28 * up
        z_front = z_front / np.linalg.norm(z_front)

        def _servo_h(target: np.ndarray, steps: int, grip: float) -> None:
            for _ in range(steps):
                pad_mid = 0.5 * (
                    sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
                )
                err = target - pad_mid
                if float(np.linalg.norm(err)) < 0.005 and steps > 15:
                    break
                R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
                finger_vec = sim.data.geom_xpos[f2_id] - sim.data.geom_xpos[f1_id]
                finger_dir = finger_vec / (np.linalg.norm(finger_vec) + 1e-8)
                if np.dot(finger_dir, up) < 0:
                    finger_dir = -finger_dir
                w_world = (
                    3.5 * np.cross(R_eef[:, 2], z_front)
                    + 2.5 * np.cross(finger_dir, up)
                )
                act = np.zeros(12, dtype=np.float64)
                act[:3] = np.clip((R_b.T @ err) * 6.5, -1.0, 1.0)
                act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
                act[6] = grip
                act[10] = -1.0
                act[11] = -1.0
                adapter.step_12d(act, mode="open_drawer")

        p_h = sim.data.geom_xpos[handle_geom].copy()
        _servo_h(p_h - 0.16 * fwd, 55, -1.0)
        _servo_h(p_h + 0.012 * fwd, 40, -1.0)
        _servo_h(p_h + 0.012 * fwd, 18, 1.0)
        for _ in range(40):
            p_h_now = sim.data.geom_xpos[handle_geom].copy()
            _servo_h(p_h_now - 0.12 * fwd, 1, 1.0)
        for _ in range(40):
            if not pull_wide and adapter.success:
                break
            act = np.zeros(12, dtype=np.float64)
            act[6] = 1.0
            act[7] = -0.70
            act[10] = -1.0
            act[11] = 1.0
            adapter.step_12d(act, mode="open_drawer_base_pull")
        for _ in range(12):
            act = np.zeros(12, dtype=np.float64)
            act[6] = -1.0
            act[10] = -1.0
            act[11] = -1.0
            adapter.step_12d(act, mode="open_drawer_release")
        pad_mid = 0.5 * (sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id])
        _servo_h(pad_mid - 0.10 * fwd + 0.22 * up, 30, -1.0)

    # If caller needs the drawer pulled extra wide (door_state >= 1.40) so interior objects clear the counter overhang
    door_val = float(list(drawer.get_door_state(env=inner).values())[0])
    if pull_wide and door_val < 1.38:
        door_geoms = [
            g for g in range(sim.model.ngeom)
            if _is_descendant(sim.model.geom_bodyid[g])
            and sim.model.geom_contype[g] > 0
            and "door_g1" in (sim.model.geom_id2name(g) or "")
        ]
        if door_geoms:
            d_geom = door_geoms[0]
            p_door = sim.data.geom_xpos[d_geom].copy()
            pos0, yaw0, R_b = get_base_pose_2d(adapter)
            fwd = R_b[:, 0]
            left = R_b[:, 1]
            d_fwd = float(np.dot(p_door[:2] - pos0, fwd[:2]))
            d_left = float(np.dot(p_door[:2] - pos0, left[:2]))
            if d_fwd < 0.52 and abs(d_left) > 0.04:
                navigate_base_to_pose(
                    adapter,
                    pos0 + (d_fwd - 0.52) * fwd[:2],
                    yaw0,
                    max_steps=35,
                    backoff_steps=0,
                    pos_tol_m=0.02,
                    gripper="open",
                )
            pos1, yaw1, R_b = get_base_pose_2d(adapter)
            d_fwd = float(np.dot(p_door[:2] - pos1, fwd[:2]))
            d_left = float(np.dot(p_door[:2] - pos1, left[:2]))
            navigate_base_to_pose(
                adapter,
                pos1 + (d_fwd - 0.52) * fwd[:2] + d_left * left[:2],
                yaw1,
                max_steps=40,
                backoff_steps=0,
                pos_tol_m=0.02,
                gripper="open",
            )
            _, _, R_b = get_base_pose_2d(adapter)
            fwd = R_b[:, 0]
            left = R_b[:, 1]

            def _servo_hook(target: np.ndarray, steps: int, grip: float) -> None:
                for _ in range(steps):
                    pad_mid = 0.5 * (
                        sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
                    )
                    err = target - pad_mid
                    if float(np.linalg.norm(err)) < 0.005 and steps > 15:
                        break
                    R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
                    f_vec = sim.data.geom_xpos[f2_id] - sim.data.geom_xpos[f1_id]
                    f_dir = f_vec / (np.linalg.norm(f_vec) + 1e-8)
                    if np.dot(f_dir, fwd) < 0:
                        f_dir = -f_dir
                    w_world = (
                        3.5 * np.cross(R_eef[:, 2], -up)
                        + 2.5 * np.cross(f_dir, fwd)
                    )
                    act = np.zeros(12, dtype=np.float64)
                    act[:3] = np.clip((R_b.T @ err) * 6.5, -1.0, 1.0)
                    act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
                    act[6] = grip
                    act[10] = -1.0
                    act[11] = -1.0
                    adapter.step_12d(act, mode="hook_drawer")

            p_door = sim.data.geom_xpos[d_geom].copy()
            pull_m = 0.16
            if align_obj_key is not None and align_obj_key in inner.obj_body_id:
                p_obj_now = sim.data.body_xpos[inner.obj_body_id[align_obj_key]].copy()
                obj_depth = abs(float(np.dot(p_obj_now[:2] - p_door[:2], fwd[:2])))
                pull_m = float(np.clip(obj_depth - 0.10, 0.12, 0.21))
            _servo_hook(p_door + 0.05 * fwd + np.array([0, 0, 0.22]), 40, 1.0)
            _servo_hook(p_door + 0.05 * fwd + np.array([0, 0, 0.018]), 40, 1.0)
            _servo_hook(p_door - pull_m * fwd + np.array([0, 0, 0.018]), 42, 1.0)
            _servo_hook(p_door - pull_m * fwd + np.array([0, 0, 0.22]), 30, -1.0)


def open_fixture_doors(adapter, fixture) -> None:
    """Physically pull closed swing doors open using their handles."""
    sim = adapter._sim
    up = np.array([0.0, 0.0, 1.0])
    f1_id = sim.model.geom_name2id("gripper0_right_finger1_pad_collision")
    f2_id = sim.model.geom_name2id("gripper0_right_finger2_pad_collision")
    eef_site_id = sim.model.site_name2id("gripper0_right_grip_site")
    navigate_base_to_fixture(
        adapter,
        fixture,
        gripper="open",
        torso=1.0,
    )

    j_names = getattr(fixture, "_fridge_door_joint_names", []) or getattr(
        fixture, "door_joint_names", []
    )
    for jn in j_names:
        jid = sim.model.joint_name2id(jn)
        if int(sim.model.jnt_type[jid]) != 3:
            continue
        qadr = sim.model.jnt_qposadr[jid]
        joint_range = sim.model.jnt_range[jid]
        open_sign = (
            1.0
            if abs(float(joint_range[1])) >= abs(float(joint_range[0]))
            else -1.0
        )
        if open_sign * float(sim.data.qpos[qadr]) > 0.65:
            continue
        root_body = sim.model.jnt_bodyid[jid]

        def _is_descendant(body_id: int) -> bool:
            while body_id > 0:
                if body_id == root_body:
                    return True
                body_id = sim.model.body_parentid[body_id]
            return False

        handle_geoms = [
            geom_id
            for geom_id in range(sim.model.ngeom)
            if _is_descendant(sim.model.geom_bodyid[geom_id])
            and (
                "door_handle_main" in (sim.model.geom_id2name(geom_id) or "")
                or "door_handle_reg_main" in (sim.model.geom_id2name(geom_id) or "")
            )
        ]
        if not handle_geoms:
            continue
        handle_geom = handle_geoms[0]

        def _servo(
            target: np.ndarray,
            steps: int,
            grip: float,
            torso: float,
        ) -> None:
            _, _, R_b = get_base_pose_2d(adapter)
            fwd = R_b[:, 0]
            left = R_b[:, 1]
            z_front = 0.92 * fwd - 0.38 * up
            z_front = z_front / np.linalg.norm(z_front)
            for _ in range(steps):
                pad_mid = 0.5 * (
                    sim.data.geom_xpos[f1_id]
                    + sim.data.geom_xpos[f2_id]
                )
                err = target - pad_mid
                if float(np.linalg.norm(err)) < 0.006:
                    break
                R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
                finger_vec = (
                    sim.data.geom_xpos[f2_id]
                    - sim.data.geom_xpos[f1_id]
                )
                finger_dir = finger_vec / (
                    np.linalg.norm(finger_vec) + 1e-8
                )
                if np.dot(finger_dir, left) < 0:
                    finger_dir = -finger_dir
                w_world = (
                    3.5 * np.cross(R_eef[:, 2], z_front)
                    + 2.0 * np.cross(finger_dir, left)
                )
                act = np.zeros(12, dtype=np.float64)
                act[:3] = np.clip((R_b.T @ err) * 6.5, -0.8, 0.8)
                act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
                act[6] = grip
                act[10] = torso
                act[11] = -1.0
                adapter.step_12d(act, mode="open_swing_door")

        for stage, pulls in enumerate((10, 14)):
            _, yaw, R_b = get_base_pose_2d(adapter)
            fwd = R_b[:, 0]
            p_handle = sim.data.geom_xpos[handle_geom].copy()
            torso_val = 1.0 if p_handle[2] > 1.22 else -1.0
            _servo(
                p_handle - 0.18 * fwd,
                40,
                -1.0,
                torso_val,
            )
            _servo(
                p_handle + 0.012 * fwd,
                35,
                -1.0,
                torso_val,
            )
            _servo(
                p_handle + 0.012 * fwd,
                16,
                1.0,
                torso_val,
            )
            for _ in range(pulls):
                p_handle = sim.data.geom_xpos[handle_geom].copy()
                tangent = open_sign * np.cross(
                    sim.data.xaxis[jid],
                    p_handle - sim.data.xanchor[jid],
                )
                tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
                _servo(
                    p_handle + 0.08 * tangent + 0.01 * fwd,
                    5,
                    1.0,
                    torso_val,
                )
            if open_sign * float(sim.data.qpos[qadr]) > 0.65:
                break
            pad_mid = 0.5 * (
                sim.data.geom_xpos[f1_id]
                + sim.data.geom_xpos[f2_id]
            )
            _servo(
                pad_mid - 0.15 * fwd,
                20,
                -1.0,
                torso_val,
            )
            if stage == 0:
                pos, yaw, R_b = get_base_pose_2d(adapter)
                navigate_base_to_pose(
                    adapter,
                    pos - 0.28 * R_b[:, 0][:2],
                    yaw,
                    max_steps=30,
                    backoff_steps=0,
                    pos_tol_m=0.02,
                    gripper="open",
                    torso=torso_val,
                )

        pad_mid = 0.5 * (
            sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
        )
        _, _, R_b = get_base_pose_2d(adapter)
        _servo(
            pad_mid - 0.18 * R_b[:, 0],
            20,
            -1.0,
            1.0,
        )


def close_fixture_doors(adapter, fixture) -> None:
    sim = adapter._sim
    _, yaw0, R_b = get_base_pose_2d(adapter)
    fwd = R_b[:, 0]
    left = R_b[:, 1]
    up = np.array([0.0, 0.0, 1.0])
    f1_id = sim.model.geom_name2id("gripper0_right_finger1_pad_collision")
    f2_id = sim.model.geom_name2id("gripper0_right_finger2_pad_collision")
    eef_site_id = sim.model.site_name2id("gripper0_right_grip_site")
    z_front = 0.90 * fwd - 0.43 * up
    z_front = z_front / np.linalg.norm(z_front)

    def _servo_d(tgt: np.ndarray, steps: int = 35, torso: float = 1.0) -> None:
        for _ in range(steps):
            if adapter.success:
                break
            curr = 0.5 * (sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id])
            err = tgt - curr
            if float(np.linalg.norm(err)) < 0.008:
                break
            R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
            w_world = 3.5 * np.cross(R_eef[:, 2], z_front)
            act = np.zeros(12, dtype=np.float64)
            act[:3] = np.clip((R_b.T @ err) * 6.5, -1.0, 1.0)
            act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
            act[6] = 1.0
            act[10] = torso
            act[11] = -1.0
            adapter.step_12d(act, mode="close_door")

    j_names = getattr(fixture, "door_joint_names", []) or getattr(
        fixture, "_fridge_door_joint_names", []
    )
    for jn in reversed(j_names):
        if adapter.success:
            break
        jid = sim.model.joint_name2id(jn)
        qadr = sim.model.jnt_qposadr[jid]
        qpos = float(sim.data.qpos[qadr])
        if abs(qpos) < 0.003:
            continue
        body_id = sim.model.jnt_bodyid[jid]
        door_geoms = [
            g for g in range(sim.model.ngeom)
            if sim.model.geom_contype[g] > 0 and (
                sim.model.geom_bodyid[g] == body_id or sim.model.body_parentid[sim.model.geom_bodyid[g]] == body_id
            )
        ]
        if not door_geoms:
            continue
        anchor = sim.data.xanchor[jid].copy()
        axis = sim.data.xaxis[jid].copy()
        best_g = max(door_geoms, key=lambda g: np.linalg.norm(sim.data.geom_xpos[g][:2] - anchor[:2]))
        p_door = sim.data.geom_xpos[best_g].copy()
        p_door[2] = float(np.clip(p_door[2], 0.95, 1.48))
        torso_val = 1.0 if p_door[2] > 1.25 else -1.0

        sign_close = -1.0 if qpos > 0 else 1.0
        v_close = sign_close * np.cross(axis, p_door - anchor)
        v_close = v_close / (np.linalg.norm(v_close) + 1e-8)
        curr_pad = 0.5 * (
            sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
        )
        d_fwd_curr = float(np.dot(curr_pad[:2] - p_door[:2], fwd[:2]))
        _servo_d(
            np.array([curr_pad[0], curr_pad[1], p_door[2]])
            - max(0.0, d_fwd_curr + 0.22) * fwd,
            30,
            torso=torso_val,
        )

        p_pre = p_door - 0.10 * v_close
        for _ in range(30):
            pos_now, _, _ = get_base_pose_2d(adapter)
            fwd_err = float(np.dot(p_pre[:2] - pos_now, fwd[:2])) - 0.54
            lat_err = float(np.dot(p_pre[:2] - pos_now, left[:2]))
            if abs(fwd_err) < 0.03 and abs(lat_err) < 0.03:
                break
            act = np.zeros(12, dtype=np.float64)
            act[6] = 1.0
            act[7] = np.clip(fwd_err * 3.5, -0.8, 0.8)
            act[8] = np.clip(lat_err * 3.5, -0.8, 0.8)
            act[10] = torso_val
            act[11] = 1.0
            adapter.step_12d(act, mode="close_door_align")

        pos_now, _, _ = get_base_pose_2d(adapter)
        if abs(float(np.dot(p_pre[:2] - pos_now, left[:2]))) > 0.14:
            for _ in range(18):
                act = np.zeros(12, dtype=np.float64)
                act[6] = 1.0
                act[7] = -0.55
                act[10] = torso_val
                act[11] = 1.0
                adapter.step_12d(act, mode="close_door_corner_backoff")
            for _ in range(30):
                pos_now, _, _ = get_base_pose_2d(adapter)
                lat_err = float(np.dot(p_pre[:2] - pos_now, left[:2]))
                if abs(lat_err) < 0.03:
                    break
                act = np.zeros(12, dtype=np.float64)
                act[6] = 1.0
                act[8] = np.clip(lat_err * 3.5, -0.55, 0.55)
                act[10] = torso_val
                act[11] = 1.0
                adapter.step_12d(act, mode="close_door_corner_slide")

        p_door = sim.data.geom_xpos[best_g].copy()
        p_door[2] = float(np.clip(p_door[2], 0.95, 1.48))
        v_close = sign_close * np.cross(
            sim.data.xaxis[jid], p_door - sim.data.xanchor[jid]
        )
        v_close = v_close / (np.linalg.norm(v_close) + 1e-8)

        _servo_d(p_door - 0.16 * fwd - 0.10 * v_close, 30, torso=torso_val)
        _servo_d(p_door - 0.08 * v_close, 25, torso=torso_val)
        for _ in range(55):
            if adapter.success or abs(float(sim.data.qpos[qadr])) < 0.002:
                break
            p_curr = sim.data.geom_xpos[best_g].copy()
            p_curr[2] = p_door[2]
            v_curr = sign_close * np.cross(
                sim.data.xaxis[jid], p_curr - sim.data.xanchor[jid]
            )
            v_curr = v_curr / (np.linalg.norm(v_curr) + 1e-8)
            _servo_d(p_curr + 0.10 * v_curr + 0.05 * fwd, 3, torso=torso_val)

        for _ in range(2):
            if abs(float(sim.data.qpos[qadr])) < 0.004:
                break
            p_curr = sim.data.geom_xpos[best_g].copy()
            p_curr[2] = p_door[2]
            v_curr = sign_close * np.cross(
                sim.data.xaxis[jid], p_curr - sim.data.xanchor[jid]
            )
            v_curr = v_curr / (np.linalg.norm(v_curr) + 1e-8)
            _servo_d(
                p_curr - 0.14 * fwd - 0.08 * v_curr,
                22,
                torso=torso_val,
            )
            for _ in range(18):
                pos_now, _, _ = get_base_pose_2d(adapter)
                fwd_err = float(np.dot(p_curr[:2] - pos_now, fwd[:2])) - 0.50
                lat_err = float(np.dot(p_curr[:2] - pos_now, left[:2]))
                if abs(fwd_err) < 0.03 and abs(lat_err) < 0.03:
                    break
                act = np.zeros(12, dtype=np.float64)
                act[6] = 1.0
                act[7] = np.clip(fwd_err * 3.5, -0.55, 0.55)
                act[8] = np.clip(lat_err * 3.5, -0.55, 0.55)
                act[10] = torso_val
                act[11] = 1.0
                adapter.step_12d(act, mode="close_door_realign")
            _servo_d(p_curr - 0.06 * v_curr, 20, torso=torso_val)
            for _ in range(50):
                if adapter.success or abs(float(sim.data.qpos[qadr])) < 0.002:
                    break
                p_curr = sim.data.geom_xpos[best_g].copy()
                p_curr[2] = p_door[2]
                v_curr = sign_close * np.cross(
                    sim.data.xaxis[jid], p_curr - sim.data.xanchor[jid]
                )
                v_curr = v_curr / (np.linalg.norm(v_curr) + 1e-8)
                _servo_d(
                    p_curr + 0.10 * v_curr + 0.08 * fwd,
                    3,
                    torso=torso_val,
                )

        if abs(float(sim.data.qpos[qadr])) > 0.05:
            p_curr = sim.data.geom_xpos[best_g].copy()
            p_curr[2] = p_door[2]
            pos_now, _, _ = get_base_pose_2d(adapter)
            yaw_face = float(
                np.arctan2(
                    p_curr[1] - pos_now[1],
                    p_curr[0] - pos_now[0],
                )
            )
            navigate_base_to_pose(
                adapter,
                pos_now,
                yaw_face,
                max_steps=28,
                backoff_steps=0,
                pos_tol_m=0.03,
                gripper="close",
                torso=torso_val,
            )
            _, _, R_b = get_base_pose_2d(adapter)
            fwd = R_b[:, 0]
            left = R_b[:, 1]
            z_front = 0.90 * fwd - 0.43 * up
            z_front = z_front / np.linalg.norm(z_front)
            p_curr = sim.data.geom_xpos[best_g].copy()
            p_curr[2] = p_door[2]
            v_curr = sign_close * np.cross(
                sim.data.xaxis[jid], p_curr - sim.data.xanchor[jid]
            )
            v_curr = v_curr / (np.linalg.norm(v_curr) + 1e-8)
            _servo_d(
                p_curr - 0.18 * fwd - 0.08 * v_curr,
                35,
                torso=torso_val,
            )
            _servo_d(p_curr - 0.04 * v_curr, 30, torso=torso_val)
            for _ in range(70):
                if abs(float(sim.data.qpos[qadr])) < 0.004:
                    break
                p_curr = sim.data.geom_xpos[best_g].copy()
                p_curr[2] = p_door[2]
                v_curr = sign_close * np.cross(
                    sim.data.xaxis[jid], p_curr - sim.data.xanchor[jid]
                )
                v_curr = v_curr / (np.linalg.norm(v_curr) + 1e-8)
                _servo_d(
                    p_curr + 0.14 * v_curr,
                    3,
                    torso=torso_val,
                )

        curr_pad = 0.5 * (
            sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
        )
        _servo_d(curr_pad - 0.24 * fwd, 25, torso=torso_val)


def close_microwave_and_start(adapter, microwave) -> dict:
    """Close a microwave with a joint-feedback fallback, then press start."""
    sim = adapter._sim
    inner = adapter.inner
    close_fixture_doors(adapter, microwave)

    jn = microwave.door_joint_names[0]
    jid = sim.model.joint_name2id(jn)
    qadr = sim.model.jnt_qposadr[jid]
    handle_gid = sim.model.geom_name2id(microwave.handle_name)
    f1_id = sim.model.geom_name2id("gripper0_right_finger1_pad_collision")
    f2_id = sim.model.geom_name2id("gripper0_right_finger2_pad_collision")
    eef_site_id = sim.model.site_name2id("gripper0_right_grip_site")
    up = np.array([0.0, 0.0, 1.0])

    def _servo(target: np.ndarray, steps: int) -> None:
        for _ in range(steps):
            _, _, R_b = get_base_pose_2d(adapter)
            fwd = R_b[:, 0]
            left = R_b[:, 1]
            pad_mid = 0.5 * (
                sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
            )
            err = np.asarray(target, dtype=np.float64) - pad_mid
            if float(np.linalg.norm(err)) < 0.007:
                break
            R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
            finger_vec = sim.data.geom_xpos[f2_id] - sim.data.geom_xpos[f1_id]
            finger_dir = finger_vec / (np.linalg.norm(finger_vec) + 1e-8)
            if np.dot(finger_dir, left) < 0:
                finger_dir = -finger_dir
            z_des = 0.88 * fwd - 0.47 * up
            z_des /= np.linalg.norm(z_des)
            w_world = (
                3.5 * np.cross(R_eef[:, 2], z_des)
                + 2.0 * np.cross(finger_dir, left)
            )
            act = np.zeros(12, dtype=np.float64)
            act[:3] = np.clip((R_b.T @ err) * 6.0, -0.8, 0.8)
            act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
            act[6] = 1.0
            act[10] = 1.0
            act[11] = -1.0
            adapter.step_12d(act, mode="microwave_close_fallback")

    if abs(float(sim.data.qpos[qadr])) > 0.005:
        p_handle = sim.data.geom_xpos[handle_gid].copy()
        base_pos, _, _ = get_base_pose_2d(adapter)
        yaw_face = float(
            np.arctan2(
                p_handle[1] - base_pos[1],
                p_handle[0] - base_pos[0],
            )
        )
        face = np.array([np.cos(yaw_face), np.sin(yaw_face)])
        navigate_base_to_pose(
            adapter,
            p_handle[:2] - 0.50 * face,
            yaw_face,
            max_steps=55,
            backoff_steps=0,
            pos_tol_m=0.025,
            gripper="close",
            torso=1.0,
        )
        _, _, R_b = get_base_pose_2d(adapter)
        fwd = R_b[:, 0]
        p_handle = sim.data.geom_xpos[handle_gid].copy()
        tangent = np.cross(
            sim.data.xaxis[jid],
            p_handle - sim.data.xanchor[jid],
        )
        tangent /= np.linalg.norm(tangent) + 1e-8
        _servo(p_handle - 0.16 * fwd - 0.08 * tangent, 45)
        _servo(p_handle - 0.02 * tangent, 35)
        for _ in range(80):
            if abs(float(sim.data.qpos[qadr])) < 0.004:
                break
            p_handle = sim.data.geom_xpos[handle_gid].copy()
            tangent = np.cross(
                sim.data.xaxis[jid],
                p_handle - sim.data.xanchor[jid],
            )
            tangent /= np.linalg.norm(tangent) + 1e-8
            _servo(p_handle + 0.10 * tangent, 2)

        # If the arm reaches a joint limit, preserve contact and let the
        # omnidirectional base provide the remaining tangential displacement.
        for _ in range(180):
            if abs(float(sim.data.qpos[qadr])) < 0.004:
                break
            p_handle = sim.data.geom_xpos[handle_gid].copy()
            tangent = np.cross(
                sim.data.xaxis[jid],
                p_handle - sim.data.xanchor[jid],
            )
            tangent /= np.linalg.norm(tangent) + 1e-8
            _, _, R_b = get_base_pose_2d(adapter)
            base_cmd = R_b[:2, :2].T @ tangent[:2]
            act = np.zeros(12, dtype=np.float64)
            act[6] = 1.0
            act[7:9] = np.clip(0.75 * base_cmd, -0.75, 0.75)
            act[10] = 1.0
            act[11] = 1.0
            adapter.step_12d(act, mode="microwave_close_base_push")

    navigate_base_to_fixture(
        adapter,
        microwave,
        gripper="close",
        torso=1.0,
    )
    start_gid = sim.model.geom_name2id(
        f"{microwave.naming_prefix}start_button"
    )
    button_trace = poke_button_l_shaped(
        adapter,
        sim.data.geom_xpos[start_gid].copy(),
    )
    if (
        bool(microwave.get_state()["turned_on"])
        and not microwave.is_closed(inner)
    ):
        # Button contact can rebound a nearly closed door by a few millimetres.
        # Re-seat it without reopening past the microwave's power-off threshold.
        close_fixture_doors(adapter, microwave)
    return {
        "door_closed": bool(microwave.is_closed(inner)),
        "door_q": round(float(sim.data.qpos[qadr]), 4),
        "microwave_on": bool(microwave.get_state()["turned_on"]),
        "button_trace": button_trace,
    }


def find_best_grasp_target(sim, inner, obj_key: str, R_b: np.ndarray, is_rim: bool = False):
    bid = inner.obj_body_id[obj_key]
    p_obj = sim.data.body_xpos[bid].copy()
    fwd = R_b[:, 0]
    left = R_b[:, 1]
    if is_rim and obj_key in inner.objects:
        obj = inner.objects[obj_key]
        hr = float(getattr(obj, "horizontal_radius", 0.09))
        p_rim = p_obj - 0.80 * hr * fwd
        try:
            quat = T.convert_quat(
                sim.data.body_xquat[bid],
                to="xyzw",
            )
            bbox = np.asarray(
                obj.get_bbox_points(trans=p_obj, rot=quat),
                dtype=np.float64,
            )
            near_proj = float(
                np.min((bbox[:, :2] - p_obj[:2]) @ fwd[:2])
            )
            p_rim[:2] = (
                p_obj[:2] + 0.84 * near_proj * fwd[:2]
            )
            p_rim[2] = float(np.max(bbox[:, 2]) - 0.018)
        except Exception:
            p_rim[2] += 0.04
        return p_rim, fwd.copy(), False
    bbox_name = f"{obj_key}_reg_bbox"
    if bbox_name not in sim.model.geom_names:
        return p_obj, left, (p_obj[2] > 1.08)
    gid = sim.model.geom_name2id(bbox_name)
    sz = sim.model.geom_size[gid]
    R_o = sim.data.geom_xmat[gid].reshape(3, 3)
    if min(sz[0], sz[1]) > 0.039:
        def _belongs_to_object(body_id: int) -> bool:
            while body_id > 0:
                if body_id == bid:
                    return True
                body_id = sim.model.body_parentid[body_id]
            return False

        cands = [
            g for g in range(sim.model.ngeom)
            if _belongs_to_object(sim.model.geom_bodyid[g])
            and sim.model.geom_contype[g] > 0
            and min(sim.model.geom_size[g][0], sim.model.geom_size[g][1]) < 0.038
        ]
        if cands:
            if max(sz[0], sz[1]) > 1.5 * min(sz[0], sz[1]):
                best_g = max(
                    cands,
                    key=lambda g: float(
                        np.linalg.norm(
                            sim.data.geom_xpos[g][:2] - p_obj[:2]
                        )
                    ),
                )
            else:
                best_g = max(cands, key=lambda g: sim.data.geom_xpos[g][2])
            p_g = sim.data.geom_xpos[best_g].copy()
            vec = p_g[:2] - p_obj[:2]
            if np.linalg.norm(vec) > 0.015:
                rad = np.array([vec[0], vec[1], 0.0]) / np.linalg.norm(vec)
                perp = np.array([-rad[1], rad[0], 0.0])
                return p_g - np.array([0, 0, 0.012]), perp, (p_obj[2] > 1.08)
            return p_g, left, True
    # Compute horizontal projection span of all 3 local bbox axes so rotated utensils
    # (e.g. knives/spatulas whose long axis is local Z rotated into XY) align fingers
    # perpendicular to their true horizontal long axis.
    horiz_spans = [
        float(sz[i]) * float(np.linalg.norm(R_o[:2, i]))
        for i in range(3)
    ]
    long_idx = int(np.argmax(horiz_spans))
    long_xy = R_o[:2, long_idx]
    if float(np.linalg.norm(long_xy)) > 1e-5:
        long_dir = long_xy / float(np.linalg.norm(long_xy))
        narrow_dir = np.array([-long_dir[1], long_dir[0], 0.0], dtype=np.float64)
    else:
        col = R_o[:, 0 if sz[0] <= sz[1] else 1]
        narrow_dir = np.array([col[0], col[1], 0.0]) / (np.linalg.norm(col[:2]) + 1e-8)
    return p_obj, narrow_dir, (p_obj[2] > 1.08)


def turn_stove_knob(adapter, stove, knob_loc: str, mode: str = "off") -> bool:
    """Physically grasp and rotate a stove knob (`mode` in {'on', 'off'})."""
    sim = adapter._sim
    inner = adapter.inner
    navigate_base_to_fixture(adapter, stove, gripper="open", torso=-1.0)
    jn = f"{stove.naming_prefix}knob_{knob_loc}_joint"
    if jn not in sim.model.joint_names:
        return False
    jid = sim.model.joint_name2id(jn)
    knob_gid = sim.model.geom_name2id(f"{stove.naming_prefix}knob_{knob_loc}_main")
    f1_id = sim.model.geom_name2id("gripper0_right_finger1_pad_collision")
    f2_id = sim.model.geom_name2id("gripper0_right_finger2_pad_collision")
    eef_site_id = sim.model.site_name2id("gripper0_right_grip_site")
    up = np.array([0.0, 0.0, 1.0])
    p_k = sim.data.geom_xpos[knob_gid].copy()
    pos0, yaw0, R_b = get_base_pose_2d(adapter)
    fwd, left = R_b[:, 0], R_b[:, 1]
    d_fwd = float(np.dot(p_k[:2] - pos0, fwd[:2]))
    d_left = float(np.dot(p_k[:2] - pos0, left[:2]))
    navigate_base_to_pose(
        adapter,
        pos0 + (d_fwd - 0.50) * fwd[:2] + d_left * left[:2],
        yaw0,
        max_steps=30,
        backoff_steps=0,
        pos_tol_m=0.02,
        gripper="open",
    )
    _, _, R_b = get_base_pose_2d(adapter)
    axis = sim.data.xaxis[jid].copy()
    z_des = 0.85 * (-axis / (np.linalg.norm(axis) + 1e-8)) - 0.52 * up
    if z_des[2] > -0.1:
        z_des = -up.copy()
    z_des /= np.linalg.norm(z_des)

    def _done() -> bool:
        on_now = bool(stove.is_burner_on(env=inner, burner_loc=knob_loc))
        return on_now if mode == "on" else (not on_now)

    def _servo_knob(tgt: np.ndarray, steps: int = 40, grip: float = -1.0, rot_rate: float = 0.0) -> None:
        for _ in range(steps):
            if _done() and rot_rate != 0.0:
                break
            pad = 0.5 * (sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id])
            err = tgt - pad
            R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
            w_world = 3.5 * np.cross(R_eef[:, 2], z_des) + rot_rate * axis
            act = np.zeros(12, dtype=np.float64)
            act[:3] = np.clip((R_b.T @ err) * 6.5, -1.0, 1.0)
            act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
            act[6] = grip
            act[10] = -1.0
            act[11] = -1.0
            adapter.step_12d(act, mode="stove_knob")

    p_k = sim.data.geom_xpos[knob_gid].copy()
    _servo_knob(p_k - 0.12 * z_des, 40, grip=-1.0)
    _servo_knob(p_k + 0.008 * z_des, 35, grip=-1.0)
    _servo_knob(p_k + 0.008 * z_des, 16, grip=1.0)
    first_sign = 3.0 if mode == "on" else -3.0
    _servo_knob(p_k + 0.008 * z_des, 48, grip=1.0, rot_rate=first_sign)
    if not _done():
        _servo_knob(p_k + 0.008 * z_des, 55, grip=1.0, rot_rate=-first_sign)
    _servo_knob(p_k + 0.008 * z_des, 10, grip=-1.0)
    _servo_knob(p_k - 0.14 * z_des + np.array([0, 0, 0.08]), 25, grip=-1.0)
    return _done()


def rinse_sink_basin_skill(adapter, sink) -> dict:
    """Turn on sink water and sweep the spout across left, center, and right."""
    sim = adapter._sim
    inner = adapter.inner
    faucet_info = turn_sink_water_on_and_align(adapter, sink)
    prefix = sink.naming_prefix
    spout_gid = sim.model.geom_name2id(f"{prefix}spout_main")
    spout_jid = sim.model.joint_name2id(f"{prefix}spout_joint")
    f1_id = sim.model.geom_name2id("gripper0_right_finger1_pad_collision")
    f2_id = sim.model.geom_name2id("gripper0_right_finger2_pad_collision")
    eef_site_id = sim.model.site_name2id("gripper0_right_grip_site")
    up = np.array([0.0, 0.0, 1.0])

    def _servo_sp(target: np.ndarray, steps: int = 25) -> None:
        for _ in range(steps):
            if adapter.success:
                break
            _, _, R_b = get_base_pose_2d(adapter)
            fwd, left = R_b[:, 0], R_b[:, 1]
            pad_mid = 0.5 * (sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id])
            err = target - pad_mid
            if float(np.linalg.norm(err)) < 0.006:
                break
            R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
            z_des = 0.90 * fwd - 0.43 * up
            z_des /= np.linalg.norm(z_des)
            w_world = 3.5 * np.cross(R_eef[:, 2], z_des)
            act = np.zeros(12, dtype=np.float64)
            act[:3] = np.clip((R_b.T @ err) * 6.0, -0.8, 0.8)
            act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
            act[6] = 1.0
            act[10] = -1.0
            act[11] = -1.0
            adapter.step_12d(act, mode="spout_sweep")

    for direction in (1.0, -1.0):
        if adapter.success:
            break
        p_sp = sim.data.geom_xpos[spout_gid].copy()
        tangent = direction * np.cross(
            sim.data.xaxis[spout_jid],
            p_sp - sim.data.xanchor[spout_jid],
        )
        tangent /= np.linalg.norm(tangent) + 1e-8
        _servo_sp(p_sp - 0.09 * tangent + np.array([0, 0, 0.08]), 25)
        _servo_sp(p_sp - 0.09 * tangent, 25)
        for _ in range(35):
            if adapter.success:
                break
            p_sp = sim.data.geom_xpos[spout_gid].copy()
            tangent = direction * np.cross(
                sim.data.xaxis[spout_jid],
                p_sp - sim.data.xanchor[spout_jid],
            )
            tangent /= np.linalg.norm(tangent) + 1e-8
            _servo_sp(p_sp + 0.05 * tangent, 2)
    return faucet_info


def grasp_and_place_smart(
    adapter,
    obj_key: str,
    dst_xyz: np.ndarray,
    *,
    release_z: float = 0.04,
    is_rim: bool = False,
    force_top_grasp: bool = False,
    force_front_grasp: bool = False,
    carry_speed_limit: float = 0.55,
    arm_speed_limit: float = 1.0,
    lift_clearance_m: float = 0.22,
    src_fixture=None,
    src_ref_obj: Optional[str] = None,
    dst_fixture=None,
    dst_ref_obj: Optional[str] = None,
    release_after_place: bool = True,
    use_center_grasp: bool = False,
    grasp_z_offset: float = -0.005,
    slide_in_place: bool = False,
    standoff_override: Optional[float] = None,
    lock_base_on_grasp: bool = False,
    lock_base_on_place: bool = False,
) -> bool:
    sim = adapter._sim
    inner = adapter.inner
    if obj_key not in inner.obj_body_id:
        return False
    bid = inner.obj_body_id[obj_key]
    if src_fixture is not None:
        navigate_base_to_fixture(
            adapter,
            src_fixture,
            ref_object=src_ref_obj or obj_key,
            gripper="open",
            torso=-1.0,
        )
    pos0, yaw0, R_b = get_base_pose_2d(adapter)
    fwd = R_b[:, 0]
    left = R_b[:, 1]
    up = np.array([0.0, 0.0, 1.0])
    p_grasp, narrow_dir, is_high = find_best_grasp_target(sim, inner, obj_key, R_b, is_rim=is_rim)
    if use_center_grasp:
        obj_cands = [
            g for g in range(sim.model.ngeom)
            if sim.model.geom_bodyid[g] == bid and sim.model.geom_contype[g] > 0
        ]
        if obj_cands and float(sim.data.body_xpos[bid][2]) < 0.80:
            best_g = max(obj_cands, key=lambda g: float(sim.data.geom_xpos[g][2]))
            p_grasp = 0.4 * sim.data.body_xpos[bid].copy() + 0.6 * sim.data.geom_xpos[best_g].copy()
        else:
            p_grasp = sim.data.body_xpos[bid].copy()
    if force_top_grasp:
        is_high = False
    elif force_front_grasp:
        is_high = True
    p_obj = sim.data.body_xpos[bid].copy()
    grasp_offset = p_grasp - p_obj
    f1_id = sim.model.geom_name2id("gripper0_right_finger1_pad_collision")
    f2_id = sim.model.geom_name2id("gripper0_right_finger2_pad_collision")
    eef_site_id = sim.model.site_name2id("gripper0_right_grip_site")

    def _retract_until_far(torso: float) -> None:
        for _ in range(28):
            if OU.gripper_obj_far(inner, obj_name=obj_key):
                break
            act = np.zeros(12, dtype=np.float64)
            act[0] = -0.8
            act[2] = 0.8
            act[6] = -1.0
            act[10] = torso
            act[11] = -1.0
            adapter.step_12d(act, mode="release_retract")

    standoff = (
        float(standoff_override)
        if standoff_override is not None
        else (0.56 if is_high else 0.50)
    )
    d_fwd = float(np.dot(p_grasp[:2] - pos0, fwd[:2]))
    d_left = float(np.dot(p_grasp[:2] - pos0, left[:2]))
    if not lock_base_on_grasp:
        navigate_base_to_pose(adapter, pos0 + (d_fwd - standoff) * fwd[:2] + d_left * left[:2], yaw0, max_steps=65, backoff_steps=0, pos_tol_m=0.02, gripper="open")
    pos_aligned, _, R_b = get_base_pose_2d(adapter)
    fwd = R_b[:, 0]
    left = R_b[:, 1]
    lateral_residual = float(np.dot(p_grasp[:2] - pos_aligned, left[:2]))
    if src_fixture is not None and is_high and abs(lateral_residual) > 0.06:
        yaw_to_obj = float(
            np.arctan2(
                p_grasp[1] - pos_aligned[1],
                p_grasp[0] - pos_aligned[0],
            )
        )
        navigate_base_to_pose(
            adapter,
            pos_aligned,
            yaw_to_obj,
            max_steps=22,
            backoff_steps=0,
            pos_tol_m=0.03,
            gripper="open",
            torso=-1.0,
        )
        _, _, R_b = get_base_pose_2d(adapter)
        fwd = R_b[:, 0]
        left = R_b[:, 1]
        p_grasp, narrow_dir, is_high = find_best_grasp_target(
            sim,
            inner,
            obj_key,
            R_b,
            is_rim=is_rim,
        )
        if use_center_grasp:
            p_grasp = sim.data.body_xpos[bid].copy()
        if force_top_grasp:
            is_high = False
        elif force_front_grasp:
            is_high = True
        p_obj = sim.data.body_xpos[bid].copy()
        grasp_offset = p_grasp - p_obj

    if is_high:
        torso_val = (
            1.0
            if p_obj[2] > 1.25
            else (-1.0 if p_obj[2] < 1.05 else 0.0)
        )
        torso_dst = 1.0 if dst_xyz[2] > 1.22 else -1.0
        z_front = 0.92 * fwd - 0.38 * up
        z_front = z_front / np.linalg.norm(z_front)
        front_finger = narrow_dir if force_front_grasp else left

        def _servo_f(tgt: np.ndarray, steps: int = 45, grip: float = -1.0, torso: float = torso_val) -> None:
            for _ in range(steps):
                if adapter.success:
                    break
                curr = 0.5 * (sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id])
                err = tgt - curr
                if float(np.linalg.norm(err)) < 0.006:
                    break
                R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
                f_vec = sim.data.geom_xpos[f2_id] - sim.data.geom_xpos[f1_id]
                f_dir = f_vec / (np.linalg.norm(f_vec) + 1e-8)
                if np.dot(f_dir, front_finger) < 0:
                    f_dir = -f_dir
                w_world = (
                    3.5 * np.cross(R_eef[:, 2], z_front)
                    + 2.0 * np.cross(f_dir, front_finger)
                )
                act = np.zeros(12, dtype=np.float64)
                act[:3] = np.clip((R_b.T @ err) * 6.0, -0.7, 0.7)
                act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
                act[6] = grip
                act[10] = torso
                act[11] = -1.0
                adapter.step_12d(act, mode="front_grasp")

        def _servo_obj_f(
            obj_target: np.ndarray,
            steps: int,
            grip: float,
            torso: float,
        ) -> None:
            for _ in range(steps):
                obj_err = obj_target - sim.data.body_xpos[bid]
                if float(np.linalg.norm(obj_err)) < 0.012:
                    break
                pad_mid = 0.5 * (
                    sim.data.geom_xpos[f1_id]
                    + sim.data.geom_xpos[f2_id]
                )
                _servo_f(
                    pad_mid + obj_err,
                    1,
                    grip=grip,
                    torso=torso,
                )

        _servo_f(p_grasp - 0.22 * fwd + np.array([0, 0, 0.01]), 45, grip=-1.0, torso=torso_val)
        _servo_f(p_grasp + 0.012 * fwd, 45, grip=-1.0, torso=torso_val)
        for _ in range(16):
            act = np.zeros(12, dtype=np.float64)
            act[6] = 1.0
            act[10] = torso_val
            act[11] = -1.0
            adapter.step_12d(act, mode="front_grasp_close")
        p_out = p_grasp - 0.28 * fwd + np.array([0, 0, 0.03])
        _servo_f(p_out, 45, grip=1.0, torso=torso_val)
        for _ in range(14):
            act = np.zeros(12, dtype=np.float64)
            act[6] = 1.0
            act[7] = -0.55
            act[10] = torso_val
            act[11] = 1.0
            adapter.step_12d(act, mode="front_grasp_backoff")
        p_out = 0.5 * (
            sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
        )
        _servo_f(np.array([p_out[0], p_out[1], min(1.10, float(dst_xyz[2]) + 0.18)]), 45, grip=1.0, torso=-1.0)

        if dst_fixture is not None:
            navigate_base_to_fixture(
                adapter,
                dst_fixture,
                ref_object=dst_ref_obj,
                gripper="close",
                torso=-1.0,
                speed_limit=carry_speed_limit,
            )
            _, _, R_b = get_base_pose_2d(adapter)
            fwd = R_b[:, 0]
            left = R_b[:, 1]
            z_front = 0.92 * fwd - 0.38 * up
            z_front = z_front / np.linalg.norm(z_front)
        pos_now, yaw_now, R_b = get_base_pose_2d(adapter)
        d_fwd_d = float(np.dot(dst_xyz[:2] - pos_now, fwd[:2]))
        d_left_d = float(np.dot(dst_xyz[:2] - pos_now, left[:2]))
        if not lock_base_on_place:
            navigate_base_to_pose(adapter, pos_now + (d_fwd_d - 0.52) * fwd[:2] + d_left_d * left[:2], yaw_now, max_steps=65, backoff_steps=0, pos_tol_m=0.02, gripper="close", torso=torso_dst, speed_limit=carry_speed_limit)
        _, _, R_b = get_base_pose_2d(adapter)
        fwd = R_b[:, 0]
        left = R_b[:, 1]
        z_front = 0.92 * fwd - 0.38 * up
        z_front = z_front / np.linalg.norm(z_front)

        if float(dst_xyz[2]) > 1.18 or slide_in_place:
            _servo_obj_f(
                dst_xyz - 0.18 * fwd + np.array([0, 0, release_z + 0.015]),
                45,
                grip=1.0,
                torso=torso_dst,
            )
        else:
            _servo_obj_f(
                dst_xyz + np.array([0, 0, 0.14]),
                45,
                grip=1.0,
                torso=torso_dst,
            )
        _servo_obj_f(
            dst_xyz + np.array([0, 0, release_z]),
            45,
            grip=1.0,
            torso=torso_dst,
        )
        if not release_after_place:
            return bool(OU.check_obj_grasped(inner, obj_key))
        pad_mid = 0.5 * (
            sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id]
        )
        for _ in range(10):
            act = np.zeros(12, dtype=np.float64)
            act[6] = -1.0
            act[10] = torso_dst
            act[11] = -1.0
            adapter.step_12d(act, mode="front_place_open")
        _servo_f(
            pad_mid - 0.24 * fwd + np.array([0, 0, 0.22]),
            30,
            grip=-1.0,
            torso=torso_dst,
        )
        _retract_until_far(torso_dst)
        return True

    for _ in range(40):
        pad_mid = 0.5 * (sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id])
        err = (p_grasp + np.array([0, 0, 0.16])) - pad_mid
        R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
        f_vec = sim.data.geom_xpos[f2_id] - sim.data.geom_xpos[f1_id]
        f_dir = f_vec / (np.linalg.norm(f_vec) + 1e-8)
        if np.dot(f_dir, narrow_dir) < 0:
            f_dir = -f_dir
        w_world = 3.5 * np.cross(R_eef[:, 2], -up) + 2.5 * np.cross(f_dir, narrow_dir)
        act = np.zeros(12, dtype=np.float64)
        act[:3] = np.clip(
            (R_b.T @ err) * 6.5,
            -arm_speed_limit,
            arm_speed_limit,
        )
        act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
        act[6] = -1.0
        act[10] = -1.0
        act[11] = -1.0
        adapter.step_12d(act, mode="hover")

    for _ in range(52):
        pad_mid = 0.5 * (sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id])
        err = (p_grasp + np.array([0, 0, grasp_z_offset])) - pad_mid
        if float(np.linalg.norm(err)) < 0.004:
            break
        R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
        f_vec = sim.data.geom_xpos[f2_id] - sim.data.geom_xpos[f1_id]
        f_dir = f_vec / (np.linalg.norm(f_vec) + 1e-8)
        if np.dot(f_dir, narrow_dir) < 0:
            f_dir = -f_dir
        w_world = 3.5 * np.cross(R_eef[:, 2], -up) + 2.5 * np.cross(f_dir, narrow_dir)
        act = np.zeros(12, dtype=np.float64)
        act[:3] = np.clip(
            (R_b.T @ err) * 6.5,
            -arm_speed_limit,
            arm_speed_limit,
        )
        act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
        act[6] = -1.0
        act[10] = -1.0
        act[11] = -1.0
        adapter.step_12d(act, mode="descend")

    for _ in range(16):
        act = np.zeros(12, dtype=np.float64)
        act[6] = 1.0
        act[10] = -1.0
        act[11] = -1.0
        adapter.step_12d(act, mode="close")

    use_slide_place = bool(slide_in_place or float(dst_xyz[2]) > 1.18)
    lift_z = (
        float(p_grasp[2]) + lift_clearance_m
        if use_slide_place
        else max(float(p_grasp[2]), float(dst_xyz[2])) + lift_clearance_m
    )
    for _ in range(38):
        err = np.array([p_grasp[0], p_grasp[1], lift_z]) - sim.data.body_xpos[bid]
        if float(np.linalg.norm(err)) < 0.015:
            break
        act = np.zeros(12, dtype=np.float64)
        act[:3] = np.clip(
            (R_b.T @ err) * 6.5,
            -arm_speed_limit,
            arm_speed_limit,
        )
        act[6] = 1.0
        act[10] = -1.0
        act[11] = -1.0
        adapter.step_12d(act, mode="lift")

    if dst_fixture is not None:
        navigate_base_to_fixture(
            adapter,
            dst_fixture,
            ref_object=dst_ref_obj,
            gripper="close",
            speed_limit=carry_speed_limit,
        )
        _, _, R_b = get_base_pose_2d(adapter)
        fwd = R_b[:, 0]
        left = R_b[:, 1]
    pos_now, yaw_now, R_b = get_base_pose_2d(adapter)
    d_fwd_d = float(np.dot(dst_xyz[:2] - pos_now, fwd[:2]))
    d_left_d = float(np.dot(dst_xyz[:2] - pos_now, left[:2]))
    dst_standoff = 0.52 if use_slide_place else 0.50
    if not lock_base_on_place:
        navigate_base_to_pose(adapter, pos_now + (d_fwd_d - dst_standoff) * fwd[:2] + d_left_d * left[:2], yaw_now, max_steps=65, backoff_steps=0, pos_tol_m=0.02, gripper="close", speed_limit=carry_speed_limit)
    _, _, R_b = get_base_pose_2d(adapter)
    fwd = R_b[:, 0]
    left = R_b[:, 1]

    torso_dst = 1.0 if dst_xyz[2] > 1.22 else -1.0
    if use_slide_place:
        # Approach in front of cabinet/dispenser at shelf height, then slide horizontally in
        pre_slide = dst_xyz - 0.18 * fwd + np.array([0, 0, release_z + 0.015])
        for tgt_obj in (pre_slide, dst_xyz + np.array([0, 0, release_z])):
            for _ in range(45):
                err = tgt_obj - sim.data.body_xpos[bid]
                if float(np.linalg.norm(err)) < 0.008:
                    break
                R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
                w_world = 3.0 * np.cross(R_eef[:, 2], -up)
                act = np.zeros(12, dtype=np.float64)
                act[:3] = np.clip(
                    (R_b.T @ err) * 6.0,
                    -arm_speed_limit,
                    arm_speed_limit,
                )
                act[3:6] = np.clip(R_b.T @ w_world, -1.0, 1.0)
                act[6] = 1.0
                act[10] = torso_dst
                act[11] = -1.0
                adapter.step_12d(act, mode="slide_place")
        if not release_after_place:
            return bool(OU.check_obj_grasped(inner, obj_key))
        for _ in range(12):
            act = np.zeros(12, dtype=np.float64)
            act[6] = -1.0
            act[10] = torso_dst
            act[11] = -1.0
            adapter.step_12d(act, mode="open")
        pad_now = 0.5 * (sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id])
        for tgt_pad in (
            pad_now + np.array([0, 0, 0.035]),
            pad_now - 0.22 * fwd + np.array([0, 0, 0.08]),
        ):
            for _ in range(28):
                pad_mid = 0.5 * (sim.data.geom_xpos[f1_id] + sim.data.geom_xpos[f2_id])
                err = tgt_pad - pad_mid
                if float(np.linalg.norm(err)) < 0.008:
                    break
                act = np.zeros(12, dtype=np.float64)
                act[:3] = np.clip((R_b.T @ err) * 6.0, -0.8, 0.8)
                act[6] = -1.0
                act[10] = torso_dst
                act[11] = -1.0
                adapter.step_12d(act, mode="slide_retract")
        _retract_until_far(torso_dst)
        return True

    for _ in range(38):
        err = (dst_xyz + np.array([0, 0, 0.16])) - sim.data.body_xpos[bid]
        if float(np.linalg.norm(err)) < 0.015:
            break
        act = np.zeros(12, dtype=np.float64)
        act[:3] = np.clip(
            (R_b.T @ err) * 6.5,
            -arm_speed_limit,
            arm_speed_limit,
        )
        act[6] = 1.0
        act[10] = torso_dst
        act[11] = -1.0
        adapter.step_12d(act, mode="carry")

    for _ in range(30):
        err = (dst_xyz + np.array([0, 0, release_z])) - sim.data.body_xpos[bid]
        if float(np.linalg.norm(err)) < 0.012:
            break
        act = np.zeros(12, dtype=np.float64)
        act[:3] = np.clip(
            (R_b.T @ err) * 6.0,
            -arm_speed_limit,
            arm_speed_limit,
        )
        act[6] = 1.0
        act[10] = torso_dst
        act[11] = -1.0
        adapter.step_12d(act, mode="place")

    if not release_after_place:
        return bool(OU.check_obj_grasped(inner, obj_key))

    for _ in range(10):
        act = np.zeros(12, dtype=np.float64)
        act[6] = -1.0
        act[10] = torso_dst
        act[11] = -1.0
        adapter.step_12d(act, mode="open")

    for _ in range(22):
        act = np.zeros(12, dtype=np.float64)
        act[0] = -0.8
        act[2] = 1.0
        act[6] = -1.0
        act[10] = torso_dst
        act[11] = -1.0
        adapter.step_12d(act, mode="retract")
    _retract_until_far(torso_dst)
    return True


def pour_held_pan_and_return(
    adapter,
    *,
    pan_key: str,
    food_key: str,
    plate_key: str,
    home_xyz: np.ndarray,
) -> dict:
    """Tilt a grasped pan over a plate, restore it upright, and return it."""
    sim = adapter._sim
    inner = adapter.inner
    pan_bid = inner.obj_body_id[pan_key]
    plate_bid = inner.obj_body_id[plate_key]
    eef_site_id = sim.model.site_name2id("gripper0_right_grip_site")
    R_eef0 = sim.data.site_xmat[eef_site_id].reshape(3, 3).copy()
    p_pan = sim.data.body_xpos[pan_bid].copy()
    pan_up0 = sim.data.body_xmat[pan_bid].reshape(3, 3)[:, 2].copy()
    max_tilt_deg = 0.0

    bbox_name = f"{pan_key}_reg_bbox"
    handle_axis = None
    if bbox_name in sim.model.geom_names:
        bbox_gid = sim.model.geom_name2id(bbox_name)
        bbox_center = sim.data.geom_xpos[bbox_gid].copy()
        handle_axis = bbox_center - p_pan
        handle_axis[2] = 0.0
    if handle_axis is None or float(np.linalg.norm(handle_axis)) < 0.03:
        descendant_geoms = []
        for gid in range(sim.model.ngeom):
            body_id = sim.model.geom_bodyid[gid]
            while body_id > 0 and body_id != pan_bid:
                body_id = sim.model.body_parentid[body_id]
            if body_id == pan_bid and sim.model.geom_contype[gid] > 0:
                descendant_geoms.append(gid)
        if descendant_geoms:
            far_gid = max(
                descendant_geoms,
                key=lambda gid: float(
                    np.linalg.norm(
                        sim.data.geom_xpos[gid][:2] - p_pan[:2]
                    )
                ),
            )
            handle_axis = sim.data.geom_xpos[far_gid] - p_pan
            handle_axis[2] = 0.0
        else:
            _, _, R_b = get_base_pose_2d(adapter)
            handle_axis = R_b[:, 1].copy()
    handle_axis /= np.linalg.norm(handle_axis) + 1e-8

    def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
        x, y, z = axis
        K = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )
        return (
            np.eye(3)
            + np.sin(angle) * K
            + (1.0 - np.cos(angle)) * (K @ K)
        )

    def _servo_held_pose(
        target_xyz: np.ndarray,
        target_R: np.ndarray,
        *,
        steps: int,
        mode: str,
        pos_tol: float = 0.015,
    ) -> None:
        nonlocal max_tilt_deg
        for _ in range(steps):
            _, _, R_b = get_base_pose_2d(adapter)
            pos_err = (
                np.asarray(target_xyz, dtype=np.float64)
                - sim.data.body_xpos[pan_bid]
            )
            R_eef = sim.data.site_xmat[eef_site_id].reshape(3, 3)
            q_delta = T.quat_multiply(
                T.mat2quat(target_R),
                T.quat_inverse(T.mat2quat(R_eef)),
            )
            if q_delta[3] < 0:
                q_delta = -q_delta
            w_world = T.quat2axisangle(q_delta)
            if (
                float(np.linalg.norm(pos_err)) < pos_tol
                and float(np.linalg.norm(w_world)) < 0.035
            ):
                break
            act = np.zeros(12, dtype=np.float64)
            act[:3] = np.clip((R_b.T @ pos_err) * 6.0, -0.75, 0.75)
            act[3:6] = np.clip(
                R_b.T @ (3.5 * w_world),
                -0.8,
                0.8,
            )
            act[6] = 1.0
            act[10] = -1.0
            act[11] = -1.0
            adapter.step_12d(act, mode=mode)
            pan_up = sim.data.body_xmat[pan_bid].reshape(3, 3)[:, 2]
            max_tilt_deg = max(
                max_tilt_deg,
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(np.dot(pan_up0, pan_up), -1.0, 1.0)
                        )
                    )
                ),
            )

    plate_xyz = sim.data.body_xpos[plate_bid].copy()
    poured = bool(OU.check_obj_in_receptacle(inner, food_key, plate_key))
    chosen_angle = 0.0
    # Try each direction separately. Different pan assets expose their handle
    # axis with opposite signs, while the physical success condition is the
    # food landing on the plate.
    for direction in (1.0, -1.0):
        if poured:
            break
        dump_dir = direction * np.cross(
            handle_axis,
            np.array([0.0, 0.0, 1.0]),
        )
        dump_dir[2] = 0.0
        dump_dir /= np.linalg.norm(dump_dir) + 1e-8
        # Put the low pan lip over the plate rather than centering the whole
        # pan over it; otherwise the item leaves the far rim and overshoots.
        pour_xyz = (
            plate_xyz
            - 0.06 * dump_dir
            + np.array([0.0, 0.0, 0.14])
        )
        _servo_held_pose(
            pour_xyz,
            R_eef0,
            steps=55,
            mode="pan_pour_position",
        )
        for angle in np.linspace(0.0, direction * np.deg2rad(100.0), 13)[1:]:
            target_R = _rot_axis(handle_axis, float(angle)) @ R_eef0
            _servo_held_pose(
                pour_xyz,
                target_R,
                steps=9,
                mode="pan_tilt",
                pos_tol=0.025,
            )
            chosen_angle = float(angle)
            if OU.check_obj_in_receptacle(inner, food_key, plate_key):
                poured = True
                break
        for shake in (-0.035, 0.035, -0.035, 0.035):
            if poured:
                break
            shake_xyz = pour_xyz + shake * np.cross(handle_axis, np.array([0.0, 0.0, 1.0]))
            _servo_held_pose(
                shake_xyz,
                _rot_axis(handle_axis, chosen_angle) @ R_eef0,
                steps=8,
                mode="pan_pour_shake",
                pos_tol=0.025,
            )
            poured = bool(
                OU.check_obj_in_receptacle(inner, food_key, plate_key)
            )
        _servo_held_pose(
            pour_xyz,
            R_eef0,
            steps=50,
            mode="pan_restore",
        )

    clear_xyz = np.asarray(home_xyz, dtype=np.float64).copy()
    clear_xyz[2] = max(
        float(sim.data.body_xpos[pan_bid][2]),
        float(home_xyz[2]) + 0.20,
    )
    _servo_held_pose(
        clear_xyz,
        R_eef0,
        steps=55,
        mode="pan_return_clear",
    )
    settle_xyz = np.asarray(home_xyz, dtype=np.float64).copy()
    settle_xyz[2] += 0.025
    _servo_held_pose(
        settle_xyz,
        R_eef0,
        steps=40,
        mode="pan_return_place",
    )
    for _ in range(12):
        act = np.zeros(12, dtype=np.float64)
        act[6] = -1.0
        act[10] = -1.0
        act[11] = -1.0
        adapter.step_12d(act, mode="pan_release")
    for _ in range(24):
        act = np.zeros(12, dtype=np.float64)
        act[0] = -0.7
        act[2] = 0.8
        act[6] = -1.0
        act[10] = -1.0
        act[11] = -1.0
        adapter.step_12d(act, mode="pan_retract")

    return {
        "food_on_plate": bool(
            OU.check_obj_in_receptacle(inner, food_key, plate_key)
        ),
        "pan_gripper_far": bool(
            OU.gripper_obj_far(inner, obj_name=pan_key)
        ),
        "robot_touched_food": bool(
            getattr(inner, "_robot_touched_food", False)
        ),
        "max_pan_tilt_deg": round(max_tilt_deg, 1),
        "food_xyz": np.round(
            sim.data.body_xpos[inner.obj_body_id[food_key]],
            4,
        ).tolist(),
        "plate_xyz": np.round(
            sim.data.body_xpos[plate_bid],
            4,
        ).tolist(),
        "pan_xyz": np.round(
            sim.data.body_xpos[pan_bid],
            4,
        ).tolist(),
    }


_CC_CACHE = None


def _make_fast_adapter(task: str, ep_seed: int):
    global _CC_CACHE
    from eval_robocasa import RoboCasaAdapter

    random.seed(ep_seed)
    np.random.seed(ep_seed)
    if _CC_CACHE is None:
        _CC_CACHE = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(
        env_name=task,
        robots="PandaOmron",
        controller_configs=_CC_CACHE,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        obj_registries=("objaverse", "lightwheel", "aigen"),
        obj_instance_split="pretrain",
        layout_ids=-2,
        style_ids=-2,
        seed=ep_seed,
    )
    env.rng = np.random.default_rng(ep_seed)
    env.reset()

    class _DummyGym:
        def __init__(self, e):
            self.unwrapped = type("U", (), {"env": e})()

    ad = RoboCasaAdapter(_DummyGym(env))
    ad.fast_no_render = True
    ad.t = 0
    ad.move_steps = 0
    ad.gripper_command = -1.0
    ad.last_info = {"success": False}
    return ad, env


def run_robocasa_code_policy_episode(task: str, ep: int, seed: int = 7) -> dict:
    started = time.monotonic()
    ep_seed = seed + ep * 100
    env = None
    try:
        adapter, env = _make_fast_adapter(task, ep_seed)
        inner = adapter.inner
        sim = adapter._sim
        pos0, yaw0, R_b = get_base_pose_2d(adapter)
        fwd = R_b[:, 0]
        left = R_b[:, 1]
        skill_trace: dict = {}

        if task == "CloseFridge":
            close_fixture_doors(adapter, inner.fxtr)

        elif task == "NavigateKitchen":
            navigate_base_to_pose(
                adapter,
                inner.target_pos[:2],
                float(inner.target_ori[2]),
                max_steps=260,
                pos_tol_m=0.04,
                cos_yaw_tol=0.99,
                gripper="open",
            )

        elif task == "TurnOnMicrowave":
            start_gid = sim.model.geom_name2id(
                f"{inner.microwave.naming_prefix}start_button"
            )
            poke_button_l_shaped(
                adapter,
                sim.data.geom_xpos[start_gid].copy(),
            )

        elif task == "ScrubCuttingBoard":
            p_board = sim.data.body_xpos[
                inner.obj_body_id["cutting_board"]
            ].copy()
            held = grasp_and_place_smart(
                adapter,
                "sponge",
                p_board,
                release_z=0.025,
                force_top_grasp=True,
                use_center_grasp=True,
                src_fixture=inner.counter,
                src_ref_obj="sponge",
                dst_fixture=inner.counter,
                dst_ref_obj="cutting_board",
                release_after_place=False,
            )
            if held:
                _, _, R_b = get_base_pose_2d(adapter)
                fwd_s, left_s = R_b[:, 0], R_b[:, 1]
                sponge_bid = inner.obj_body_id["sponge"]
                for ofs in (-0.075, -0.035, 0.0, 0.035, 0.075, -0.055, 0.055):
                    p_board = sim.data.body_xpos[
                        inner.obj_body_id["cutting_board"]
                    ].copy()
                    tgt_s = (
                        p_board
                        + ofs * left_s
                        + np.array([0.0, 0.0, 0.008])
                    )
                    for _ in range(18):
                        err = tgt_s - sim.data.body_xpos[sponge_bid]
                        act = np.zeros(12, dtype=np.float64)
                        act[:3] = np.clip((R_b.T @ err) * 6.5, -0.8, 0.8)
                        act[2] = min(act[2], -0.25)
                        act[6] = 1.0
                        act[10] = -1.0
                        act[11] = -1.0
                        adapter.step_12d(act, mode="scrub_sweep")
                for _ in range(12):
                    act = np.zeros(12, dtype=np.float64)
                    act[6] = -1.0
                    act[10] = -1.0
                    act[11] = -1.0
                    adapter.step_12d(act, mode="scrub_open")
                for _ in range(28):
                    if OU.gripper_obj_far(inner, obj_name="sponge", th=0.15):
                        break
                    act = np.zeros(12, dtype=np.float64)
                    act[0] = -0.8
                    act[2] = 0.9
                    act[6] = -1.0
                    act[10] = -1.0
                    act[11] = -1.0
                    adapter.step_12d(act, mode="scrub_retract")

        elif task == "WeighIngredients":
            p_scale = sim.data.body_xpos[inner.obj_body_id["digital_scale"]].copy()
            grasp_and_place_smart(
                adapter,
                "obj",
                p_scale,
                release_z=0.04,
                src_fixture=inner.cab,
                dst_fixture=inner.counter,
                dst_ref_obj="digital_scale",
            )
            close_fixture_doors(adapter, inner.cab)

        elif task == "BreadSelection":
            p_board = sim.data.body_xpos[
                inner.obj_body_id["cutting_board"]
            ].copy()
            grasp_and_place_smart(
                adapter,
                "jam",
                p_board + 0.055 * left,
                release_z=0.05,
                src_fixture=inner.cab,
                dst_fixture=inner.counter,
                dst_ref_obj="cutting_board",
            )
            _, _, R_b = get_base_pose_2d(adapter)
            left = R_b[:, 1]
            p_board = sim.data.body_xpos[
                inner.obj_body_id["cutting_board"]
            ].copy()
            grasp_and_place_smart(
                adapter,
                "croissant",
                p_board - 0.055 * left,
                release_z=0.03,
            )

        elif task == "OpenDrawer":
            open_slide_drawer(adapter, inner.drawer, pull_wide=False)

        elif task == "PickPlaceDrawerToCounter":
            open_slide_drawer(
                adapter, inner.drawer, pull_wide=True, align_obj_key="obj"
            )
            _, _, R_b = get_base_pose_2d(adapter)
            left = R_b[:, 1]
            p_distr = sim.data.body_xpos[inner.obj_body_id["distr"]].copy()
            p_cnt = fixture_region_center_world(
                inner.counter, inner, z_offset=0.03, ref=inner.drawer
            )
            side = (
                1.0
                if float(np.dot(p_cnt[:2] - p_distr[:2], left[:2])) >= 0.0
                else -1.0
            )
            p_dst = p_distr + side * 0.16 * left
            p_dst[2] = p_cnt[2]
            grasp_and_place_smart(
                adapter,
                "obj",
                p_dst,
                release_z=0.035,
                force_top_grasp=True,
                use_center_grasp=True,
                grasp_z_offset=-0.008,
                lock_base_on_grasp=True,
                lock_base_on_place=True,
                lift_clearance_m=0.28,
            )

        elif task == "CuttingToolSelection":
            tool = inner._CUTTING_MAP[inner.food]
            open_slide_drawer(
                adapter, inner.drawer, pull_wide=True, align_obj_key=tool
            )
            _, _, R_b = get_base_pose_2d(adapter)
            left = R_b[:, 1]
            p_board = sim.data.body_xpos[
                inner.obj_body_id["food_container"]
            ].copy()
            grasp_and_place_smart(
                adapter,
                tool,
                p_board + 0.045 * left,
                release_z=0.035,
                force_top_grasp=True,
                use_center_grasp=True,
                grasp_z_offset=-0.008,
                lock_base_on_grasp=True,
                lock_base_on_place=True,
                lift_clearance_m=0.28,
            )

        elif task == "PickPlaceToasterToCounter":
            p_plate = sim.data.body_xpos[inner.obj_body_id["plate"]].copy()
            grasp_and_place_smart(
                adapter,
                "obj",
                p_plate,
                release_z=0.05,
                force_top_grasp=True,
                use_center_grasp=True,
                grasp_z_offset=0.042,
                lift_clearance_m=0.18,
                src_fixture=inner.toaster,
                dst_fixture=inner.counter,
                dst_ref_obj="plate",
            )

        elif task == "CoffeeSetupMug":
            pour_site_id = sim.model.site_name2id(
                f"{inner.coffee_machine.naming_prefix}receptacle_place_site"
            )
            p_pour = sim.data.site_xpos[pour_site_id].copy()
            grasp_and_place_smart(
                adapter,
                "obj",
                p_pour,
                release_z=0.015,
                is_rim=True,
                force_top_grasp=True,
                slide_in_place=True,
                carry_speed_limit=0.35,
                lift_clearance_m=0.10,
                src_fixture=inner.counter,
                src_ref_obj="obj",
                dst_fixture=inner.coffee_machine,
            )

        elif task == "TurnOffStove":
            turn_stove_knob(adapter, inner.stove, inner.knob, mode="off")

        elif task == "RinseSinkBasin":
            rinse_sink_basin_skill(adapter, inner.sink)

        elif task == "WashLettuce":
            turn_sink_water_on_and_align(adapter, inner.sink)
            water_sid = sim.model.site_name2id(
                f"{inner.sink.naming_prefix}water"
            )
            water_xyz = sim.data.site_xpos[water_sid].copy()
            p_target = np.array(
                [water_xyz[0], water_xyz[1], water_xyz[2] - 0.14],
                dtype=np.float64,
            )
            grasp_and_place_smart(
                adapter,
                "lettuce",
                p_target,
                release_z=0.0,
                force_top_grasp=True,
                use_center_grasp=True,
                carry_speed_limit=0.35,
                src_fixture=inner.counter,
                src_ref_obj="lettuce",
                dst_fixture=inner.sink,
                release_after_place=False,
            )
            for _ in range(32):
                if adapter.success:
                    break
                act = np.zeros(12, dtype=np.float64)
                act[6] = 1.0
                act[10] = -1.0
                act[11] = -1.0
                adapter.step_12d(act, mode="wash_hold")

        elif task == "PreSoakPan":
            water_sid = sim.model.site_name2id(
                f"{inner.sink.naming_prefix}water"
            )
            water_xyz = sim.data.site_xpos[water_sid].copy()
            R_sink = T.euler2mat(np.array([0.0, 0.0, inner.sink.rot]))
            region_targets = [
                np.asarray(inner.sink.pos, dtype=np.float64)
                + R_sink @ np.asarray(reg["offset"], dtype=np.float64)
                for reg in inner.sink.get_reset_regions(inner).values()
            ]
            basin_target = min(
                region_targets,
                key=lambda p: float(np.linalg.norm(p[:2] - water_xyz[:2])),
            )
            p_sink = np.array(
                [basin_target[0], basin_target[1], basin_target[2] + 0.05],
                dtype=np.float64,
            )
            grasp_and_place_smart(
                adapter,
                "obj1",
                p_sink,
                release_z=0.04,
                force_top_grasp=True,
                carry_speed_limit=0.30,
                src_fixture=inner.counter,
                src_ref_obj="obj1",
                dst_fixture=inner.sink,
            )
            grasp_and_place_smart(
                adapter,
                "obj2",
                p_sink,
                release_z=0.06,
                force_top_grasp=True,
                use_center_grasp=True,
                carry_speed_limit=0.35,
                src_fixture=inner.counter,
                src_ref_obj="obj2",
                dst_fixture=inner.sink,
            )
            turn_sink_water_on_and_align(adapter, inner.sink)

        elif task == "KettleBoiling":
            p_distr = sim.data.body_xpos[inner.obj_body_id["stove_distr"]].copy()
            best_loc = None
            best_site_pos = None
            best_dist = -1.0
            knobs_state = inner.stove.get_knobs_state(env=inner)
            for loc, site in inner.stove.burner_sites.items():
                if site is not None and loc in knobs_state:
                    s_pos = np.asarray(
                        sim.data.get_site_xpos(site.get("name")),
                        dtype=np.float64,
                    ).copy()
                    d = float(np.linalg.norm(s_pos[:2] - p_distr[:2]))
                    if d > best_dist:
                        best_dist = d
                        best_loc = loc
                        best_site_pos = s_pos
            if best_loc is not None and best_site_pos is not None:
                grasp_and_place_smart(
                    adapter,
                    "obj",
                    best_site_pos,
                    release_z=0.04,
                    force_top_grasp=True,
                    carry_speed_limit=0.35,
                    src_fixture=inner.counter,
                    src_ref_obj="obj",
                    dst_fixture=inner.stove,
                )
                turn_stove_knob(adapter, inner.stove, best_loc, mode="on")

        elif task == "SearingMeat":
            site = inner.stove.burner_sites.get(inner.knob)
            if site is not None:
                p_burner = np.asarray(
                    sim.data.get_site_xpos(site.get("name")),
                    dtype=np.float64,
                ).copy()
                grasp_and_place_smart(
                    adapter,
                    "pan",
                    p_burner,
                    release_z=0.035,
                    force_front_grasp=True,
                    carry_speed_limit=0.30,
                    src_fixture=inner.cab,
                    src_ref_obj="pan",
                    dst_fixture=inner.stove,
                )
                p_pan = sim.data.body_xpos[inner.obj_body_id["pan"]].copy()
                grasp_and_place_smart(
                    adapter,
                    "meat",
                    p_pan,
                    release_z=0.05,
                    force_top_grasp=True,
                    use_center_grasp=True,
                    carry_speed_limit=0.35,
                    src_fixture=inner.counter,
                    src_ref_obj="meat",
                    dst_fixture=inner.stove,
                    dst_ref_obj="pan",
                )
                turn_stove_knob(adapter, inner.stove, inner.knob, mode="on")

        elif task == "RecycleBottlesByType":
            mystery_target = (
                "bottle_glass2"
                if getattr(inner, "choice", "alcohol") == "alcohol"
                else "bottle_plastic2"
            )
            pairs = [
                ("bottle_plastic_middle", "bottle_plastic1", 0.10),
                ("bottle_glass_middle", "bottle_glass1", -0.10),
                ("mystery_middle", mystery_target, 0.10),
            ]
            pairs.sort(
                key=lambda item: float(
                    np.dot(
                        sim.data.body_xpos[inner.obj_body_id[item[0]]][:2]
                        - pos0,
                        fwd[:2],
                    )
                )
            )
            for idx, (src_k, ref_k, lat_ofs) in enumerate(pairs):
                bid = inner.obj_body_id[src_k]
                p_obj = sim.data.body_xpos[bid].copy()
                if idx > 0:
                    pos_c, yaw_c, R_b = get_base_pose_2d(adapter)
                    fwd_c, left_c = R_b[:, 0], R_b[:, 1]
                    d_fwd_c = float(np.dot(p_obj[:2] - pos_c, fwd_c[:2]))
                    navigate_base_to_pose(
                        adapter,
                        pos_c + (d_fwd_c - 0.78) * fwd_c[:2],
                        yaw0,
                        max_steps=35,
                        backoff_steps=0,
                        pos_tol_m=0.02,
                        gripper="open",
                    )
                    pos_c2, _, R_b = get_base_pose_2d(adapter)
                    d_fwd_c2 = float(np.dot(p_obj[:2] - pos_c2, fwd_c[:2]))
                    d_left_c2 = float(np.dot(p_obj[:2] - pos_c2, left_c[:2]))
                    navigate_base_to_pose(
                        adapter,
                        pos_c2 + (d_fwd_c2 - 0.78) * fwd_c[:2] + d_left_c2 * left_c[:2],
                        yaw0,
                        max_steps=65,
                        backoff_steps=0,
                        pos_tol_m=0.02,
                        gripper="open",
                    )
                obj = inner.objects[src_k]
                quat = T.convert_quat(sim.data.body_xquat[bid], to="xyzw")
                bbox = np.asarray(
                    obj.get_bbox_points(trans=p_obj, rot=quat),
                    dtype=np.float64,
                )
                top_z = float(np.max(bbox[:, 2]))
                z_ofs = max(-0.005, (top_z - 0.032) - float(p_obj[2]))
                p_ref = sim.data.body_xpos[inner.obj_body_id[ref_k]].copy()
                _, _, R_b = get_base_pose_2d(adapter)
                left = R_b[:, 1]
                grasp_and_place_smart(
                    adapter,
                    src_k,
                    p_ref + lat_ofs * left,
                    release_z=0.05,
                    force_top_grasp=True,
                    use_center_grasp=True,
                    grasp_z_offset=z_ofs,
                    standoff_override=0.55,
                    lift_clearance_m=0.26,
                    carry_speed_limit=0.35,
                )

        elif task == "DeliverStraw":
            open_slide_drawer(
                adapter, inner.drawer, pull_wide=True, align_obj_key="straw"
            )
            p_cup = sim.data.body_xpos[
                inner.obj_body_id["glass_cup"]
            ].copy()
            grasp_and_place_smart(
                adapter,
                "straw",
                p_cup,
                release_z=0.09,
                force_top_grasp=True,
                use_center_grasp=True,
                grasp_z_offset=-0.008,
                lock_base_on_grasp=True,
                lift_clearance_m=0.28,
                dst_fixture=inner.dining_counter,
                dst_ref_obj="glass_cup",
            )

        elif task == "SetUpCuttingStation":
            _, _, R_b = get_base_pose_2d(adapter)
            left = R_b[:, 1]
            p_board = sim.data.body_xpos[
                inner.obj_body_id["receptacle"]
            ].copy()
            grasp_and_place_smart(
                adapter,
                "meat",
                p_board + 0.045 * left,
                release_z=0.04,
                force_top_grasp=True,
                use_center_grasp=True,
                src_fixture=inner.counter,
                src_ref_obj="meat",
                dst_fixture=inner.counter,
                dst_ref_obj="receptacle",
            )
            open_slide_drawer(
                adapter, inner.drawer, pull_wide=True, align_obj_key="knife"
            )
            _, _, R_b = get_base_pose_2d(adapter)
            left = R_b[:, 1]
            p_board = sim.data.body_xpos[
                inner.obj_body_id["receptacle"]
            ].copy()
            grasp_and_place_smart(
                adapter,
                "knife",
                p_board - 0.045 * left,
                release_z=0.035,
                force_top_grasp=True,
                use_center_grasp=True,
                grasp_z_offset=-0.008,
                lock_base_on_grasp=True,
                lock_base_on_place=True,
                lift_clearance_m=0.28,
            )

        elif task == "CategorizeCondiments":
            p_cab_center = fixture_region_center_world(
                inner.cab, inner, z_offset=0.02
            )
            _, _, R_b = get_base_pose_2d(adapter)
            left = R_b[:, 1]
            p_bottle = sim.data.body_xpos[
                inner.obj_body_id["cab_obj1"]
            ].copy()
            side1 = (
                1.0
                if float(np.dot(p_cab_center[:2] - p_bottle[:2], left[:2])) >= 0.0
                else -1.0
            )
            dst1 = p_bottle + side1 * 0.085 * left
            dst1[2] = p_cab_center[2]
            grasp_and_place_smart(
                adapter,
                "obj1",
                dst1,
                release_z=0.02,
                force_top_grasp=True,
                slide_in_place=True,
                carry_speed_limit=0.35,
                src_fixture=inner.counter,
                src_ref_obj="obj1",
                dst_fixture=inner.cab,
                dst_ref_obj="cab_obj1",
            )
            _, _, R_b = get_base_pose_2d(adapter)
            left = R_b[:, 1]
            p_shaker = sim.data.body_xpos[
                inner.obj_body_id["cab_obj2"]
            ].copy()
            side2 = (
                1.0
                if float(np.dot(p_cab_center[:2] - p_shaker[:2], left[:2])) >= 0.0
                else -1.0
            )
            dst2 = p_shaker + side2 * 0.085 * left
            dst2[2] = p_cab_center[2]
            grasp_and_place_smart(
                adapter,
                "obj2",
                dst2,
                release_z=0.02,
                force_top_grasp=True,
                slide_in_place=True,
                carry_speed_limit=0.35,
                src_fixture=inner.counter,
                src_ref_obj="obj2",
                dst_fixture=inner.cab,
                dst_ref_obj="cab_obj2",
            )

        elif task == "GatherTableware":
            p_g1 = sim.data.body_xpos[inner.obj_body_id["glass1"]].copy()
            p_g2 = sim.data.body_xpos[inner.obj_body_id["glass2"]].copy()
            p_bowl = sim.data.body_xpos[inner.obj_body_id["bowl"]].copy()
            # Choose the glass in cab1 farther from bowl as cluster anchor
            d1 = float(np.linalg.norm(p_g1[:2] - p_bowl[:2]))
            d2 = float(np.linalg.norm(p_g2[:2] - p_bowl[:2]))
            anchor_k = "glass1" if d1 >= d2 else "glass2"
            other_k = "glass2" if d1 >= d2 else "glass1"
            p_anchor = sim.data.body_xpos[inner.obj_body_id[anchor_k]].copy()
            _, _, R_b = get_base_pose_2d(adapter)
            left = R_b[:, 1]
            away_dir = p_anchor[:2] - p_bowl[:2]
            side = 1.0 if float(np.dot(away_dir, left[:2])) >= 0.0 else -1.0
            if float(np.linalg.norm(p_g1[:2] - p_g2[:2])) > 0.22:
                grasp_and_place_smart(
                    adapter,
                    other_k,
                    p_anchor + side * 0.075 * left,
                    release_z=0.02,
                    force_front_grasp=True,
                    carry_speed_limit=0.30,
                    src_fixture=inner.cab,
                    src_ref_obj=other_k,
                    dst_fixture=inner.cab,
                    dst_ref_obj=anchor_k,
                )
            p_anchor = sim.data.body_xpos[inner.obj_body_id[anchor_k]].copy()
            _, _, R_b = get_base_pose_2d(adapter)
            left = R_b[:, 1]
            grasp_and_place_smart(
                adapter,
                "glass3",
                p_anchor - side * 0.075 * left,
                release_z=0.02,
                force_front_grasp=True,
                carry_speed_limit=0.30,
                src_fixture=inner.cab2,
                src_ref_obj="glass3",
                dst_fixture=inner.cab,
                dst_ref_obj=anchor_k,
            )

        elif task == "SeparateFreezerRack":
            p_meat_shelf = fixture_region_center_world(
                inner.fridge,
                inner,
                z_offset=0.04,
                compartment="freezer",
                rack_index=-2,
                reg_type=("shelf",),
            )
            grasp_and_place_smart(
                adapter,
                "meat_tupperware",
                p_meat_shelf,
                release_z=0.04,
                is_rim=True,
                carry_speed_limit=0.15,
                src_fixture=inner.counter,
                src_ref_obj="meat_tupperware",
                dst_fixture=inner.fridge,
            )
            p_veg_shelf = fixture_region_center_world(
                inner.fridge,
                inner,
                z_offset=0.04,
                compartment="freezer",
                rack_index=-1,
                reg_type=("shelf",),
            )
            grasp_and_place_smart(
                adapter,
                "veg_tupperware",
                p_veg_shelf,
                release_z=0.04,
                is_rim=True,
                carry_speed_limit=0.15,
                src_fixture=inner.counter,
                src_ref_obj="veg_tupperware",
                dst_fixture=inner.fridge,
            )

        elif task == "SteamInMicrowave":
            p_bowl = sim.data.body_xpos[inner.obj_body_id["bowl"]].copy()
            grasp_and_place_smart(
                adapter,
                "vegetable",
                p_bowl,
                release_z=0.07,
                src_fixture=inner.sink,
                dst_fixture=inner.counter,
                dst_ref_obj="bowl",
            )
            skill_trace["vegetable_in_bowl"] = bool(
                OU.check_obj_in_receptacle(inner, "vegetable", "bowl")
            )
            p_microwave = fixture_region_center_world(
                inner.microwave,
                inner,
                z_offset=0.04,
            )
            grasp_and_place_smart(
                adapter,
                "bowl",
                p_microwave,
                release_z=0.035,
                is_rim=True,
                dst_fixture=inner.microwave,
            )
            skill_trace["bowl_in_microwave"] = bool(
                OU.obj_inside_of(inner, "bowl", inner.microwave)
            )
            skill_trace["microwave_terminal"] = close_microwave_and_start(
                adapter,
                inner.microwave,
            )

        elif task == "WashFruitColander":
            water_sid = sim.model.site_name2id(
                f"{inner.sink.naming_prefix}water"
            )
            water_xyz = sim.data.site_xpos[water_sid].copy()
            R_sink = T.euler2mat(
                np.array([0.0, 0.0, inner.sink.rot])
            )
            region_targets = []
            for region in inner.sink.get_reset_regions(inner).values():
                p_region = (
                    np.asarray(inner.sink.pos, dtype=np.float64)
                    + R_sink
                    @ np.asarray(region["offset"], dtype=np.float64)
                )
                region_targets.append(p_region)
            basin_target = min(
                region_targets,
                key=lambda p: float(
                    np.linalg.norm(p[:2] - water_xyz[:2])
                ),
            )
            p_colander = np.array(
                [water_xyz[0], water_xyz[1], basin_target[2] + 0.065],
                dtype=np.float64,
            )
            grasp_and_place_smart(
                adapter,
                "colander",
                p_colander,
                release_z=0.02,
                is_rim=True,
                carry_speed_limit=0.30,
                src_fixture=inner.counter,
                src_ref_obj="colander",
                dst_fixture=inner.sink,
            )
            for fruit_idx in range(inner.num_fruit):
                fruit_key = f"fruit{fruit_idx}"
                p_colander_now = sim.data.body_xpos[
                    inner.obj_body_id["colander"]
                ].copy()
                _, _, R_b = get_base_pose_2d(adapter)
                side = (
                    fruit_idx - 0.5 * (inner.num_fruit - 1)
                ) * 0.045
                grasp_and_place_smart(
                    adapter,
                    fruit_key,
                    p_colander_now + side * R_b[:, 1],
                    release_z=0.075,
                    src_fixture=inner.counter,
                    src_ref_obj=fruit_key,
                    dst_fixture=inner.sink,
                    dst_ref_obj="colander",
                )
            skill_trace["fruits_in_colander"] = bool(
                all(
                    OU.check_obj_in_receptacle(
                        inner,
                        f"fruit{idx}",
                        "colander",
                    )
                    for idx in range(inner.num_fruit)
                )
            )
            skill_trace["faucet"] = turn_sink_water_on_and_align(
                adapter,
                inner.sink,
                target_obj="colander",
            )

        elif task == "PanTransfer":
            p_home = sim.data.body_xpos[
                inner.obj_body_id["vegetable_container"]
            ].copy()
            p_plate = sim.data.body_xpos[
                inner.obj_body_id["plate"]
            ].copy()
            held = grasp_and_place_smart(
                adapter,
                "vegetable_container",
                p_plate + np.array([0.0, 0.0, 0.19]),
                release_z=0.0,
                force_top_grasp=True,
                carry_speed_limit=0.18,
                arm_speed_limit=0.40,
                lift_clearance_m=0.10,
                src_fixture=inner.stove,
                src_ref_obj="vegetable_container",
                release_after_place=False,
            )
            skill_trace["pan_grasped"] = bool(held)
            if held:
                skill_trace["tilt_pour"] = pour_held_pan_and_return(
                    adapter,
                    pan_key="vegetable_container",
                    food_key="vegetable",
                    plate_key="plate",
                    home_xyz=p_home,
                )

        elif task == "PrepareCoffee":
            open_fixture_doors(adapter, inner.cab)
            pour_site_id = sim.model.site_name2id(
                f"{inner.coffee_machine.naming_prefix}receptacle_place_site"
            )
            p_pour = sim.data.site_xpos[pour_site_id].copy()
            grasp_and_place_smart(
                adapter,
                "obj",
                p_pour,
                release_z=0.015,
                force_front_grasp=True,
                slide_in_place=True,
                carry_speed_limit=0.35,
                src_fixture=inner.cab,
                src_ref_obj="obj",
                dst_fixture=inner.coffee_machine,
            )
            navigate_base_to_fixture(
                adapter,
                inner.coffee_machine,
                gripper="close",
                torso=-1.0,
            )
            button_name = inner.coffee_machine._start_button_names[0]
            button_id = sim.model.geom_name2id(
                f"{inner.coffee_machine.naming_prefix}{button_name}"
            )
            poke_button_l_shaped(
                adapter,
                sim.data.geom_xpos[button_id].copy(),
            )

        elif task == "GetToastedBread":
            toast_slot = next(
                (
                    slot_pair
                    for slot_pair in range(
                        len(inner.toaster.get_state(inner).keys())
                    )
                    if inner.toaster.check_slot_contact(
                        inner,
                        "obj",
                        slot_pair=slot_pair,
                    )
                ),
                0,
            )
            press_toaster_lever(adapter, inner.toaster, toast_slot)
            for _ in range(520):
                act = np.zeros(12, dtype=np.float64)
                act[6] = -1.0
                act[11] = -1.0
                adapter.step_12d(act, mode="wait_for_toast")
            p_plate = sim.data.body_xpos[
                inner.obj_body_id["plate"]
            ].copy()
            grasp_and_place_smart(
                adapter,
                "obj",
                p_plate,
                release_z=0.05,
                force_top_grasp=True,
                use_center_grasp=True,
                grasp_z_offset=0.042,
                lift_clearance_m=0.18,
                src_fixture=inner.toaster,
                dst_fixture=inner.dining_counter,
                dst_ref_obj="plate",
            )

        elif task == "StackBowlsCabinet":
            # Stack smaller bowl1 into larger bowl2 on the open counter first, then slide stacked bowl2 into cabinet
            p_bowl2 = sim.data.body_xpos[
                inner.obj_body_id["bowl2"]
            ].copy()
            grasp_and_place_smart(
                adapter,
                "bowl1",
                p_bowl2,
                release_z=0.055,
                is_rim=True,
                carry_speed_limit=0.35,
                src_fixture=inner.counter,
                src_ref_obj="bowl1",
                dst_fixture=inner.counter,
                dst_ref_obj="bowl2",
            )
            p_cabinet = fixture_region_center_world(
                inner.cabinet,
                inner,
                z_offset=0.03,
            )
            grasp_and_place_smart(
                adapter,
                "bowl2",
                p_cabinet,
                release_z=0.025,
                is_rim=True,
                slide_in_place=True,
                carry_speed_limit=0.25,
                src_fixture=inner.counter,
                src_ref_obj="bowl2",
                dst_fixture=inner.cabinet,
            )

        elif task == "ArrangeTea":
            p_tray = sim.data.body_xpos[inner.obj_body_id["container"]].copy()
            _, _, R_b = get_base_pose_2d(adapter)
            left = R_b[:, 1]
            grasp_and_place_smart(
                adapter,
                "obj",
                p_tray - 0.055 * left,
                release_z=0.04,
                force_front_grasp=True,
                carry_speed_limit=0.35,
                src_fixture=inner.cab,
                src_ref_obj="obj",
                dst_fixture=inner.counter,
                dst_ref_obj="container",
            )
            p_tray = sim.data.body_xpos[inner.obj_body_id["container"]].copy()
            _, _, R_b = get_base_pose_2d(adapter)
            left = R_b[:, 1]
            grasp_and_place_smart(
                adapter,
                "obj2",
                p_tray + 0.055 * left,
                release_z=0.04,
                force_top_grasp=True,
                carry_speed_limit=0.35,
                src_fixture=inner.counter,
                src_ref_obj="obj2",
                dst_fixture=inner.counter,
                dst_ref_obj="container",
            )
            navigate_base_to_fixture(
                adapter, inner.cab, gripper="close", torso=1.0
            )
            close_fixture_doors(adapter, inner.cab)

        elif task == "PortionHotDogs":
            for src_k, dst_k, side in [
                ("sausage1", "plate1", 0.03),
                ("sausage2", "plate2", 0.03),
                ("hotdog_bun1", "plate1", -0.03),
                ("hotdog_bun2", "plate2", -0.03),
            ]:
                p_plate = sim.data.body_xpos[inner.obj_body_id[dst_k]].copy()
                _, _, R_b = get_base_pose_2d(adapter)
                left = R_b[:, 1]
                grasp_and_place_smart(
                    adapter,
                    src_k,
                    p_plate + side * left,
                    release_z=0.045,
                    force_top_grasp=True,
                    use_center_grasp=True,
                    carry_speed_limit=0.35,
                    src_fixture=inner.dining_counter,
                    src_ref_obj=src_k,
                    dst_fixture=inner.dining_counter,
                    dst_ref_obj=dst_k,
                )

        elif task == "PackIdenticalLunches":
            open_fixture_doors(adapter, inner.fridge)
            for src, dst_k, side in [
                ("vegetable0", "tupperware0", -0.035),
                ("meat0", "tupperware0", 0.035),
                ("vegetable1", "tupperware1", -0.035),
                ("meat1", "tupperware1", 0.035),
            ]:
                p_dst = sim.data.body_xpos[inner.obj_body_id[dst_k]].copy()
                _, _, R_b = get_base_pose_2d(adapter)
                left = R_b[:, 1]
                grasp_and_place_smart(
                    adapter,
                    src,
                    p_dst + side * left,
                    release_z=0.05,
                    force_front_grasp=True,
                    carry_speed_limit=0.35,
                    src_fixture=inner.fridge,
                    src_ref_obj=src,
                    dst_fixture=inner.counter,
                    dst_ref_obj=dst_k,
                )

        elif task == "MakeIceLemonade":
            open_fixture_doors(adapter, inner.fridge)
            p_cup = sim.data.body_xpos[inner.obj_body_id["glass_cup"]].copy()
            grasp_and_place_smart(
                adapter,
                "lemon_wedge",
                p_cup,
                release_z=0.08,
                force_front_grasp=True,
                carry_speed_limit=0.35,
                src_fixture=inner.fridge,
                src_ref_obj="lemon_wedge",
                dst_fixture=inner.counter,
                dst_ref_obj="glass_cup",
            )
            p_cup = sim.data.body_xpos[inner.obj_body_id["glass_cup"]].copy()
            grasp_and_place_smart(
                adapter,
                "ice_cube1",
                p_cup,
                release_z=0.08,
                force_top_grasp=True,
                use_center_grasp=True,
                carry_speed_limit=0.35,
                src_fixture=inner.counter,
                src_ref_obj="ice_cube1",
                dst_fixture=inner.counter,
                dst_ref_obj="glass_cup",
            )

        elif task == "GarnishPancake":
            open_fixture_doors(adapter, inner.fridge)
            p_pk = sim.data.body_xpos[inner.obj_body_id["pancake"]].copy()
            grasp_and_place_smart(
                adapter,
                "strawberry",
                p_pk,
                release_z=0.035,
                force_front_grasp=True,
                carry_speed_limit=0.35,
                src_fixture=inner.fridge,
                src_ref_obj="strawberry",
                dst_fixture=inner.dining_counter,
                dst_ref_obj="pancake",
            )

        elif task == "ArrangeBreadBasket":
            open_fixture_doors(adapter, inner.cab)
            p_bsk = sim.data.body_xpos[inner.obj_body_id["basket"]].copy()
            grasp_and_place_smart(
                adapter,
                "bread",
                p_bsk,
                release_z=0.06,
                is_rim=True,
                force_front_grasp=True,
                src_fixture=inner.cab,
                dst_fixture=inner.counter_small,
                dst_ref_obj="basket",
            )
            p_tbl = fixture_region_center_world(
                inner.dining_table,
                inner,
                z_offset=0.045,
            )
            R_tbl = T.euler2mat(
                np.array([0.0, 0.0, inner.dining_table.rot])
            )
            candidates = [
                p_tbl + R_tbl @ np.array([side * 0.25, 0.0, 0.0])
                for side in (-1.0, 1.0)
            ]
            distractors = [
                sim.data.body_xpos[inner.obj_body_id[key]]
                for key in ("dstr_dining", "dstr_dining2")
            ]
            p_tbl = max(
                candidates,
                key=lambda p: min(
                    np.linalg.norm(p[:2] - q[:2])
                    for q in distractors
                ),
            )
            grasp_and_place_smart(
                adapter,
                "basket",
                p_tbl,
                release_z=0.04,
                is_rim=True,
                carry_speed_limit=0.25,
                dst_fixture=inner.dining_table,
            )

        elif task == "StoreLeftoversInBowl":
            p_bowl = sim.data.body_xpos[inner.obj_body_id["bowl"]].copy()
            grasp_and_place_smart(
                adapter,
                "chicken_drumstick",
                p_bowl - 0.02 * left,
                release_z=0.06,
                src_fixture=inner.dining_counter,
                src_ref_obj="chicken_drumstick",
                dst_fixture=inner.dining_counter,
                dst_ref_obj="bowl",
            )
            p_bowl = sim.data.body_xpos[inner.obj_body_id["bowl"]].copy()
            grasp_and_place_smart(
                adapter,
                "vegetable",
                p_bowl + 0.02 * left,
                release_z=0.06,
                src_fixture=inner.dining_counter,
                src_ref_obj="vegetable",
                dst_fixture=inner.dining_counter,
                dst_ref_obj="bowl",
            )
            fr = inner.fridge
            regs = fr.get_reset_regions(inner, compartment="fridge", rack_index=0, reg_type=("shelf",))
            rname = list(regs.keys())[0]
            reg = fr._regions[rname]
            p0, px, py = reg["p0"], reg["px"], reg["py"]
            local_pt = np.array([0.5 * (p0[0] + px[0]), 0.5 * (p0[1] + py[1]), p0[2] + 0.025])
            R_fr = T.euler2mat(np.array([0.0, 0.0, fr.rot]))
            p_shelf = np.asarray(fr.pos, dtype=np.float64) + R_fr @ local_pt
            grasp_and_place_smart(adapter, "bowl", p_shelf, release_z=0.04, is_rim=True, carry_speed_limit=0.25, dst_fixture=fr)

        skill_trace["final_objs"] = {
            k: np.round(sim.data.body_xpos[v], 3).tolist()
            for k, v in inner.obj_body_id.items()
        }
        return {
            "task": task,
            "episode": int(ep),
            "seed": int(seed),
            "mode": "code_policy",
            "policy_scope": "full_task_privileged_geometry",
            "success": bool(adapter.success),
            "steps": int(adapter.t),
            "crash": False,
            "skill_trace": skill_trace,
            "duration_s": round(time.monotonic() - started, 2),
        }
    except Exception as exc:
        return {
            "task": task,
            "episode": int(ep),
            "seed": int(seed),
            "mode": "code_policy",
            "success": False,
            "steps": 0,
            "crash": True,
            "reason": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.monotonic() - started, 2),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _worker_run(args: Tuple[str, List[int], str]) -> str:
    task, eps, out_path_str = args
    out_path = Path(out_path_str)
    succ = 0
    for ep in eps:
        r = run_robocasa_code_policy_episode(task, ep)
        succ += int(bool(r["success"]))
        with out_path.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    return f"[robocasa_code_policy_v7] {task:28s}: {succ:2d}/{len(eps)} ({succ/len(eps)*100:5.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/robocasa/results_code_policy.jsonl"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional subset of task class names to evaluate.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    validated_v7_tasks = ["WeighIngredients", "BreadSelection"]
    target_tasks = args.tasks or validated_v7_tasks
    target_ep_set = set(range(args.start_episode, args.start_episode + args.episodes))
    done_pairs = set()
    if args.output.exists():
        kept = []
        for line in args.output.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if not r.get("crash"):
                    kept.append(line)
                    done_pairs.add((r.get("task"), int(r.get("episode", -1))))
            except Exception:
                pass
        args.output.write_text("\n".join(kept) + ("\n" if kept else ""))

    jobs = []
    for t in target_tasks:
        rem_eps = [ep for ep in sorted(target_ep_set) if (t, ep) not in done_pairs]
        if rem_eps:
            jobs.append((t, rem_eps, str(args.output)))
    print(f"Running RoboCasa365 v7 Skill+Navigation Code Policy across {len(jobs)} tasks...", flush=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for msg in pool.map(_worker_run, jobs):
            print(msg, flush=True)
    print("ROBOCASA_CODE_POLICY_V7_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
