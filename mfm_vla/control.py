"""Closed-loop Cartesian primitives: one action, then both cameras, then the next.

Proprio still turns a Cartesian error into an OSC command. After every physics
step we re-render `agentview` + wrist so the caller can move the target the
way a hand-eye loop does. NVIDIA EGL makes that look cheap (~2 ms).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from harness import TaskEnv

# robosuite PandaGripper.format_action: -1 opens, +1 closes.
OPEN = -1.0
CLOSE = 1.0

LookFn = Callable[[object], Optional[np.ndarray]]


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert a world-frame xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    n = max(float(np.linalg.norm([x, y, z, w])), 1e-12)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def rotvec_to_mat(rotvec: np.ndarray) -> np.ndarray:
    """Convert an axis-angle rotation vector to a rotation matrix."""
    v = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    axis = v / theta
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def eef_orientation(env: TaskEnv) -> np.ndarray:
    return quat_xyzw_to_mat(env.proprio().eef_quat)


def nominal_orientation(env: TaskEnv) -> np.ndarray:
    """The reset wrist orientation used by ordinary top-down manipulation."""
    if not hasattr(env, "_nominal_eef_orientation"):
        env._nominal_eef_orientation = eef_orientation(env)
    return np.asarray(env._nominal_eef_orientation, dtype=np.float64).copy()


def orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-frame axis-angle error taking `current` toward `target`."""
    return 0.5 * sum(
        np.cross(current[:, i], target[:, i]) for i in range(3)
    )


def orientation_for_approach(
    env: TaskEnv, direction: np.ndarray, reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Full wrist orientation whose approach axis points along `direction`."""
    ref = nominal_orientation(env) if reference is None else np.asarray(reference)
    source = ref[:, 2]
    target = np.asarray(direction, dtype=np.float64)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    cross = np.cross(source, target)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    s = float(np.linalg.norm(cross))
    if s < 1e-8:
        if dot > 0:
            return ref.copy()
        axis = ref[:, 0]
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return (np.eye(3) + 2 * K @ K) @ ref
    axis = cross / s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    align = np.eye(3) + K * s + K @ K * (1 - dot)
    return align @ ref


def _max_step_m(err: np.ndarray, eef_z: float, table_z: Optional[float]) -> float:
    """How far this action may travel, in metres.

    OSC still caps one unit at 5 cm. We take that only when the hand is high
    and far; near the table or the target the last look should be able to
    abort a shove, so the step has to be smaller than the object.
    """
    dist = float(np.linalg.norm(err))
    height = 1.0 if table_z is None else float(eef_z - table_z)
    if dist < 0.025 or height < 0.04:
        return 0.012
    if dist < 0.08 or height < 0.10:
        return 0.025
    return 0.05


def move_to(
    env: TaskEnv,
    target: np.ndarray,
    gripper: float = OPEN,
    tol: float = 0.012,
    max_steps: int = 100,
    kp: float = 1.6,
    record_every: int = 3,
    look_every: int = 1,
    on_look: Optional[LookFn] = None,
    table_z: Optional[float] = None,
    target_ori: Optional[np.ndarray] = None,
    ori_tol: float = 0.06,
    ori_kp: float = 1.8,
) -> bool:
    """Drive the grip site to a full world-frame 6D pose.

    After each action, `observe()` both policy cameras. If `on_look` returns a
    new xyz, that becomes the target for the next action. Step length shrinks
    as the hand gets close to the target or the table. If no orientation is
    supplied, actively hold the nominal top-down wrist pose.
    """
    if table_z is None:
        table_z = getattr(env, "table_z", None)
    scale = env.action_pos_scale
    target = np.asarray(target, dtype=np.float64).copy()
    desired_ori = (nominal_orientation(env) if target_ori is None
                   else np.asarray(target_ori, dtype=np.float64).copy())
    last_pos = None
    stall = 0
    for i in range(max_steps):
        p = env.proprio()
        err = target - p.eef_pos
        ori_err = orientation_error(quat_xyzw_to_mat(p.eef_quat), desired_ori)
        if np.linalg.norm(err) < tol and np.linalg.norm(ori_err) < ori_tol:
            if i % record_every == 0:
                env.grab_frame()
            return True
        # Wrist can walk the target past the Panda's reach (front-right
        # cookie-side bowls). If the hand has sat still within 4.5 cm,
        # close / place from here instead of burning the step budget.
        if last_pos is not None and np.linalg.norm(p.eef_pos - last_pos) < 0.002:
            stall += 1
        else:
            stall = 0
        last_pos = p.eef_pos.copy()
        if (stall >= 8 and np.linalg.norm(err) < 0.045
                and np.linalg.norm(ori_err) < ori_tol):
            return True
        raw = kp * err / scale
        cap = _max_step_m(err, float(p.eef_pos[2]), table_z) / scale
        act = np.zeros(7)
        act[:3] = np.clip(raw, -cap, cap)
        act[3:6] = np.clip(ori_kp * ori_err, -0.4, 0.4)
        act[6] = gripper
        env.step(act)
        tr = getattr(env, "trace", None)
        if tr is not None:
            tr.act(env.t, env.proprio().eef_pos, act, target)
        if look_every and i % look_every == 0:
            obs = env.observe()
            if on_look is not None:
                updated = on_look(obs)
                if updated is not None:
                    target = np.asarray(updated, dtype=np.float64)
        if i % record_every == 0:
            env.grab_frame()
        if getattr(env, "allow_goal_shortcut", True) and env.success:
            return True
    return False


def hold(
    env: TaskEnv,
    gripper: float,
    steps: int = 12,
    on_look: Optional[LookFn] = None,
    target_ori: Optional[np.ndarray] = None,
) -> None:
    desired_ori = (nominal_orientation(env) if target_ori is None
                   else np.asarray(target_ori, dtype=np.float64))
    for i in range(steps):
        act = np.zeros(7)
        ori_err = orientation_error(eef_orientation(env), desired_ori)
        act[3:6] = np.clip(1.8 * ori_err, -0.4, 0.4)
        act[6] = gripper
        env.step(act)
        tr = getattr(env, "trace", None)
        if tr is not None:
            tr.act(env.t, env.proprio().eef_pos, act, note="hold")
        obs = env.observe()
        if on_look is not None:
            on_look(obs)
        if i % 3 == 0:
            env.grab_frame()


def grasped(env: TaskEnv, min_opening: float = 0.008) -> bool:
    """True if the fingers stopped on something instead of closing fully."""
    return env.proprio().gripper_opening > min_opening
