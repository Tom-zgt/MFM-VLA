"""Platform adapter: everything the generic agent may touch on a robot.

The agent sees calibrated RGB cameras, proprioception, a VLA, and one extra
primitive, ``move_to``: contact-free end-effector motion that stops as soon as
progress stalls. A real robot or another benchmark replaces this module only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from generic_config import MoveConfig

HERE = Path(__file__).resolve().parent
LIBERO_CALIBRATION = HERE.parent / "configs" / "adapters" / "libero.json"

GRIPPER_COMMAND = {"open": -1.0, "close": 1.0}


@dataclass(frozen=True)
class Calibration:
    """Per-setup workspace calibration, measured once like camera extrinsics."""

    support_eef_z: float
    source: str


class GraspMonitor:
    """Proprioceptive grasp-and-lift detector: closed for a while, fingers apart, hand risen."""

    def __init__(self, adapter, close_steps: int = 5, rise_m: float = 0.03,
                 settle_m: float | None = None, settle_steps: int = 2, from_lowest: bool = False):
        """``settle_m``: the fingers must also have stopped closing (opening changed by
    less than this over ``settle_steps`` steps); a missed grasp keeps closing.
    ``from_lowest``: measure the rise from the lowest hand height while closed rather
    than from where closing began (the policy often starts closing while descending)."""
        self.adapter = adapter
        self.from_lowest = from_lowest
        self.close_steps = close_steps
        self.rise_m = rise_m
        self.settle_m = settle_m
        self.settle_steps = settle_steps
        self.streak = 0
        self.close_z: float | None = None
        self.openings: list[float] = []

    def update(self, action: np.ndarray) -> bool:
        proprio = self.adapter.proprio()
        eef_z = float(proprio.eef_pos[2])
        self.openings = (self.openings + [float(proprio.gripper_opening)])[-(self.settle_steps + 1):]
        if action[6] > 0:
            if self.streak == 0 or (self.from_lowest and eef_z < self.close_z):
                self.close_z = eef_z
            self.streak += 1
        else:
            self.streak, self.close_z = 0, None
        settled = self.settle_m is None or (
            len(self.openings) > self.settle_steps
            and abs(self.openings[0] - self.openings[-1]) < self.settle_m
        )
        return (
            self.streak >= self.close_steps
            and self.adapter.holding()
            and settled
            and eef_z > self.close_z + self.rise_m
        )


def libero_calibration(suite: str, path: Path = LIBERO_CALIBRATION) -> Calibration:
    table = json.loads(path.read_text())["support_eef_z"]
    family = next((name for name in table if suite.startswith(name)), None)
    if family is None:
        raise KeyError(f"no support calibration for scene family of {suite}")
    return Calibration(float(table[family]["value"]), table[family]["source"])


class LiberoAdapter:
    """LIBERO / robosuite Panda behind the generic interface."""

    static_cameras = ("agentview",)
    movable_cameras = ("robot0_eye_in_hand",)
    # Panda hand finger separation: ~0 when closed on nothing, ~0.08 m fully open.
    hold_opening_m = (0.008, 0.075)

    def __init__(
        self,
        env,
        calibration: Calibration,
        move: MoveConfig,
        on_step: Optional[Callable[[np.ndarray, str], None]] = None,
    ):
        self.env = env
        self.calibration = calibration
        self.move = move
        self.on_step = on_step
        self.gripper_command = GRIPPER_COMMAND["open"]
        self.move_steps = 0
        self._last_motion = np.zeros(3)

    @property
    def instruction(self) -> str:
        return self.env.instruction

    @property
    def t(self) -> int:
        return self.env.t

    def observe(self, cameras: Sequence[str], size: int):
        return self.env.observe(cameras=list(cameras), size=size)

    def proprio(self):
        return self.env.proprio()

    def apply(self, action: np.ndarray, mode: str) -> None:
        self.env.step(action)
        self.gripper_command = float(np.sign(action[6]) or self.gripper_command)
        if self.on_step is not None:
            self.on_step(action, mode)

    def move_to(self, goal: np.ndarray, *, gripper: str = "hold", phase: str,
                max_action_change: float | None = None, max_action_norm: float | None = None,
                tolerance_m: float | None = None, quat_xyzw: np.ndarray | None = None,
                tolerance_rad: float = 0.1) -> dict:
        """Straight free-space motion to ``goal``; returns early on stall.

        ``max_action_change`` limits how much the commanded motion may change per
        step (per axis), so a held object is not shaken loose at corners.
        ``tolerance_m`` overrides the arrival tolerance, for waypoints that need no precision.
        ``quat_xyzw``: also turn the hand to this orientation (world frame) on the way;
        arrival then needs both tolerances.
        """
        command = self.gripper_command if gripper == "hold" else GRIPPER_COMMAND[gripper]
        goal = np.asarray(goal, dtype=np.float64)
        cfg = self.move
        tolerance = cfg.tolerance_m if tolerance_m is None else tolerance_m
        residuals: list[float] = []
        stalled = False
        start = self.t
        if max_action_change is None:
            self._last_motion = np.zeros(3)
        target_rot = None if quat_xyzw is None else Rotation.from_quat(quat_xyzw)
        turn = np.zeros(3)
        for _ in range(cfg.max_steps):
            proprio = self.proprio()
            error = goal - proprio.eef_pos
            residual = float(np.linalg.norm(error))
            if target_rot is not None:
                turn = (target_rot * Rotation.from_quat(proprio.eef_quat).inv()).as_rotvec()
            if residual <= tolerance and np.linalg.norm(turn) <= tolerance_rad:
                break
            residuals.append(residual)
            if (
                len(residuals) > cfg.stall_window
                and residual > cfg.stall_min_residual_m
                and residuals[-cfg.stall_window - 1] - residual < cfg.stall_progress_m
            ):
                stalled = True
                break
            action = np.zeros(7, dtype=np.float64)
            action[:3] = np.clip(cfg.gain * error / self.env.action_pos_scale, -1.0, 1.0)
            cap = max_action_norm
            if max_action_change is not None and cfg.brake:
                stop = float(np.sqrt(2.0 * max_action_change * residual / cfg.metres_per_step_per_unit))
                cap = stop if cap is None else min(cap, stop)
            if cap is not None:
                norm = float(np.linalg.norm(action[:3]))
                if norm > cap:
                    action[:3] *= cap / norm
            if max_action_change is not None:
                action[:3] = self._last_motion + np.clip(
                    action[:3] - self._last_motion, -max_action_change, max_action_change)
                self._last_motion = action[:3].copy()
            if target_rot is not None:
                action[3:6] = np.clip(cfg.gain * turn / self.env.action_rot_scale, -1.0, 1.0)
            action[6] = command
            self.apply(action, f"move_{phase}")
        self.move_steps += self.t - start
        return {
            "phase": phase,
            "goal": np.round(goal, 4).tolist(),
            "stalled": stalled,
            "residual_m": round(float(np.linalg.norm(goal - self.proprio().eef_pos)), 4),
            "steps": self.t - start,
        }

    def _arm_points(self):
        data = self.env._sim.data
        return (data.get_body_xpos("robot0_link2").copy(), data.get_body_xpos("robot0_link4").copy(),
                data.get_body_xpos("robot0_link6").copy())

    def max_reach_m(self) -> float:
        """Shoulder-to-wrist distance with the arm fully extended. Past it the elbow
        flips onto the over-extended side, where pulling the hand back would need the
        elbow joint beyond its limit, so Cartesian control stalls there."""
        shoulder, elbow, wrist = self._arm_points()
        return float(np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow))

    def reach_slack_m(self, goal: np.ndarray) -> float:
        """How far inside the arm's reach the wrist would be with the hand at ``goal``
        in its current orientation."""
        shoulder, _, wrist = self._arm_points()
        wrist_goal = np.asarray(goal, dtype=np.float64) + wrist - self.proprio().eef_pos
        return self.max_reach_m() - float(np.linalg.norm(wrist_goal - shoulder))

    def within_reach(self, goal: np.ndarray, margin_m: float) -> np.ndarray:
        """``goal`` pulled horizontally toward the shoulder until the wrist keeps
        ``margin_m`` of the arm's reach (current orientation); unchanged if it already does."""
        goal = np.asarray(goal, dtype=np.float64)
        shoulder, _, wrist = self._arm_points()
        v = goal + wrist - self.proprio().eef_pos - shoulder
        radius = self.max_reach_m() - margin_m
        if np.linalg.norm(v) <= radius:
            return goal.copy()
        u = np.array([v[0], v[1], 0.0]) / max(np.linalg.norm(v[:2]), 1e-9)
        along = float(v @ u)
        s = along - np.sqrt(max(along * along - float(v @ v) + radius * radius, 0.0))
        return goal - s * u

    def _elbow(self) -> tuple[float, float]:
        """Elbow (joint 4) angle, and the angle at which the arm is fully extended
        (from the link offsets either side of the elbow)."""
        model = self.env._sim.model
        joint = model.joint_name2id("robot0_joint4")
        angle = float(self.env._sim.data.qpos[model.jnt_qposadr[joint]])
        pos = lambda body: model.body_pos[model.body_name2id(body)]
        upper, offset, lower = pos("robot0_link3"), pos("robot0_link4"), pos("robot0_link5")
        straight = -(np.arctan2(abs(offset[0]), abs(upper[1])) + np.arctan2(abs(lower[0]), abs(lower[1])))
        return angle, float(straight)

    def elbow_past_straight_rad(self) -> float:
        """Positive when the elbow is bent past full extension (over-extended)."""
        angle, straight = self._elbow()
        return angle - straight

    def unfold(self, phase: str, max_push_m: float, within_rad: float, step_m: float = 0.01,
               **kw) -> dict | None:
        """From an over-extended elbow, push the hand outward (horizontally, away from the
        shoulder) until the elbow is back within ``within_rad`` of full extension, from
        where pulling the hand back works again. None if the elbow is not over-extended."""
        _, straight = self._elbow()
        if self._elbow()[0] <= straight + within_rad:
            return None
        start = self.proprio().eef_pos.copy()
        shoulder, _, wrist = self._arm_points()
        out = np.array([wrist[0] - shoulder[0], wrist[1] - shoulder[1], 0.0])
        out /= max(np.linalg.norm(out), 1e-9)
        t0, pushed = self.t, 0.0
        while pushed < max_push_m and self._elbow()[0] > straight + within_rad:
            pushed += step_m
            self.move_to(start + pushed * out, phase=phase, tolerance_m=step_m / 2, **kw)
        angle = self._elbow()[0]
        return {"phase": phase, "pushed_m": round(pushed, 3), "elbow_rad": round(angle, 3),
                "unfolded": angle <= straight + within_rad, "steps": self.t - t0}

    def reach_guard(self, action: np.ndarray, margin_m: float) -> np.ndarray:
        """Scale down the outward (shoulder-to-wrist) part of a commanded translation as
        the wrist comes within ``margin_m`` of full extension, so the arm never flips
        onto its over-extended side."""
        shoulder, _, wrist = self._arm_points()
        radius = float(np.linalg.norm(wrist - shoulder))
        slack = self.max_reach_m() - radius
        if slack >= margin_m:
            return action
        radial = (wrist - shoulder) / radius
        outward = float(action[:3] @ radial)
        if outward <= 0.0:
            return action
        action = np.array(action, dtype=np.float64)
        action[:3] -= (1.0 - max(slack, 0.0) / margin_m) * outward * radial
        return action

    def holding(self) -> bool:
        low, high = self.hold_opening_m
        return low < self.proprio().gripper_opening < high

    def move_over(self, goal: np.ndarray, *, gripper: str = "hold", phase: str,
                  traverse_z: float | None = None, max_action_change: float | None = None,
                  max_action_norm: float | None = None) -> list[dict]:
        """Rise to the highest of start, goal and ``traverse_z``, traverse, then descend;
        stops on any stall."""
        self._last_motion = np.zeros(3)
        eef = self.proprio().eef_pos
        goal = np.asarray(goal, dtype=np.float64)
        high = max(eef[2], goal[2], -np.inf if traverse_z is None else traverse_z)
        legs = [
            (np.array([eef[0], eef[1], high]), "rise"),
            (np.array([goal[0], goal[1], high]), "traverse"),
            (goal, "descend"),
        ]
        results = []
        for waypoint, leg in legs:
            result = self.move_to(waypoint, gripper=gripper, phase=f"{phase}_{leg}",
                                  max_action_change=max_action_change, max_action_norm=max_action_norm)
            results.append(result)
            if result["stalled"]:
                break
        return results
