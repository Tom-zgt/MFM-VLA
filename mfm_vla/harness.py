"""A LIBERO wrapper that only hands out what a real robot could measure.

The point of this project is a zero-training, vision-only policy, so the
temptation to peek at `sim.data` has to be removed structurally rather than by
discipline. `TaskEnv` therefore never returns object poses: a policy sees RGB-D
from the same two cameras VLA papers use (`agentview` + wrist), plus proprio
and the language string.

`frontview` / `sideview` / `galleryview` stay off. Those are spectator cameras
the robot does not have.

`step()` advances physics; `observe()` renders both policy cameras. With NVIDIA
EGL a look is a couple of milliseconds, so the controller can see after every
action the way a hand-eye loop does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import libero_compat  # noqa: F401  (must precede libero/mujoco imports)
import mujoco
import numpy as np
import robosuite.utils.camera_utils as CU

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from libero_pro import get_libero_pro_task, is_libero_pro_suite, load_init_states

# Proprioception a real Panda publishes. Everything else robosuite exposes
# (object poses, object-to-eef vectors) is privileged and must not leak through.
_PROPRIO_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_vel",
)

# Same pair OpenVLA / π0 / GR00T eat on LIBERO: workspace cam + wrist.
AGENT_CAM = "agentview"
WRIST_CAM = "robot0_eye_in_hand"
POLICY_CAMERAS = (AGENT_CAM, WRIST_CAM)
DEFAULT_CAMERAS = POLICY_CAMERAS
VIDEO_CAM = AGENT_CAM


@dataclass
class Proprio:
    """Cheap, render-free feedback for the inner control loop."""

    t: int
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper_qpos: np.ndarray
    joint_pos: np.ndarray

    @property
    def gripper_opening(self) -> float:
        """Finger separation in metres; near zero means the hand closed on nothing."""
        return float(abs(self.gripper_qpos[0] - self.gripper_qpos[1]))


@dataclass
class Observation:
    """What the policy is allowed to see at one instant."""

    t: int
    instruction: str
    rgb: Dict[str, np.ndarray]  # cam -> (H, W, 3) uint8, top-down image order
    depth: Dict[str, np.ndarray]  # cam -> (H, W) float32, metres
    intrinsics: Dict[str, np.ndarray]  # cam -> (3, 3)
    cam_to_world: Dict[str, np.ndarray]  # cam -> (4, 4)
    eef_pos: np.ndarray  # (3,) world frame
    eef_quat: np.ndarray  # (4,) xyzw, world frame
    gripper_qpos: np.ndarray  # (2,) finger joints
    joint_pos: np.ndarray  # (7,)
    joint_vel: np.ndarray  # (7,)

    def deproject(self, cam: str, uv: Sequence[float]) -> np.ndarray:
        """Back-project one (u, v) pixel through its depth to a world point.

        `cam_to_world` already carries robosuite's axis correction, so the
        camera frame here is the OpenCV one: +x right, +y down, +z into the
        scene. That makes this the textbook pinhole formula.
        """
        u, v = int(round(uv[0])), int(round(uv[1]))
        z = float(self.depth[cam][v, u])
        K = self.intrinsics[cam]
        p_cam = np.array([
            (u - K[0, 2]) * z / K[0, 0],
            (v - K[1, 2]) * z / K[1, 1],
            z,
            1.0,
        ])
        return (self.cam_to_world[cam] @ p_cam)[:3]

    def point_cloud(self, cam: str, stride: int = 1, with_rgb: bool = False):
        """Full organised cloud for `cam` as (N, 3) world-frame points.

        With `with_rgb`, also returns the matching (N, 3) uint8 colours.
        """
        d = self.depth[cam][::stride, ::stride]
        H, W = d.shape
        K = self.intrinsics[cam]
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        uu *= stride
        vv *= stride
        xc = (uu - K[0, 2]) * d / K[0, 0]
        yc = (vv - K[1, 2]) * d / K[1, 1]
        pts = np.stack([xc, yc, d, np.ones_like(d)], axis=-1).reshape(-1, 4)
        world = (pts @ self.cam_to_world[cam].T)[:, :3]
        if not with_rgb:
            return world
        return world, self.rgb[cam][::stride, ::stride].reshape(-1, 3)

    def project(self, cam: str, world_xyz: Sequence[float]) -> np.ndarray:
        """Forward-project a world point to (u, v) pixels. Inverse of deproject."""
        T = np.linalg.inv(self.cam_to_world[cam])
        p_cam = (T @ np.append(np.asarray(world_xyz, dtype=np.float64), 1.0))[:3]
        uvw = self.intrinsics[cam] @ p_cam
        return uvw[:2] / uvw[2]


class TaskEnv:
    """One LIBERO task, restricted to real-robot-observable quantities."""

    def __init__(
        self,
        suite_name: str = "libero_spatial",
        task_id: int = 0,
        img_size: int = 128,
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        control_freq: int = 20,
        seed: int = 0,
        env_horizon: int = 1000,
        bddl_path: Optional[str] = None,
    ):
        self.suite_name = suite_name
        self.task_id = task_id
        self.seed = int(seed)
        # Evaluation/scripted policies may stop low-level motion as soon as a
        # benchmark predicate becomes true. Online visual-brain runs disable
        # this so hidden BDDL predicates cannot influence control.
        self.allow_goal_shortcut = True
        self.cameras = tuple(cameras)
        self.policy_cameras = tuple(
            c for c in POLICY_CAMERAS if c in self.cameras
        ) or (self.cameras[-1],)
        self.img_size = img_size

        if bddl_path is not None:
            # A generated task: no stored init states; layouts come from the
            # BDDL region samplers on reset().
            bddl = str(bddl_path)
            self._init_states = []
            instruction = None
        elif is_libero_pro_suite(suite_name):
            pro_task = get_libero_pro_task(suite_name, task_id)
            bddl = str(pro_task.bddl_path)
            self._init_states = load_init_states(pro_task)
            instruction = None
        else:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            task = suite.get_task(task_id)
            bddl = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            self._init_states = suite.get_task_init_states(task_id)
            instruction = task.language
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_names=list(self.cameras),
            camera_heights=img_size,
            camera_widths=img_size,
            control_freq=control_freq,
            horizon=env_horizon,
            # Rendering happens in observe(), not on every step.
            use_camera_obs=False,
            # A hard reset rebuilds sim and GL context every episode, which is
            # slow and invalidates cached handles; init states fully specify
            # qpos/qvel anyway.
            hard_reset=False,
        )
        # LIBERO-PRO task files intentionally retain the original filename, so
        # their changed instruction must come from BDDL. Standard LIBERO keeps
        # its official filename-derived prompt for checkpoint compatibility.
        self.instruction = (
            str(self._env.language_instruction) if instruction is None else instruction
        )
        self._env.seed(self.seed)
        self._env.reset()

        self._t = 0
        self._render_tuned = False
        self._K: Dict[str, np.ndarray] = {}
        self._frames: list = []
        self._wrist_frames: list = []
        self.step_hook = None
        self.step_hook_interval = 0
        self._next_hook_t = 0

    @property
    def _sim(self):
        # robosuite may rebuild sim on reset, so never cache this.
        return self._env.env.sim

    # ---------------------------------------------------------------- episode

    @property
    def n_init_states(self) -> int:
        return len(self._init_states)

    def reset(self, init_idx: int = 0, settle_steps: int = 15) -> Observation:
        self._env.reset()
        # A rebuilt sim brings a fresh render context and camera handles.
        self._render_tuned = False
        self._K.clear()
        self._env.set_init_state(self._init_states[init_idx])
        # set_init_state teleports qpos; let contacts resolve before acting.
        for _ in range(settle_steps):
            self._env.step(np.zeros(7))
        self._t = 0
        self._frames = []
        self._wrist_frames = []
        self._next_hook_t = int(self.step_hook_interval or 0)
        return self.observe()

    def step(self, action: np.ndarray, repeat: int = 1) -> None:
        """Advance physics. Renders nothing; call observe() when you need eyes."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for _ in range(repeat):
            self._env.step(a)
            self._t += 1
            if (self.step_hook is not None and self.step_hook_interval > 0
                    and self._t >= self._next_hook_t):
                self._next_hook_t = self._t + self.step_hook_interval
                self.step_hook(self)

    @property
    def success(self) -> bool:
        return bool(self._env.check_success())

    @property
    def t(self) -> int:
        return self._t

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------ observation

    def _tune_render_context(self) -> None:
        """Drop effects that cost a lot under llvmpipe and teach us nothing."""
        ctx = getattr(self._sim, "_render_context_offscreen", None)
        if ctx is None:
            return
        # Planar reflections roughly double render time on the software path.
        ctx.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        ctx.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self._render_tuned = True

    def _proprio(self) -> Dict[str, np.ndarray]:
        obs = self._env.env._get_observations()
        return {k: np.asarray(obs[k]).copy() for k in _PROPRIO_KEYS if k in obs}

    def proprio(self) -> Proprio:
        """Joint/gripper feedback with no rendering, for the inner control loop."""
        p = self._proprio()
        return Proprio(
            t=self._t,
            eef_pos=p["robot0_eef_pos"],
            eef_quat=p["robot0_eef_quat"],
            gripper_qpos=p["robot0_gripper_qpos"],
            joint_pos=p["robot0_joint_pos"],
        )

    @property
    def action_pos_scale(self) -> np.ndarray:
        """Metres of commanded end-effector offset per unit of action.

        The OSC controller maps action in [-1, 1] onto its own output range, so
        a P controller needs this to turn a position error into an action.
        """
        ctrl = self._env.env.robots[0].controller
        return np.abs(np.asarray(ctrl.output_max[:3], dtype=np.float64))

    @property
    def action_rot_scale(self) -> np.ndarray:
        ctrl = self._env.env.robots[0].controller
        return np.abs(np.asarray(ctrl.output_max[3:6], dtype=np.float64))

    def observe(
        self,
        cameras: Optional[Sequence[str]] = None,
        size: Optional[int] = None,
    ) -> Observation:
        """Render policy cameras, optionally at a one-call resolution override."""
        px = int(size or self.img_size)
        rgb, depth, K, T = {}, {}, {}, {}
        for cam in (tuple(cameras) if cameras is not None else self.policy_cameras):
            im, dp = self._sim.render(
                width=px, height=px, camera_name=cam, depth=True,
            )
            if not self._render_tuned:
                # The context only exists after the first render call.
                self._tune_render_context()
                im, dp = self._sim.render(
                    width=px, height=px, camera_name=cam, depth=True,
                )
            # MuJoCo returns bottom-up buffers; flip to conventional image order.
            rgb[cam] = im[::-1].copy()
            depth[cam] = CU.get_real_depth_map(self._sim, dp[::-1, :, None])[..., 0]

            if (cam, px) not in self._K:
                self._K[(cam, px)] = CU.get_camera_intrinsic_matrix(
                    self._sim, cam, px, px
                )
            K[cam] = self._K[(cam, px)]
            # Wrist camera moves, so extrinsics are re-read every observation.
            T[cam] = CU.get_camera_extrinsic_matrix(self._sim, cam)

        p = self._proprio()
        return Observation(
            t=self._t,
            instruction=self.instruction,
            rgb=rgb,
            depth=depth,
            intrinsics=K,
            cam_to_world=T,
            eef_pos=p["robot0_eef_pos"],
            eef_quat=p["robot0_eef_quat"],
            gripper_qpos=p["robot0_gripper_qpos"],
            joint_pos=p["robot0_joint_pos"],
            joint_vel=p["robot0_joint_vel"],
        )

    def robot_pixel_mask(self, cam: str, size: Optional[int] = None) -> np.ndarray:
        """Render only the known robot embodiment as a boolean image mask.

        The mask is where the depth buffer changes when the known embodiment
        geometry is hidden. (MuJoCo ID-colour segmentation is not honoured by
        the NVIDIA EGL stack here.) A real deployment obtains the equivalent
        mask by rendering the known URDF/CAD model at measured joint positions
        into the calibrated camera. Object labels and poses are never returned.
        """
        px = int(size or self.img_size)
        model = self._sim.model
        robot_geoms = np.array(
            [
                index
                for index in range(model.ngeom)
                if (model.geom_id2name(index) or "").startswith(
                    ("robot0_", "gripper0_", "mount0_")
                )
            ],
            dtype=int,
        )
        _, with_robot = self._sim.render(
            width=px, height=px, camera_name=cam, depth=True
        )
        hidden_group = 5
        vopt = self._sim._render_context_offscreen.vopt
        groups = model.geom_group[robot_geoms].copy()
        group_visible = int(vopt.geomgroup[hidden_group])
        model.geom_group[robot_geoms] = hidden_group
        vopt.geomgroup[hidden_group] = 0
        try:
            _, without_robot = self._sim.render(
                width=px, height=px, camera_name=cam, depth=True
            )
        finally:
            model.geom_group[robot_geoms] = groups
            vopt.geomgroup[hidden_group] = group_visible
        return (np.abs(with_robot - without_robot) > 1e-7)[::-1]

    # ------------------------------------------------------------------ video

    def grab_frame(self, cam: str = VIDEO_CAM) -> None:
        """Record workspace and wrist. `cam` is ignored; both policy cams go in."""
        del cam
        for name, bucket in ((AGENT_CAM, self._frames), (WRIST_CAM, self._wrist_frames)):
            im = self._sim.render(width=256, height=256, camera_name=name)
            bucket.append(im[::-1].copy())

    def save_video(self, path: str, fps: int = 20) -> Optional[str]:
        """Write agentview `path`, plus `_wrist.mp4` and a side-by-side `_both.mp4`."""
        if not self._frames:
            return None
        import imageio

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        imageio.mimsave(path, self._frames, fps=fps)
        stem, ext = os.path.splitext(path)
        if self._wrist_frames:
            imageio.mimsave(stem + "_wrist" + ext, self._wrist_frames, fps=fps)
            n = min(len(self._frames), len(self._wrist_frames))
            if n:
                both = [
                    np.concatenate([self._frames[i], self._wrist_frames[i]], axis=1)
                    for i in range(n)
                ]
                imageio.mimsave(stem + "_both" + ext, both, fps=fps)
        return path

    # ------------------------------------------------------- eval-only escape

    def _privileged_object_poses(self) -> Dict[str, np.ndarray]:
        """Ground truth, for validating the harness itself. Never call from a policy."""
        obs = self._env.env._get_observations()
        return {
            k: np.asarray(v).copy()
            for k, v in obs.items()
            if k.endswith("_pos") and not k.startswith("robot0")
        }


def list_tasks(suite_name: str = "libero_spatial") -> list:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    return [suite.get_task(i).language for i in range(suite.n_tasks)]
