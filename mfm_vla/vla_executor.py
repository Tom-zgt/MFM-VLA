"""Chunked VLA execution through an openpi-compatible policy server.

The VLA owns fast 20 Hz Cartesian control.  The online code agent remains the
supervisor: ``TaskEnv.step_hook`` runs while actions are being applied, so a
brain checkpoint can continue the current rollout or interrupt it with a new
subgoal.  No simulator predicates or privileged object poses are read here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Optional

import numpy as np
from openpi_client import image_tools
from openpi_client import msgpack_numpy
import websockets.sync.client

from skill_schema import BrainDecision
from harness import AGENT_CAM, TaskEnv, WRIST_CAM
from policy import EpisodeResult


@dataclass(frozen=True)
class VLAConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    max_skill_steps: int = 200
    connect_timeout_s: float = 15.0
    inference_timeout_s: float = 120.0
    record_every: int = 3
    prompt_mode: str = "task"

    def __post_init__(self) -> None:
        if self.resize_size < 64:
            raise ValueError("VLA resize_size must be at least 64")
        if self.replan_steps < 1:
            raise ValueError("VLA replan_steps must be positive")
        if self.max_skill_steps < self.replan_steps:
            raise ValueError("VLA max_skill_steps must cover one replan chunk")
        if self.prompt_mode not in {"task", "decision"}:
            raise ValueError("VLA prompt_mode must be task or decision")


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Match openpi's official LIBERO preprocessing (quaternion is xyzw)."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(0.0, 1.0 - float(quat[3]) ** 2))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(float(quat[3])) / den)


def decision_prompt(decision: BrainDecision) -> str:
    """Turn the supervisor's structured subgoal into a VLA instruction."""
    if decision.action == "pick_place":
        prep = "in" if decision.relation == "in" else "on"
        return f"put the {decision.target} {prep} the {decision.receptacle}"
    if decision.action == "open":
        return f"open {decision.target}"
    if decision.action == "close":
        return f"close {decision.target}"
    if decision.action == "turn_on":
        return f"turn on {decision.target}"
    raise ValueError(f"decision is not executable by the VLA: {decision.action}")


class OpenPiPolicyClient:
    """Small fail-fast variant of openpi's websocket client."""

    def __init__(self, config: VLAConfig):
        self._packer = msgpack_numpy.Packer()
        uri = config.host if config.host.startswith("ws") else f"ws://{config.host}"
        if config.port and not config.host.startswith("ws"):
            uri += f":{config.port}"
        try:
            self._ws = websockets.sync.client.connect(
                uri,
                compression=None,
                max_size=None,
                open_timeout=config.connect_timeout_s,
                close_timeout=2,
                ping_interval=None,
                ping_timeout=None,
                legacy=True,
            )
            self.metadata = msgpack_numpy.unpackb(
                self._ws.recv(timeout=config.connect_timeout_s)
            )
        except Exception as exc:
            raise RuntimeError(f"cannot connect to VLA policy server at {uri}: {exc}") from exc
        self._inference_timeout_s = config.inference_timeout_s

    def infer(self, observation: dict) -> dict:
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv(timeout=self._inference_timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"VLA inference server returned an error: {response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._ws.close()


class OpenPiVLAExecutor:
    """Receding-horizon π0.5 executor that remains interruptible each step."""

    backend_name = "openpi"

    def __init__(
        self,
        config: VLAConfig = VLAConfig(),
        policy_client: Optional[object] = None,
    ):
        self.config = config
        self._client = policy_client or OpenPiPolicyClient(config)
        self.inference_calls = 0
        # Optional safety layer applied to every action just before it is executed.
        self.action_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None
        # Optional visual prompt: (env, observation, 256px policy image) -> marked image.
        self.image_marker: Optional[Callable[[TaskEnv, object, np.ndarray], np.ndarray]] = None
        # Optional dynamic dense-caption provider: (env, observation, default_prompt) -> prompt.
        self.prompt_builder: Optional[Callable[[TaskEnv, object, str], str]] = None

    def _observation(self, env: TaskEnv, prompt: str) -> dict:
        # Official π0.5-LIBERO rotates robosuite's raw OpenGL buffers by 180
        # degrees. TaskEnv.observe() has already flipped those buffers
        # vertically into conventional image order, so only the remaining
        # horizontal flip belongs here.
        obs = env.observe(size=256)
        if self.prompt_builder is not None:
            prompt = self.prompt_builder(env, obs, prompt)
        image = np.ascontiguousarray(obs.rgb[AGENT_CAM][:, ::-1])
        wrist = np.ascontiguousarray(obs.rgb[WRIST_CAM][:, ::-1])
        if self.image_marker is not None:
            image = self.image_marker(env, obs, image)
        image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image, self.config.resize_size, self.config.resize_size)
        )
        wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, self.config.resize_size, self.config.resize_size)
        )
        state = np.concatenate(
            (obs.eef_pos, _quat2axisangle(obs.eef_quat), obs.gripper_qpos)
        )
        return {
            "observation/image": image,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": prompt,
        }

    def execute(self, env: TaskEnv, decision: BrainDecision) -> EpisodeResult:
        # The released pi05_libero checkpoint was trained on LIBERO's exact
        # task language. Keep that distribution by default; decision mode is
        # available for VLAs specifically trained to execute arbitrary
        # supervisor-generated subgoals.
        prompt = (
            env.instruction
            if self.config.prompt_mode == "task"
            else decision_prompt(decision)
        )
        return self._execute_prompt(env, prompt, self.config.max_skill_steps)

    def execute_task(
        self,
        env: TaskEnv,
        max_steps: Optional[int] = None,
        benchmark_stop_on_success: bool = False,
        step_callback: Optional[
            Callable[[TaskEnv, np.ndarray, int], None]
        ] = None,
    ) -> EpisodeResult:
        """Run the exact task prompt without a code-agent supervisor.

        ``benchmark_stop_on_success`` matches the official LIBERO evaluator.
        It is deliberately opt-in because the predicate is privileged and must
        never become a policy input or a production completion detector.
        """
        return self._execute_prompt(
            env,
            env.instruction,
            int(max_steps or self.config.max_skill_steps),
            benchmark_stop_on_success=benchmark_stop_on_success,
            step_callback=step_callback,
        )

    def execute_prompt(
        self,
        env: TaskEnv,
        prompt: str,
        max_steps: int,
        benchmark_stop_on_success: bool = False,
        step_callback: Optional[
            Callable[[TaskEnv, np.ndarray, int], None]
        ] = None,
    ) -> EpisodeResult:
        """Run an agent-written sub-instruction (G3b task programs)."""
        return self._execute_prompt(
            env,
            prompt,
            int(max_steps),
            benchmark_stop_on_success=benchmark_stop_on_success,
            step_callback=step_callback,
        )

    def propose_task(self, env: TaskEnv) -> np.ndarray:
        """Return one fresh, validated π0.5 LIBERO action proposal."""
        response = self._client.infer(self._observation(env, env.instruction))
        self.inference_calls += 1
        actions = np.asarray(response.get("actions"), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise RuntimeError(
                f"VLA returned actions with shape {actions.shape}, expected (T, 7)"
            )
        if not len(actions):
            raise RuntimeError("VLA returned an empty action proposal")
        if not np.isfinite(actions).all():
            raise RuntimeError("VLA returned non-finite actions")
        return np.clip(actions, -1.0, 1.0)

    def _execute_prompt(
        self,
        env: TaskEnv,
        prompt: str,
        max_steps: int,
        benchmark_stop_on_success: bool = False,
        step_callback: Optional[
            Callable[[TaskEnv, np.ndarray, int], None]
        ] = None,
    ) -> EpisodeResult:
        action_plan: deque[np.ndarray] = deque()
        started = env.t
        while env.t - started < max_steps:
            if not action_plan:
                if prompt == env.instruction:
                    actions = self.propose_task(env)
                else:
                    response = self._client.infer(self._observation(env, prompt))
                    self.inference_calls += 1
                    actions = np.asarray(response.get("actions"), dtype=np.float64)
                    if actions.ndim != 2 or actions.shape[1] != 7:
                        raise RuntimeError(
                            f"VLA returned actions with shape {actions.shape}, "
                            "expected (T, 7)"
                        )
                    if not np.isfinite(actions).all():
                        raise RuntimeError("VLA returned non-finite actions")
                    actions = np.clip(actions, -1.0, 1.0)
                if len(actions) < self.config.replan_steps:
                    raise RuntimeError(
                        f"VLA returned {len(actions)} actions, fewer than "
                        f"replan_steps={self.config.replan_steps}"
                    )
                action_plan.extend(actions[: self.config.replan_steps])

            action = action_plan.popleft()
            if self.action_filter is not None:
                action = self.action_filter(action)
            env.step(action)
            if step_callback is not None:
                step_callback(env, action, self.inference_calls)
            if self.config.record_every and (env.t - started) % self.config.record_every == 0:
                env.grab_frame()
            if benchmark_stop_on_success and env.success:
                return EpisodeResult(
                    True,
                    env.t,
                    f"LIBERO benchmark predicate first passed after "
                    f"{env.t - started} VLA steps",
                )

        return EpisodeResult(
            bool(env.success) if benchmark_stop_on_success else False,
            env.t,
            f"VLA action budget reached after {env.t - started} steps; "
            f"inference_calls={self.inference_calls}",
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()
