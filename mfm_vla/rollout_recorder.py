"""Durable per-step traces and resumable checkpoints for LIBERO rollouts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
from typing import Any, Optional

import numpy as np
from PIL import Image

from harness import POLICY_CAMERAS, TaskEnv


def default_staging_root() -> Optional[Path]:
    """Local staging directory for rollouts; empty MFM_VLA_STAGING disables."""
    value = os.environ.get("MFM_VLA_STAGING", "/tmp/mfm_vla_staging")
    return Path(value) if value else None


class RolloutRecorder:
    """Persist observable state, actions, camera frames, and policy state.

    With ``staging_root`` every write goes to local disk and ``finalize``
    publishes the complete directory to ``root`` with one rename, so a remote
    network destination never holds a half-written episode. With
    ``sync_checkpoints`` each checkpoint is additionally mirrored to ``root`` so
    a resumable run survives loss of the local disk.
    """

    def __init__(
        self,
        root: Path,
        *,
        image_size: int = 256,
        fps: int = 20,
        staging_root: Optional[Path] = None,
        sync_checkpoints: bool = False,
    ):
        self.final_root = Path(root)
        if staging_root is None:
            self.root = self.final_root
        else:
            digest = hashlib.sha1(str(self.final_root.resolve()).encode()).hexdigest()[:12]
            self.root = Path(staging_root) / f"{self.final_root.name}-{digest}"
        self.staged = self.root != self.final_root
        self.sync_checkpoints = bool(sync_checkpoints) and self.staged
        self.image_size = int(image_size)
        self.fps = int(fps)
        self.frames_dir = self.root / "frames"
        self.trace_path = self.root / "steps.jsonl"
        self.metadata_path = self.root / "metadata.json"
        self.status_path = self.root / "status.json"
        self.checkpoint_path = self.root / "checkpoint.pkl"

    @staticmethod
    def _atomic_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @classmethod
    def _atomic_json(cls, path: Path, value: Any) -> None:
        cls._atomic_bytes(
            path,
            (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(),
        )

    def _mirror(self, source: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-r", "--inplace", "--delete", f"{source}/", f"{destination}/"],
            check=True,
        )

    def begin(self, metadata: dict, env: TaskEnv, *, resume: bool = False) -> None:
        if self.staged:
            final_checkpoint = self.final_root / "checkpoint.pkl"
            if resume and not self.checkpoint_path.exists() and final_checkpoint.exists():
                self._mirror(self.final_root, self.root)
            elif not resume and self.root.exists():
                # A leftover local stage from a crashed process is never
                # authoritative; the destination decides what exists.
                shutil.rmtree(self.root)
            if not resume and (self.final_root / "checkpoint.pkl").exists():
                raise FileExistsError(
                    f"rollout artifacts already exist at {self.final_root}; use --resume "
                    "or choose a different --artifact-dir"
                )
        self.root.mkdir(parents=True, exist_ok=True)
        if resume:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"no rollout checkpoint at {self.checkpoint_path}")
            return
        if self.checkpoint_path.exists() or self.trace_path.exists():
            raise FileExistsError(
                f"rollout artifacts already exist at {self.root}; use --resume "
                "or choose a different --artifact-dir"
            )
        self._atomic_json(self.metadata_path, metadata)
        self._atomic_json(self.status_path, {"state": "running", "step": 0})
        self.record_step(env, action=None, mode="initial", decision=-1)

    def save_checkpoint(self, state: dict) -> None:
        self._atomic_bytes(
            self.checkpoint_path, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        )
        self._atomic_json(
            self.status_path,
            {
                "state": state.get("stage", "running"),
                "step": state.get("step", 0),
                "next_decision": state.get("next_decision", 0),
            },
        )
        if self.sync_checkpoints:
            self._mirror(self.root, self.final_root)

    def load_checkpoint(self) -> dict:
        with self.checkpoint_path.open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("unsupported or corrupt rollout checkpoint")
        return value

    def prepare_resume(self, step: int) -> None:
        """Drop trace fragments newer than the last atomic policy checkpoint."""
        if self.trace_path.exists():
            kept = []
            for line in self.trace_path.read_text().splitlines():
                record = json.loads(line)
                if int(record["step"]) <= step:
                    kept.append(line)
            self._atomic_bytes(
                self.trace_path,
                (("\n".join(kept) + "\n") if kept else "").encode(),
            )
        for camera_dir in self.frames_dir.glob("*"):
            if not camera_dir.is_dir():
                continue
            for frame in camera_dir.glob("*.jpg"):
                if int(frame.stem) > step:
                    frame.unlink()

    def replay(self, env: TaskEnv, checkpoint: dict) -> None:
        for action in checkpoint.get("actions", []):
            env.step(np.asarray(action, dtype=np.float64))
        expected = int(checkpoint.get("step", len(checkpoint.get("actions", []))))
        if env.t != expected:
            raise RuntimeError(f"checkpoint replay reached step {env.t}, expected {expected}")

    def actions_through(self, step: Optional[int] = None) -> list[list[float]]:
        """Load the durable action prefix, optionally ending at a chosen step."""
        actions = []
        for line in self.trace_path.read_text().splitlines():
            record = json.loads(line)
            if step is not None and int(record["step"]) > step:
                break
            if record.get("action") is not None:
                actions.append(record["action"])
        if step is not None and len(actions) != step:
            raise ValueError(
                f"trace has {len(actions)} actions through requested step {step}"
            )
        return actions

    def record_step(
        self,
        env: TaskEnv,
        *,
        action: Optional[np.ndarray],
        mode: str,
        decision: int,
    ) -> None:
        obs = env.observe(size=self.image_size)
        frame_paths = {}
        for camera in POLICY_CAMERAS:
            if camera not in obs.rgb:
                continue
            relative = Path("frames") / camera / f"{env.t:06d}.jpg"
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            Image.fromarray(obs.rgb[camera]).save(temporary, format="JPEG", quality=90)
            os.replace(temporary, destination)
            frame_paths[camera] = str(relative)

        record = {
            "step": env.t,
            "decision": decision,
            "mode": mode,
            "action": None if action is None else np.asarray(action).round(7).tolist(),
            "eef_position": obs.eef_pos.round(7).tolist(),
            "eef_quaternion_xyzw": obs.eef_quat.round(7).tolist(),
            "gripper_qpos": obs.gripper_qpos.round(7).tolist(),
            "joint_position": obs.joint_pos.round(7).tolist(),
            "joint_velocity": obs.joint_vel.round(7).tolist(),
            "frames": frame_paths,
        }
        with self.trace_path.open("a") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def finalize(self, state: str, **details: Any) -> dict[str, str]:
        self._atomic_json(self.status_path, {"state": state, **details})
        videos: dict[str, str] = {}
        try:
            import imageio.v2 as imageio

            def write_video(destination: Path, frames: list[np.ndarray]) -> None:
                # Network filesystems may not implement random-access seek operations
                # needed to finalize MP4 trailers. Encode on local disk, then copy
                # the closed file sequentially.
                with tempfile.TemporaryDirectory() as temporary:
                    local = Path(temporary) / destination.name
                    imageio.mimsave(local, frames, fps=self.fps)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(local, destination)

            camera_frames: dict[str, list[Path]] = {}
            for camera in POLICY_CAMERAS:
                frames = sorted((self.frames_dir / camera).glob("*.jpg"))
                if not frames:
                    continue
                destination = self.root / f"{camera}.mp4"
                write_video(
                    destination,
                    [imageio.imread(frame) for frame in frames],
                )
                videos[camera] = str(destination)
                camera_frames[camera] = frames
            if all(camera in camera_frames for camera in POLICY_CAMERAS):
                first, second = (camera_frames[camera] for camera in POLICY_CAMERAS)
                destination = self.root / "both.mp4"
                write_video(
                    destination,
                    [
                        np.concatenate(
                            [imageio.imread(left), imageio.imread(right)], axis=1
                        )
                        for left, right in zip(first, second)
                    ],
                )
                videos["both"] = str(destination)
        except Exception as exc:
            self._atomic_json(
                self.status_path,
                {"state": state, **details, "video_error": f"{type(exc).__name__}: {exc}"},
            )
        if self.staged:
            self.publish()
            videos = {
                name: str(self.final_root / Path(path).name) for name, path in videos.items()
            }
        return videos

    def publish(self) -> None:
        """Move the complete local stage to the destination in one rename."""
        partial = self.final_root.with_name(
            f".{self.final_root.name}.partial-{os.getpid()}"
        )
        if partial.exists():
            shutil.rmtree(partial)
        partial.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.root, partial, copy_function=shutil.copyfile)
        if self.final_root.exists():
            shutil.rmtree(self.final_root)
        os.rename(partial, self.final_root)
        shutil.rmtree(self.root)
