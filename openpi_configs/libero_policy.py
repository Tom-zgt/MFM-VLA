"""LIBERO policy transforms with on-the-fly Dense Geometric Captioning (DGC) support."""

import dataclasses
import os
import pathlib
import pickle
import sys

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

_SIDECAR_CACHE: dict[str, list[dict]] = {}
_WORKER_RNG: dict[int, np.random.Generator] = {}


@dataclasses.dataclass(frozen=True)
class LiberoDenseCaptionTransform(transforms.DataTransformFn):
    """On-the-fly frame-level Dense Caption generator for `action_horizon = 10` chunks."""

    sidecar_path: str
    pos_noise_std_m: float = 0.025
    raw_prompt_prob: float = 0.10
    include_task_prefix_prob: float = 0.50
    causal_motion_prob: float = 0.50
    qualitative_prob: float = 0.20

    def __call__(self, data: dict) -> dict:
        if "episode_index" not in data or "frame_index" not in data:
            return data

        pid = os.getpid()
        rng = _WORKER_RNG.get(pid)
        if rng is None:
            seed = (pid * 1000003 + int(data.get("index", 0))) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            _WORKER_RNG[pid] = rng

        if rng.random() < self.raw_prompt_prob:
            return data

        episodes = _SIDECAR_CACHE.get(self.sidecar_path)
        if episodes is None:
            for cand in (
                os.environ.get("MFM_VLA_CODE_DIR", ""),
                str(pathlib.Path(__file__).resolve().parents[3] / "mfm_vla"),
                str(pathlib.Path(self.sidecar_path).resolve().parents[5] / "mfm_vla"),
            ):
                if cand and pathlib.Path(cand).is_dir() and cand not in sys.path:
                    sys.path.insert(0, cand)
            with open(self.sidecar_path, "rb") as fh:
                episodes = pickle.load(fh)
            _SIDECAR_CACHE[self.sidecar_path] = episodes

        import dense_caption

        ep_idx = int(data["episode_index"])
        frame_idx = int(data["frame_index"])
        if ep_idx < 0 or ep_idx >= len(episodes):
            return data

        meta = episodes[ep_idx]
        dc = dense_caption.render_dense_caption(
            meta,
            frame_idx,
            horizon=dense_caption.HORIZON,
            pos_noise_std_m=self.pos_noise_std_m,
            qualitative_only=bool(rng.random() < self.qualitative_prob),
            use_causal_motion=bool(rng.random() < self.causal_motion_prob),
            rng=rng,
        )
        caption = dc["caption"]
        if rng.random() < self.include_task_prefix_prob:
            task_str = str(data.get("prompt", meta.get("prompt", ""))).strip()
            if task_str:
                caption = f"task: {task_str}. {caption}"
        return {**data, "prompt": caption}


def make_libero_example() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}
