"""Fast PyTorch Dataset for RoboCasa365 Dense-Caption + G5j Hand-off Recovery fine-tuning."""

from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import pickle
import sys
from typing import List, SupportsIndex, Tuple

import numpy as np
import torch


_WORKER_RNG: dict[int, np.random.Generator] = {}
_WORKER_EP_CACHE: OrderedDict[str, dict] = OrderedDict()
_MAX_CACHED_EPS = 96


def _ensure_mfm_vla_path(data_root: Path) -> None:
    candidates = [
        os.environ.get("MFM_VLA_CODE_DIR", ""),
        str(Path(__file__).resolve().parents[3] / "mfm_vla"),
        str(data_root.resolve().parents[1] / "mfm_vla"),
    ]
    for c in candidates:
        if c and Path(c).is_dir() and c not in sys.path:
            sys.path.insert(0, c)


def _load_episode_cached(path_str: str) -> dict:
    ep = _WORKER_EP_CACHE.get(path_str)
    if ep is not None:
        _WORKER_EP_CACHE.move_to_end(path_str)
        return ep
    with open(path_str, "rb") as fh:
        ep = pickle.load(fh)
    _WORKER_EP_CACHE[path_str] = ep
    while len(_WORKER_EP_CACHE) > _MAX_CACHED_EPS:
        _WORKER_EP_CACHE.popitem(last=False)
    return ep


def _decode_rgb_jpg(buf: np.ndarray) -> np.ndarray:
    try:
        import cv2
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        import io
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(buf.tobytes())).convert("RGB"), dtype=np.uint8)


class RobocasaDenseDataset(torch.utils.data.Dataset):
    """Dataset returning OpenPI-formatted RoboCasa samples with on-the-fly Dense-Captions."""

    def __init__(
        self,
        data_root: str | Path,
        action_horizon: int = 50,
        raw_prompt_prob: float = 0.20,
        stage_prompt_prob: float = 0.30,
        pos_noise_std_m: float = 0.018,
    ) -> None:
        self.data_root = Path(data_root)
        self.action_horizon = int(action_horizon)
        self.raw_prompt_prob = float(raw_prompt_prob)
        self.stage_prompt_prob = float(stage_prompt_prob)
        self.pos_noise_std_m = float(pos_noise_std_m)

        _ensure_mfm_vla_path(self.data_root)

        human_files = sorted((self.data_root / "human_demos").glob("*.pkl"))
        handoff_files = sorted((self.data_root / "handoff_rollouts").glob("*.pkl"))
        all_files = [str(p) for p in (human_files + handoff_files)]
        if not all_files:
            raise FileNotFoundError(f"No .pkl episodes found under {self.data_root}")

        index_path = self.data_root / "meta_index.pkl"
        need_rebuild = True
        if index_path.exists():
            try:
                with index_path.open("rb") as fh:
                    idx_data = pickle.load(fh)
                if idx_data.get("ep_paths") == all_files:
                    self.ep_paths = idx_data["ep_paths"]
                    self.samples = idx_data["samples"]
                    need_rebuild = False
            except Exception:
                need_rebuild = True

        if need_rebuild:
            self.ep_paths: List[str] = all_files
            self.samples: List[Tuple[int, int, str]] = []
            for ep_idx, p_str in enumerate(self.ep_paths):
                with open(p_str, "rb") as fh:
                    ep = pickle.load(fh)
                T = len(ep["states"])
                source = str(ep.get("source", "human_demo"))
                meta = ep["dense_meta"]
                ho_set = set(meta.get("handoff_frames", []))
                stages = meta.get("stages", [])

                if source != "human_demo":
                    for t in range(T):
                        for _ in range(3):
                            self.samples.append((ep_idx, t, "handoff"))
                else:
                    for t in range(T):
                        self.samples.append((ep_idx, t, "auto"))
                        if t in ho_set:
                            in_later_stage = any(
                                stg["stage_idx"] >= 1 and stg["start_t"] <= t < stg["end_t"]
                                for stg in stages
                            )
                            extra_repeats = 4 if in_later_stage else 2
                            for _ in range(extra_repeats):
                                self.samples.append((ep_idx, t, "handoff"))

            tmp_idx = index_path.with_suffix(".tmp")
            with tmp_idx.open("wb") as fh:
                pickle.dump({"ep_paths": self.ep_paths, "samples": self.samples}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_idx.replace(index_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: SupportsIndex) -> dict:
        _ensure_mfm_vla_path(self.data_root)
        import robocasa_dense_caption as rdc

        idx = int(index) % len(self.samples)
        ep_idx, t, tag = self.samples[idx]
        ep = _load_episode_cached(self.ep_paths[ep_idx])
        T = len(ep["states"])
        t = int(np.clip(t, 0, T - 1))

        pid = os.getpid()
        rng = _WORKER_RNG.get(pid)
        if rng is None:
            seed = (pid * 1000003 + idx) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            _WORKER_RNG[pid] = rng

        left_rgb = _decode_rgb_jpg(ep["left_jpg"][t])
        wrist_rgb = _decode_rgb_jpg(ep["wrist_jpg"][t])
        right_rgb = _decode_rgb_jpg(ep["right_jpg"][t])

        state_t = np.asarray(ep["states"][t], dtype=np.float32)
        step_indices = np.minimum(t + np.arange(self.action_horizon), T - 1)
        actions_chunk = np.asarray(ep["actions"][step_indices], dtype=np.float32)

        u = rng.random()
        if tag == "handoff":
            mode = "raw" if u < 0.10 else ("stage" if u < 0.40 else "full")
        else:
            mode = "raw" if u < self.raw_prompt_prob else ("stage" if u < self.raw_prompt_prob + self.stage_prompt_prob else "full")

        prompt = rdc.render_robocasa_dense_prompt(
            ep["dense_meta"],
            t,
            mode=mode,
            pos_noise_std_m=self.pos_noise_std_m,
            rng=rng,
        )

        return {
            "observation/image": left_rgb,
            "observation/wrist_image": wrist_rgb,
            "observation/right_image": right_rgb,
            "observation/state": state_t,
            "actions": actions_chunk,
            "prompt": prompt,
        }
