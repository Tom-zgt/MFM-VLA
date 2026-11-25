"""OpenPI TrainConfig entries for MFM-VLA adapted checkpoints.

Add these entries to `openpi/src/openpi/training/config.py` when serving or fine-tuning
`mfm-vla-pi05-libero-pro` and `mfm-vla-pi05-robocasa365`.
"""

import os
from pathlib import Path
from openpi.models import pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import AssetsConfig, DataConfig, LeRobotLiberoDataConfig, LeRobotRobocasaDataConfig, TrainConfig

CKPT_ROOT = Path(os.environ.get("MFM_VLA_WEIGHTS_DIR", "./checkpoints"))

MFM_VLA_CONFIGS = [
    TrainConfig(
        name="pi05_libero_dense_v3",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="local/libero_dense_v3",
            assets=AssetsConfig(
                assets_dir=str(CKPT_ROOT / "mfm-vla-pi05-libero-pro/assets"),
                asset_id="physical-intelligence/libero",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=10_000,
    ),
    TrainConfig(
        name="pi05_robocasa_dense_v2",
        model=pi0_config.Pi0Config(pi05=True, max_token_len=200),
        data=LeRobotRobocasaDataConfig(
            assets=AssetsConfig(
                assets_dir=str(CKPT_ROOT / "mfm-vla-pi05-robocasa365"),
                asset_id="assets",
            ),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=150, peak_lr=2.0e-5, decay_steps=2_500, decay_lr=2.0e-6
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=2_500,
    ),
]
