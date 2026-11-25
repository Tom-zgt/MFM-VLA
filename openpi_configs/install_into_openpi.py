"""Idempotent installer that registers MFM-VLA policies, DataConfigs, and TrainConfigs into an OpenPI checkout.

Supports BOTH closed-loop policy serving (`scripts/serve_policy.py`) AND full SFT training
from scratch (`scripts/train.py`) on `pi05_libero_dense_v3`, `pi05_pretrain_human300`,
`pi05_robocasa_dense_v1`, and `pi05_robocasa_dense_v2`.

Usage:
  python openpi_configs/install_into_openpi.py --openpi-dir ./openpi
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent

ROBOCASA_DATACONFIG_SNIPPET = '''

@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaDataConfig(DataConfigFactory):
    """Data config for RoboCasa365 checkpoints and Dense-Caption fine-tuning."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        import openpi.policies.robocasa_policy as robocasa_policy
        repack_transform = _transforms.Group()
        data_transforms = _transforms.Group(
            inputs=[
                robocasa_policy.RobocasaInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[robocasa_policy.RobocasaOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=False,
        )
'''

MFM_TRAIN_CONFIGS_SNIPPET = '''
    # --- MFM-VLA Baselines & Adapted Checkpoints (auto-registered by install_into_openpi.py) ---
    TrainConfig(
        name="pi05_libero_dense_v3",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id=__import__("os").environ.get("MFM_LIBERO_REPO_ID", "local/libero_skill_v2"),
            assets=AssetsConfig(
                assets_dir=__import__("os").environ.get(
                    "MFM_LIBERO_ASSETS_DIR",
                    str(pathlib.Path(__import__("os").environ.get("MFM_VLA_WEIGHTS_DIR", "./checkpoints")).resolve() / "mfm-vla-pi05-libero-pro/assets"),
                ),
                asset_id="physical-intelligence/libero",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            dense_caption_sidecar=__import__("os").environ.get("MFM_LIBERO_DENSE_SIDECAR") or None,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            __import__("os").environ.get("MFM_LIBERO_BASE_PARAMS", "gs://openpi-assets/checkpoints/pi05_libero/params")
        ),
        num_train_steps=10_000,
        save_interval=2_500,
        keep_period=2_500,
    ),
    TrainConfig(
        name="pi05_pretrain_human300",
        model=pi0_config.Pi0Config(pi05=True, max_token_len=200),
        data=LeRobotRobocasaDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=100_000,
        save_interval=5_000,
        keep_period=10_000,
        batch_size=64,
    ),
    TrainConfig(
        name="pi05_robocasa_dense_v1",
        model=pi0_config.Pi0Config(pi05=True, max_token_len=200),
        data=LeRobotRobocasaDataConfig(
            repo_id="robocasa_dense:" + __import__("os").environ.get("MFM_ROBOCASA_DATASET_DIR", "./data/robocasa_dense_v1"),
            assets=AssetsConfig(
                assets_dir=__import__("os").environ.get(
                    "MFM_ROBOCASA_ASSETS_DIR",
                    str(pathlib.Path(__import__("os").environ.get("MFM_VLA_WEIGHTS_DIR", "./checkpoints")).resolve() / "mfm-vla-pi05-robocasa365"),
                ),
                asset_id="assets",
            ),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=2.5e-5, decay_steps=5_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            __import__("os").environ.get("MFM_ROBOCASA_BASE_PARAMS", "./checkpoints/pi05_pretrain_human300/multitask_learning/75000/params")
        ),
        num_train_steps=5_000,
        save_interval=2_500,
        keep_period=2_500,
    ),
    TrainConfig(
        name="pi05_robocasa_dense_v2",
        model=pi0_config.Pi0Config(pi05=True, max_token_len=200),
        data=LeRobotRobocasaDataConfig(
            repo_id="robocasa_dense:" + __import__("os").environ.get("MFM_ROBOCASA_DATASET_DIR", "./data/robocasa_dense_v1"),
            assets=AssetsConfig(
                assets_dir=__import__("os").environ.get(
                    "MFM_ROBOCASA_ASSETS_DIR",
                    str(pathlib.Path(__import__("os").environ.get("MFM_VLA_WEIGHTS_DIR", "./checkpoints")).resolve() / "mfm-vla-pi05-robocasa365"),
                ),
                asset_id="assets",
            ),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=150, peak_lr=2.0e-5, decay_steps=2_500, decay_lr=2.0e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            __import__("os").environ.get("MFM_ROBOCASA_STAGE1_PARAMS", "./checkpoints/pi05_robocasa_dense_v1/dense_v1/4999/params")
        ),
        num_train_steps=2_500,
        save_interval=2_500,
        keep_period=2_500,
    ),
'''


def _patch_libero_dense_sidecar(cfg_text: str) -> str:
    if "dense_caption_sidecar: str | None = None" in cfg_text:
        return cfg_text
    field_target = "    extra_delta_transform: bool = False\n"
    if field_target in cfg_text:
        cfg_text = cfg_text.replace(
            field_target,
            field_target + "    dense_caption_sidecar: str | None = None\n",
            1,
        )
    old_repack = (
        "        repack_transform = _transforms.Group(\n"
        "            inputs=[\n"
        "                _transforms.RepackTransform(\n"
        "                    {\n"
        '                        "observation/image": "image",\n'
        '                        "observation/wrist_image": "wrist_image",\n'
        '                        "observation/state": "state",\n'
        '                        "actions": "actions",\n'
        '                        "prompt": "prompt",\n'
        "                    }\n"
        "                )\n"
        "            ]\n"
        "        )"
    )
    new_repack = (
        "        repack_inputs = []\n"
        "        if self.dense_caption_sidecar is not None:\n"
        "            repack_inputs.append(\n"
        "                libero_policy.LiberoDenseCaptionTransform(sidecar_path=self.dense_caption_sidecar)\n"
        "            )\n"
        "        repack_inputs.append(\n"
        "            _transforms.RepackTransform(\n"
        "                {\n"
        '                    "observation/image": "image",\n'
        '                    "observation/wrist_image": "wrist_image",\n'
        '                    "observation/state": "state",\n'
        '                    "actions": "actions",\n'
        '                    "prompt": "prompt",\n'
        "                }\n"
        "            )\n"
        "        )\n"
        "        repack_transform = _transforms.Group(inputs=repack_inputs)"
    )
    if old_repack in cfg_text:
        cfg_text = cfg_text.replace(old_repack, new_repack, 1)
    return cfg_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpi-dir", type=Path, default=Path("./openpi"))
    args = ap.parse_args()

    src_openpi = args.openpi_dir / "src" / "openpi"
    if not src_openpi.is_dir():
        raise SystemExit(f"OpenPI source directory not found at {src_openpi}")

    shutil.copy2(HERE / "robocasa_policy.py", src_openpi / "policies" / "robocasa_policy.py")
    shutil.copy2(HERE / "libero_policy.py", src_openpi / "policies" / "libero_policy.py")
    shutil.copy2(HERE / "robocasa_dense_dataset.py", src_openpi / "training" / "robocasa_dense_dataset.py")

    cfg_file = src_openpi / "training" / "config.py"
    cfg_text = _patch_libero_dense_sidecar(cfg_file.read_text())

    if "class LeRobotRobocasaDataConfig" not in cfg_text:
        anchor = "@dataclasses.dataclass(frozen=True)\nclass RLDSDroidDataConfig"
        if anchor in cfg_text:
            cfg_text = cfg_text.replace(anchor, ROBOCASA_DATACONFIG_SNIPPET.lstrip("\n") + "\n\n" + anchor, 1)
        else:
            cfg_text += "\n" + ROBOCASA_DATACONFIG_SNIPPET

    if '"pi05_libero_dense_v3"' not in cfg_text:
        anchor = "_CONFIGS = ["
        if anchor not in cfg_text:
            raise RuntimeError("Could not find `_CONFIGS = [` in openpi/src/openpi/training/config.py")
        cfg_text = cfg_text.replace(anchor, anchor + MFM_TRAIN_CONFIGS_SNIPPET, 1)

    cfg_file.write_text(cfg_text)

    dl_file = src_openpi / "training" / "data_loader.py"
    if dl_file.is_file():
        dl_text = dl_file.read_text()
        if "RobocasaDenseDataset" not in dl_text and "def create_torch_dataset(" in dl_text:
            target = "    repo_id = data_config.repo_id"
            hook = (
                "    repo_id = data_config.repo_id\n"
                '    if isinstance(repo_id, str) and repo_id.startswith("robocasa_dense:"):\n'
                "        from openpi.training.robocasa_dense_dataset import RobocasaDenseDataset\n"
                '        return RobocasaDenseDataset(repo_id.removeprefix("robocasa_dense:"), action_horizon=action_horizon)\n'
            )
            if target in dl_text:
                dl_file.write_text(dl_text.replace(target, hook, 1))

    print(f"[install_into_openpi] Installed MFM-VLA policies, DataConfigs, and TrainConfigs into {src_openpi}")


if __name__ == "__main__":
    main()
