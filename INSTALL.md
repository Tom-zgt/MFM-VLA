# Installation & Reproduction Guide

## 1. Core Dependencies & OpenPI Policy Server
1. Clone [OpenPI](https://github.com/Physical-Intelligence/openpi) into `./openpi` and install its virtual environment (`uv sync`).
2. Register the MFM-VLA policy transforms (`robocasa_policy.py`, `libero_policy.py`, `robocasa_dense_dataset.py`) and `TrainConfig` entries (`pi05_libero_dense_v3`, `pi05_pretrain_human300`, `pi05_robocasa_dense_v1`, `pi05_robocasa_dense_v2`) into `./openpi`:
   ```bash
   python3 openpi_configs/install_into_openpi.py --openpi-dir ./openpi
   ```
3. Install client/adapter dependencies (including the lightweight `openpi-client` WebSocket client) in your benchmark evaluation environment:
   ```bash
   pip install -r requirements.txt
   pip install -e ./openpi/packages/openpi-client
   ```

## 2. Simulation Benchmarks
- **LIBERO-PRO**: Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), set `LIBERO_DIR=/path/to/LIBERO`, and download the pinned LIBERO-PRO static BDDL/init suites into `./third_party/LIBERO-PRO-data`:
  ```bash
  bash scripts/setup_libero_pro.sh
  ```
- **ALOHA-Sim-PRO**: Install `gym-aloha` (`pip install gym-aloha` inside `./openpi/.venv` or set `PYTHON_BIN`) and place the `pi0_aloha_sim` OpenPI checkpoint at `./checkpoints/pi0_aloha_sim` (or set `ALOHA_PI0_CKPT`).
- **RoboCasa365**: Install `robosuite` and `robocasa` (`RoboCasa365` kitchen assets) in a Python 3.11 environment and pass `PYTHON_BIN=/path/to/robocasa_env/bin/python`.

## 3. Pretrained MFM-VLA Checkpoints
Place the released weight folders under `./checkpoints/`:
- `./checkpoints/mfm-vla-pi05-libero-pro` (`pi05_libero_dense_v3`, step 10,000 / `9999`)
- `./checkpoints/mfm-vla-pi05-robocasa365` (`pi05_robocasa_dense_v2`, step 7,500 cumulative / `2499`)

## 4. Optional: Fine-Tuning from Scratch (Dense Geometric Captioning + Hand-off Recovery)
To reproduce policy adaptation from the official base $\pi_{0.5}$ checkpoints instead of using the released weights:
- **LIBERO**:
  1. Build perturbed hand-off recovery episodes and Dense Geometric Caption sidecar (`mfm_vla/build_handoff_dataset.py` and `mfm_vla/build_dense_caption_sidecar.py`).
  2. Set `MFM_LIBERO_REPO_ID`, `MFM_LIBERO_DENSE_SIDECAR`, and `MFM_LIBERO_BASE_PARAMS`, then run OpenPI's `scripts/train.py pi05_libero_dense_v3`.
- **RoboCasa365**:
  1. Build seen-task human dense episodes and hand-off recovery rollouts (`mfm_vla/build_robocasa_human_dense_dataset.py` and `mfm_vla/build_robocasa_handoff_dataset.py` — strictly excluding all 16 Composite-Unseen tasks).
  2. Set `MFM_ROBOCASA_DATASET_DIR` and `MFM_ROBOCASA_BASE_PARAMS`, then run stage-1 (`pi05_robocasa_dense_v1`, 5k steps) followed by stage-2 (`pi05_robocasa_dense_v2`, 2.5k steps) via OpenPI's `scripts/train.py`.
