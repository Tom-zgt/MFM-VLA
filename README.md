# Ground, Guide, Act: Improving VLA Spatial Generalization with Multimodal Foundation Models (MFM-VLA)

[![arXiv](https://img.shields.io/badge/arXiv-2611.31452-b31b1b.svg)](https://arxiv.org/abs/2611.31452)
[![Paper PDF](https://img.shields.io/badge/PDF-arXiv%202611.31452-2457c5.svg)](https://arxiv.org/pdf/2611.31452)
[![Project Page](https://img.shields.io/badge/Project%20Page-MFM--VLA-0d7a75.svg)](https://tom-zgt.github.io/MFM-VLA/)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Models-zhengzhou%2FMFM--VLA-ffb000.svg)](https://huggingface.co/zhengzhou/MFM-VLA)
[![License](https://img.shields.io/badge/License-Apache%202.0-slate.svg)](LICENSE)

**[Guangting Zheng](https://tom-zgt.github.io/)\*, Shaoxuan Wang\*, Yu Yuan, Zhipeng Tang, Jiajun Deng, Yanyong Zhang**  
*University of Science and Technology of China (`*` Equal contribution)*

Official codebase for **MFM-VLA**: a selective collaboration framework uniting **Dense Geometric Policy Adaptation**, **Development-Calibrated Capability Routing**, and **Depth-Free Two-View 3D Grounding & Spatial Guidance**.

## Highlights
- **Diagnosing Pretrained VLAs**: Reveals visual scene shortcuts under counterfactual instruction audits and shows that pretrained VLAs retain strong local contact skills once placed near target objects.
- **Selective System-2 / System-1 Collaboration**: Invokes Multimodal Foundation Models (MFMs) only on perturbation regimes where external 3D grounding outperforms direct VLA execution, issuing **0 MFM calls on 68.75% of LIBERO-PRO suites** (`2.38` queries / `3,446` tokens per episode on average).
- **Consistent Cross-Embodiment Gains**:
  - **LIBERO-PRO** (7-DoF Desktop, 8,000 episodes): `54.9% -> 69.8%` overall (`84.4%` with modular Code Policy extension; `55.5% -> 75.1%` suite-routed / `78.5%` task-routed on 800-ep ablation split)
  - **ALOHA-Sim-PRO** (14-DoF Bimanual, 400-step): `23.5% -> 97.5%` perturbation average (`+74.0` pp)
  - **RoboCasa365** (12-DoF Mobile Kitchen, 2,500 episodes): `16.9% -> 20.5%` (`30.3%` with modular Code Policy extension)
  - **Real-World Franka-XHand** (Depth-Free Transparent Glassware): `15.3% -> 54.4%` atomic, `0.8% -> 31.7%` multi-stage

## Repository Structure
- `mfm_vla/` — Core library:
  - `mfm_client.py` — OpenAI-compatible multimodal foundation model session client (`OPENAI_API_KEY` / `OPENAI_BASE_URL`)
  - `generic_grounding.py`, `generic_multiview.py` — Calibrated two-view RGB ray triangulation, visibility survey, and geometric fallback
  - `g5j_router_v2.py`, `generic_route.py` — Development-calibrated capability router (suite-level & task-level)
  - `dense_caption.py`, `runtime_dense_caption.py`, `robocasa_dense_caption.py` — Action-chunk-level Dense Geometric Captioning (DGC)
  - `generic_verify.py` — Visual grasp verification and local re-grasp recovery
  - `eval_g5.py`, `eval_aloha_sim_pro.py`, `eval_robocasa.py` — Closed-loop evaluation harnesses across all three simulation benchmarks
  - `serve_grounding_dino.py`, `eval_grounding_dino_ablation.py` — Open-vocabulary detector (`GroundingDINO-Base`) controlled replacement ablations
- `configs/` — Frozen router decisions (`libero_suite_router_frozen.json`, `manual_task_router_v1.json`, `robocasa_task_router_frozen.json`) and profiled handoff offsets
- `openpi_configs/` — OpenPI `TrainConfig` definitions for adapted $\pi_{0.5}$ checkpoints
- `scripts/` — Ready-to-run launch scripts for policy serving and benchmark evaluation

## Quickstart

### 1. Download Checkpoints & Start the Adapted $\pi_{0.5}$ Policy Server
```bash
huggingface-cli download zhengzhou/MFM-VLA --local-dir ./checkpoints
python3 openpi_configs/install_into_openpi.py --openpi-dir ./openpi
bash scripts/serve_vla_policy.sh pi05_libero_dense_v3 ./checkpoints/mfm-vla-pi05-libero-pro 8130
```

### 2. Run MFM-VLA Evaluation
```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export MFM_MODEL="gpt-4o"   # or any OpenAI-compatible vision-language model

# LIBERO-PRO (reads frozen suite router: 11 Direct suites use 0 API calls; 5 Assisted suites use MFM)
bash scripts/eval_libero_pro.sh

# ALOHA-Sim-PRO (400-step bimanual transfer)
bash scripts/eval_aloha_sim_pro.sh

# RoboCasa365
bash scripts/eval_robocasa365.sh --task PickPlaceToasterToCounter --episode 5
```

## Citation
```bibtex
@article{zheng2026mfmvla,
  title   = {Ground, Guide, Act: Improving VLA Spatial Generalization with Multimodal Foundation Models},
  author  = {Zheng, Guangting and Wang, Shaoxuan and Yuan, Yu and Tang, Zhipeng and Deng, Jiajun and Zhang, Yanyong},
  journal = {arXiv preprint arXiv:2611.31452},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.31452}
}
```

## License
Released under the Apache-2.0 License.
