#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/mfm_vla:${LIBERO_DIR:-$REPO_ROOT/LIBERO}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HARNESS_SETTLE_OPEN=1

python3 -u "$REPO_ROOT/mfm_vla/serve_grounding_dino.py" \
  --model-dir "${GDINO_MODEL_DIR:-IDEA-Research/grounding-dino-base}" \
  --port 9130 --device cuda:0 &
DINO_PID=$!
trap "kill $DINO_PID 2>/dev/null || true" EXIT INT TERM
sleep 8

python3 -u "$REPO_ROOT/mfm_vla/eval_grounding_dino_ablation.py" \
  --vla-ports "${VLA_PORT:-8130}" \
  --dino-ports 9130 \
  --output "${OUTPUT_JSONL:-./outputs/gdino_eval_250.jsonl}" "$@"
