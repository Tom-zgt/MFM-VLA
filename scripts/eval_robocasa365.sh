#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/mfm_vla:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" -u "$REPO_ROOT/mfm_vla/eval_robocasa.py" \
  --port "${VLA_PORT:-8170}" \
  --mode g5j_router \
  --dense-prompt \
  --frozen-router "$REPO_ROOT/configs/robocasa_task_router_frozen.json" \
  --artifact-dir "${ARTIFACT_DIR:-./outputs/robocasa_artifacts}" \
  --output "${OUTPUT_JSONL:-./outputs/robocasa_eval.jsonl}" "$@"
