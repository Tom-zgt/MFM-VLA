#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/mfm_vla:${LIBERO_DIR:-$REPO_ROOT/LIBERO}:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HARNESS_SETTLE_OPEN=1
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" -u "$REPO_ROOT/mfm_vla/run_libero_pro_benchmark.py" \
  --vla-port "${VLA_PORT:-8130}" \
  --output "${OUTPUT_JSONL:-./outputs/libero_pro_eval.jsonl}" "$@"
