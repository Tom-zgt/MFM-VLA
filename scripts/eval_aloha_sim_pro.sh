#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$REPO_ROOT/openpi}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$OPENPI_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$OPENPI_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

export PYTHONPATH="$REPO_ROOT/mfm_vla:${PYTHONPATH:-}"
export MUJOCO_GL=egl

exec "$PYTHON_BIN" -u "$REPO_ROOT/mfm_vla/eval_aloha_sim_pro.py" \
  --max-steps 400 --n-seeds 50 --num-workers 1 "$@"
