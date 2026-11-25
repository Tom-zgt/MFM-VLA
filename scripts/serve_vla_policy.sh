#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="${1:-pi05_libero_dense_v3}"
CKPT_DIR="$(realpath "${2:-./checkpoints/mfm-vla-pi05-libero-pro}")"
PORT="${3:-8130}"
OPENPI_DIR="${OPENPI_DIR:-$REPO_ROOT/openpi}"
export MFM_VLA_WEIGHTS_DIR="${MFM_VLA_WEIGHTS_DIR:-$(dirname "$CKPT_DIR")}"

cd "$OPENPI_DIR"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 .venv/bin/python -u scripts/serve_policy.py \
  --port "$PORT" policy:checkpoint \
  --policy.config "$CONFIG_NAME" \
  --policy.dir "$CKPT_DIR"
