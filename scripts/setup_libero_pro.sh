#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${LIBERO_PRO_DATA:-$ROOT/third_party/LIBERO-PRO-data}"
DATA_REVISION="c86fc3b8293185a6f373677018ff3e37f8391602"
SOURCE_COMMIT="eafdb809426b13153aa1e4c42d6601844217dfec"

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

"$PYTHON_BIN" - "$DATA_ROOT" "$DATA_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zhouxueyang/LIBERO-Pro",
    repo_type="dataset",
    revision=sys.argv[2],
    local_dir=sys.argv[1],
    allow_patterns=[
        "bddl_files/libero_*_swap/**",
        "bddl_files/libero_*_object/**",
        "bddl_files/libero_*_lan/**",
        "bddl_files/libero_*_task/**",
        "init_files/libero_*_swap/**",
        "init_files/libero_*_object/**",
        "init_files/libero_*_lan/**",
        "init_files/libero_*_task/**",
        "metadata/**",
        "README.md",
    ],
    max_workers=8,
)
PY

printf '%s\n' "$DATA_REVISION" > "$DATA_ROOT/.source-revision"
printf '%s\n' "$SOURCE_COMMIT" > "$DATA_ROOT/.benchmark-source-commit"

for base in spatial object goal 10; do
  for suffix in swap object lan task; do
    suite="libero_${base}_${suffix}"
    [[ "$(find "$DATA_ROOT/bddl_files/$suite" -type f -name '*.bddl' | wc -l)" == 10 ]]
    [[ "$(find "$DATA_ROOT/init_files/$suite" -type f -name '*.pruned_init' | wc -l)" == 10 ]]
  done
done

echo "LIBERO-PRO static perturbation suites are ready at $DATA_ROOT"
