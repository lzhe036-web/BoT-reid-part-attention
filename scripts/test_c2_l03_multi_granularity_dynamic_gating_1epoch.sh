#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEFAULT_SMOKE_OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2_l03_multi_granularity_dynamic_gating_tau1_seed42_market1501_smoke"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${DEFAULT_SMOKE_OUTPUT_DIR}}"

if [[ -e "$SMOKE_OUTPUT_DIR" && ! -d "$SMOKE_OUTPUT_DIR" ]]; then
  printf 'ERROR: SMOKE_OUTPUT_DIR exists and is not a directory: %s\n' "$SMOKE_OUTPUT_DIR" >&2
  exit 1
fi
if [[ -d "$SMOKE_OUTPUT_DIR" ]] && [[ -n "$(find "$SMOKE_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf 'ERROR: SMOKE_OUTPUT_DIR exists and is non-empty: %s\n' "$SMOKE_OUTPUT_DIR" >&2
  exit 1
fi

exec "${PYTHON:-python}" tools/run_experiment.py \
  --run-kind smoke \
  --config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_autodl.yml \
  --output-dir "${SMOKE_OUTPUT_DIR}" \
  --feature-reference-commit 9cd7dbcee07b255803c8c21f4d9c5ee67a30930e
