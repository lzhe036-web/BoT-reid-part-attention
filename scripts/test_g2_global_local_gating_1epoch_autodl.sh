#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_autodl.yml"
OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_global_local_tau1_seed42_market1501_smoke}"

cd "${REPO_ROOT}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to reuse smoke output directory: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi

python tools/train.py \
  --config_file "${CONFIG}" \
  SOLVER.MAX_EPOCHS 1 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  SOLVER.EVAL_PERIOD 1 \
  OUTPUT_DIR "${OUTPUT_DIR}"
