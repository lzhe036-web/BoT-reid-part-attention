#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="exp/c2l03-fixed-index-pcc"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Refusing PCC smoke test from branch '${CURRENT_BRANCH}'; expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi

CONFIG="configs/softmax_triplet_c2l03_fixed_index_pcc_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2l03_fixed_index_pcc_market1501_smoke_1epoch"
if [[ -e "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty smoke OUTPUT_DIR: ${OUTPUT_DIR}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
  --config_file "${CONFIG}" \
  SOLVER.MAX_EPOCHS 1 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  SOLVER.EVAL_PERIOD 1 \
  OUTPUT_DIR "${OUTPUT_DIR}"

echo "Smoke test finished. This run is intentionally excluded from formal experiment tables."
