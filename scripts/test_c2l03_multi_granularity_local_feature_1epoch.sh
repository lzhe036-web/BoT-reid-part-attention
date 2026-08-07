#!/usr/bin/env bash
set -euo pipefail

# Non-formal smoke entry. It is intentionally separate from the 120-epoch
# configuration and must never be used as a paper result.
export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_c2l03_multi_granularity_local_feature_autodl.yml"
python tools/train.py \
  --config_file "${CONFIG}" \
  SOLVER.MAX_EPOCHS 1 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  SOLVER.EVAL_PERIOD 1 \
  OUTPUT_DIR "/root/autodl-tmp/experiments/BoT/c2l03_multi_granularity_local_market1501_smoke_1epoch"
