#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
CONFIG="configs/softmax_triplet_c2_l03_multi_granularity_part_autodl.yml"

python tools/run_c2_l03_multi_granularity_part.py \
  --config "${CONFIG}" \
  --experiment-family C2-MGP-K246 \
  --run-id C2-MGP-K246-S42 \
  --evidence-id EV-TRAIN-MKT-C2-MGP-K246-S42
