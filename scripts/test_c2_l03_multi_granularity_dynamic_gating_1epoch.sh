#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" tools/run_experiment.py \
  --run-kind smoke \
  --config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_autodl.yml \
  --output-dir /root/autodl-tmp/experiments/BoT/c2_l03_multi_granularity_dynamic_gating_tau1_seed42_market1501_smoke \
  --feature-reference-commit 9cd7dbcee07b255803c8c21f4d9c5ee67a30930e
