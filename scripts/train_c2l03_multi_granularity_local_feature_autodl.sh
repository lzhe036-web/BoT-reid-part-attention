#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="exp/c2l03-multi-granularity-local-feature"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Refusing formal training from branch '${CURRENT_BRANCH}'; expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_c2l03_multi_granularity_local_feature_autodl.yml"
python tools/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-id C2-L03-MGL \
  --experiment-family c2l03_multi_granularity_local_feature \
  --expected-branch "${EXPECTED_BRANCH}" \
  --notes "baseline=C2-L03; method=C2-L03 + Multi-Granularity Local Feature (Global + K2 + K4 + K6, mean aggregation); lambda=0.3; baseline_existing_attention=True; new_module_attention=False."
