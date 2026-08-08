#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="exp/c2l03-dynamic-granularity-gating"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Refusing formal training from branch '${CURRENT_BRANCH}'; expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_c2l03_static_granularity_fusion_autodl.yml"
python tools/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-id C2-L03-STATIC-GRANULARITY-FUSION \
  --experiment-family c2l03_static_granularity_fusion \
  --expected-branch "${EXPECTED_BRANCH}" \
  --method "C2-L03 + Multi-Granularity Static Fusion" \
  --baseline-method "C2-L03" \
  --notes "Parallel B static control; Global/K2/K4/K6; PCC=False; CondPA=False; seed=42; C2 lambda=0.3."
