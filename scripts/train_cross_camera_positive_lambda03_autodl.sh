#!/usr/bin/env bash
set -euo pipefail

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "C2L03" ]]; then
  echo "Refusing formal C2-L03 training from branch '${CURRENT_BRANCH}'. Switch to C2L03 first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_cross_camera_positive_lambda03_autodl.yml"
python tools/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-id C2-L03 \
  --experiment-family c2_lambda_sensitivity \
  --expected-branch C2L03 \
  --notes "C2-L03 independent formal run, cross-camera positive only, lambda=0.3, no extra hard negative."
