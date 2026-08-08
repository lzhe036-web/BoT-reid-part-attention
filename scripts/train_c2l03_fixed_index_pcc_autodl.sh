#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="exp/c2l03-fixed-index-pcc"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Refusing formal PCC training from branch '${CURRENT_BRANCH}'; expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing formal PCC training from a dirty Git worktree." >&2
  exit 1
fi

CONFIG="configs/softmax_triplet_c2l03_fixed_index_pcc_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2l03_fixed_index_pcc_market1501"
if [[ -e "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty OUTPUT_DIR: ${OUTPUT_DIR}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
python tools/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-id C2-PCC-Fixed \
  --experiment-family c2l03_fixed_index_part_correspondence_consistency \
  --expected-branch "${EXPECTED_BRANCH}" \
  --notes "Baseline C2-L03; Fixed-Index PCC; K=6; cross_camera_positive_lambda=0.3; pcc_lambda=0.1; one formal run."
