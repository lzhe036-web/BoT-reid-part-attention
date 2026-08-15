#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="exp/c2l03-hard-shortest-path-alignment"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Refusing hard-alignment formal training from branch '${CURRENT_BRANCH}'; expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing hard-alignment formal training from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi

CURRENT_COMMIT="$(git rev-parse HEAD)"
CONFIG="configs/softmax_triplet_c2l03_hard_shortest_path_alignment_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2l03_hard_shortest_path_alignment_seed42_market1501"
if [[ -e "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty OUTPUT_DIR: ${OUTPUT_DIR}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
python tools/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-id C2-HARD-ALIGN-K6-S42 \
  --experiment-family c2l03_hard_shortest_path_alignment \
  --run-kind formal \
  --expected-branch "${EXPECTED_BRANCH}" \
  --expected-commit "${CURRENT_COMMIT}" \
  --notes "C2-L03 hard shortest-path part alignment; K=6; Seed=42; formal 120-epoch protocol."
