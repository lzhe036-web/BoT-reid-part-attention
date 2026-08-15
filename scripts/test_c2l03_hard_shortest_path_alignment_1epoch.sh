#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="exp/c2l03-hard-shortest-path-alignment"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Refusing hard-alignment smoke training from branch '${CURRENT_BRANCH}'; expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing hard-alignment smoke training from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi

CURRENT_COMMIT="$(git rev-parse HEAD)"
CONFIG="configs/softmax_triplet_c2l03_hard_shortest_path_alignment_autodl.yml"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/root/autodl-tmp/experiments/BoT/c2l03_hard_shortest_path_alignment_seed42_market1501_smoke_1epoch}"
if [[ -e "${SMOKE_OUTPUT_DIR}" ]] && [[ -n "$(find "${SMOKE_OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty smoke OUTPUT_DIR: ${SMOKE_OUTPUT_DIR}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
python tools/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-id C2-HARD-ALIGN-K6-S42-SMOKE \
  --experiment-family c2l03_hard_shortest_path_alignment \
  --run-kind smoke \
  --expected-branch "${EXPECTED_BRANCH}" \
  --expected-commit "${CURRENT_COMMIT}" \
  --notes "One-epoch hard shortest-path alignment smoke run; excluded from formal result tables." \
  SOLVER.MAX_EPOCHS 1 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  SOLVER.EVAL_PERIOD 1 \
  OUTPUT_DIR "${SMOKE_OUTPUT_DIR}"
