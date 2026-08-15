#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="exp/c2l03-soft-min-alignment"
PARENT_BRANCH="exp/c2l03-hard-shortest-path-alignment"
PARENT_COMMIT="6b46f2c3747124b97d59ed5cf987f33efb82282b"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Refusing Soft-Min smoke training from branch '${CURRENT_BRANCH}'; expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing Soft-Min smoke training from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi
if [[ "$(git rev-parse "${PARENT_BRANCH}")" != "${PARENT_COMMIT}" ]] || \
   [[ "$(git rev-parse "origin/${PARENT_BRANCH}")" != "${PARENT_COMMIT}" ]] || \
   [[ "$(git merge-base "${PARENT_BRANCH}" HEAD)" != "${PARENT_COMMIT}" ]]; then
  echo "Refusing Soft-Min smoke training: fixed Hard parent lineage differs." >&2
  exit 1
fi

CURRENT_COMMIT="$(git rev-parse HEAD)"
CONFIG="configs/softmax_triplet_c2l03_soft_min_alignment_autodl.yml"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/root/autodl-tmp/experiments/BoT/c2l03_soft_min_alignment_tau0p1_seed42_market1501_smoke}"
if [[ -e "${SMOKE_OUTPUT_DIR}" ]] && [[ -n "$(find "${SMOKE_OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty smoke OUTPUT_DIR: ${SMOKE_OUTPUT_DIR}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
python tools/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-id C2-L03-SOFTMIN-T0P1-S42-SMOKE \
  --experiment-family c2l03_soft_min_alignment \
  --run-kind smoke \
  --expected-branch "${EXPECTED_BRANCH}" \
  --expected-commit "${CURRENT_COMMIT}" \
  --parent-branch "${PARENT_BRANCH}" \
  --parent-commit "${PARENT_COMMIT}" \
  --notes "One-epoch Soft-Min tau=0.1 smoke; retained in run evidence and excluded from formal tables." \
  SOLVER.MAX_EPOCHS 1 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  SOLVER.EVAL_PERIOD 1 \
  OUTPUT_DIR "${SMOKE_OUTPUT_DIR}"
