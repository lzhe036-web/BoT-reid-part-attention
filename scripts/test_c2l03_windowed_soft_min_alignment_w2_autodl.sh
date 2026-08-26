#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="exp/c2l03-windowed-soft-min-alignment-window-sweep-tau0p2-lambda0p05"
PARENT_BRANCH="exp/c2l03-soft-min-alignment-lambda-sweep-tau0p2"
PARENT_COMMIT="67b7bbf528a0a6279a3f9ab86aed43ad91b1ef63"
FEATURE_REFERENCE_BRANCH="exp/c2l03-hard-shortest-path-alignment"
FEATURE_REFERENCE_COMMIT="6b46f2c3747124b97d59ed5cf987f33efb82282b"
CONFIG="configs/softmax_triplet_c2l03_windowed_soft_min_alignment_tau0p2_lambda0p05_w2_autodl.yml"
EXPECTED_SMOKE_OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2l03_windowed_soft_min_alignment_w2_tau0p2_lambda0p05_seed42_market1501_smoke"

CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"
if [[ -z "${CURRENT_BRANCH}" ]] || [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Refusing W2 windowed Soft-Min smoke from branch '${CURRENT_BRANCH:-detached HEAD}'; expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Refusing W2 windowed Soft-Min smoke from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi
if [[ "$(git rev-parse "${PARENT_BRANCH}")" != "${PARENT_COMMIT}" ]] || \
   [[ "$(git rev-parse "origin/${PARENT_BRANCH}")" != "${PARENT_COMMIT}" ]] || \
   [[ "$(git merge-base "${PARENT_BRANCH}" HEAD)" != "${PARENT_COMMIT}" ]]; then
  echo "Refusing W2 smoke: fixed lambda=0.05 parent lineage differs." >&2
  exit 1
fi
if [[ "$(git rev-parse "${FEATURE_REFERENCE_BRANCH}")" != "${FEATURE_REFERENCE_COMMIT}" ]] || \
   [[ "$(git rev-parse "origin/${FEATURE_REFERENCE_BRANCH}")" != "${FEATURE_REFERENCE_COMMIT}" ]]; then
  echo "Refusing W2 smoke: fixed Hard feature reference differs." >&2
  exit 1
fi
if [[ -e "${EXPECTED_SMOKE_OUTPUT_DIR}" ]] && \
   { [[ ! -d "${EXPECTED_SMOKE_OUTPUT_DIR}" ]] || [[ -n "$(find "${EXPECTED_SMOKE_OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; }; then
  echo "Refusing to overwrite non-empty W2 smoke OUTPUT_DIR: ${EXPECTED_SMOKE_OUTPUT_DIR}" >&2
  exit 1
fi

CURRENT_COMMIT="$(git rev-parse HEAD)"
export CUDA_VISIBLE_DEVICES=0
python tools/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-id C2-L03-WSOFTMIN-W2-T0P2-LP0P05-S42-SMOKE \
  --experiment-family c2l03_windowed_soft_min_alignment_tau0p2_lambda0p05 \
  --run-kind smoke \
  --expected-branch "${EXPECTED_BRANCH}" \
  --expected-commit "${CURRENT_COMMIT}" \
  --parent-branch "${PARENT_BRANCH}" \
  --parent-commit "${PARENT_COMMIT}" \
  --feature-reference-commit "${FEATURE_REFERENCE_COMMIT}" \
  --notes "One-epoch Windowed Soft-Min W2 smoke; tau=0.2; lambda_p=0.05; K=6; Seed=42."
