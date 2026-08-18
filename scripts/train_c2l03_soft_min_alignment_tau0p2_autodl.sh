#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="exp/c2l03-soft-min-alignment-tau-sweep"
PARENT_BRANCH="exp/c2l03-hard-shortest-path-alignment"
PARENT_COMMIT="6b46f2c3747124b97d59ed5cf987f33efb82282b"
CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"
if [[ -z "${CURRENT_BRANCH}" ]]; then
  echo "Refusing tau=0.2 Soft-Min formal training from detached HEAD." >&2
  exit 1
fi
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Refusing tau=0.2 Soft-Min formal training from branch '${CURRENT_BRANCH}'; expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing tau=0.2 Soft-Min formal training from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi
if [[ "$(git rev-parse "${PARENT_BRANCH}")" != "${PARENT_COMMIT}" ]] || \
   [[ "$(git rev-parse "origin/${PARENT_BRANCH}")" != "${PARENT_COMMIT}" ]] || \
   [[ "$(git merge-base "${PARENT_BRANCH}" HEAD)" != "${PARENT_COMMIT}" ]]; then
  echo "Refusing tau=0.2 Soft-Min formal training: fixed Hard parent/reference lineage differs." >&2
  exit 1
fi

CURRENT_COMMIT="$(git rev-parse HEAD)"
CONFIG="configs/softmax_triplet_c2l03_soft_min_alignment_tau0p2_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2l03_soft_min_alignment_tau0p2_seed42_market1501"
if [[ -e "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty OUTPUT_DIR: ${OUTPUT_DIR}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
python tools/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-id C2-L03-SOFTMIN-T0P2-S42 \
  --experiment-family c2l03_soft_min_alignment_tau_sweep \
  --run-kind formal \
  --required-smoke-experiment-id C2-L03-SOFTMIN-T0P2-S42-SMOKE \
  --expected-branch "${EXPECTED_BRANCH}" \
  --expected-commit "${CURRENT_COMMIT}" \
  --parent-branch "${PARENT_BRANCH}" \
  --parent-commit "${PARENT_COMMIT}" \
  --feature-reference-commit "${PARENT_COMMIT}" \
  --notes "C2-L03 Soft-Min tau sweep formal; tau=0.2; lambda=0.1; K=6; Seed=42; 120 epochs."
