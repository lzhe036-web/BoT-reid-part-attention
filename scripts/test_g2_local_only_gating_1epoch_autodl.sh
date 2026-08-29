#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="codex/g2-local-only"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_local_only_autodl.yml"
DEFAULT_SMOKE_OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_local_only_tau1_seed42_market1501_smoke"
OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${DEFAULT_SMOKE_OUTPUT_DIR}}"

cd "${REPO_ROOT}"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  printf 'Expected branch %s, got %s\n' "${EXPECTED_BRANCH}" "${CURRENT_BRANCH}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  printf 'G2-local-only smoke requires a clean worktree.\n' >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to reuse smoke output directory: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python tools/train.py \
  --config_file "${CONFIG}" \
  SOLVER.MAX_EPOCHS 1 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  SOLVER.EVAL_PERIOD 1 \
  OUTPUT_DIR "${OUTPUT_DIR}"
