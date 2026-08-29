#!/usr/bin/env bash
set -euo pipefail
EXPECTED_BRANCH="codex/g2-without-z6"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_without_z6_autodl.yml"
DEFAULT_SMOKE_OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z6_tau1_seed42_market1501_smoke"
OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${DEFAULT_SMOKE_OUTPUT_DIR}}"
cd "${REPO_ROOT}"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "Expected branch ${EXPECTED_BRANCH}" >&2; exit 1; }
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || { echo "G2-without-z6 smoke requires a clean worktree." >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to reuse smoke output directory: ${OUTPUT_DIR}" >&2; exit 1; }
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python tools/train.py --config_file "${CONFIG}" SOLVER.MAX_EPOCHS 1 SOLVER.CHECKPOINT_PERIOD 1 SOLVER.EVAL_PERIOD 1 OUTPUT_DIR "${OUTPUT_DIR}"
