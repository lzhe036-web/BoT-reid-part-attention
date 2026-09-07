#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="codex/g2-d1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_diff46_autodl.yml"
OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_d1_tau0p5_seed42_market1501_smoke}"

cd "${REPO_ROOT}"
test "$(git branch --show-current)" = "${EXPECTED_BRANCH}"
test -z "$(git status --porcelain=v1 --untracked-files=all)" || {
  echo "Smoke requires a clean worktree." >&2; exit 1;
}
test ! -e "${OUTPUT_DIR}" || { echo "Refusing to reuse ${OUTPUT_DIR}" >&2; exit 1; }
python tools/verify_g2_diff46_protocol.py \
  --baseline-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau0p5_autodl.yml \
  --candidate-config "${CONFIG}"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python tools/train.py --config_file "${CONFIG}" \
  SOLVER.MAX_EPOCHS 1 SOLVER.CHECKPOINT_PERIOD 1 SOLVER.EVAL_PERIOD 1 \
  OUTPUT_DIR "${OUTPUT_DIR}"
