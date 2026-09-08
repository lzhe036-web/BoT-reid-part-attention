#!/usr/bin/env bash
# One-epoch real-data smoke for G2-F; never a formal result.
set -euo pipefail

EXPECTED_BRANCH="codex/g2-f-top2-tau0p5"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_f_top2_tau0p5_autodl.yml"
BASE_CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau0p5_autodl.yml"
OUTPUT_DIR="${G2_F_SMOKE_OUTPUT_DIR:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_f_top2_tau0p5_seed42_market1501_smoke}"

cd "${REPO_ROOT}"
test "$(git branch --show-current)" = "${EXPECTED_BRANCH}"
test -z "$(git status --porcelain=v1 --untracked-files=all)" || {
  echo "G2-F smoke requires a clean worktree." >&2; exit 1; }
test ! -e "${OUTPUT_DIR}" || { echo "Refusing to reuse ${OUTPUT_DIR}" >&2; exit 1; }
python tools/verify_g2_f_top2_tau0p5_protocol.py \
  --baseline-config "${BASE_CONFIG}" --candidate-config "${CONFIG}"
python tools/train.py --config_file "${CONFIG}" \
  SOLVER.MAX_EPOCHS 1 SOLVER.CHECKPOINT_PERIOD 1 SOLVER.EVAL_PERIOD 1 \
  OUTPUT_DIR "${OUTPUT_DIR}"
CHECKPOINT="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.pt' -print -quit)"
test -n "${CHECKPOINT}" && test -f "${CHECKPOINT}"
python tools/analyze_g2_f_top2_tau0p5.py \
  --config-file "${CONFIG}" --weight "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}/g2_f_smoke_analysis" \
  --epoch-stats "${OUTPUT_DIR}/dynamic_gating_epoch_stats.jsonl" --sample-limit 32
printf 'G2-F smoke evidence: %s\n' \
  "${OUTPUT_DIR}/g2_f_smoke_analysis/g2_f_top2_tau0p5_analysis_manifest.json"
