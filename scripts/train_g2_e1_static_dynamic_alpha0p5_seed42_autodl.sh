#!/usr/bin/env bash
# Formal E1: original G2 plus static concatenation and a fixed alpha=.5 residual.
set -euo pipefail

EXPECTED_BRANCH="codex/g2-e1-static-dynamic-alpha0p5"
ORIGINAL_G2_BASE_COMMIT="5a9a2a33e19f3d9dfc3afaf356dc8b9ee3f0f737"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_e1_static_dynamic_alpha0p5_autodl.yml"
BASE_CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_e1_static_dynamic_alpha0p5_tau1_seed42_market1501"
CONSOLE_LOG="${OUTPUT_DIR}.console.log"
SMOKE_OUTPUT="${E1_SMOKE_OUTPUT_DIR:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_e1_static_dynamic_alpha0p5_tau1_seed42_market1501_smoke}"

cd "${REPO_ROOT}"
CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_COMMIT="$(git rev-parse HEAD)"
test "${CURRENT_BRANCH}" = "${EXPECTED_BRANCH}" || {
  echo "Expected ${EXPECTED_BRANCH}, got ${CURRENT_BRANCH}" >&2; exit 1; }
test -z "$(git status --porcelain=v1 --untracked-files=all)" || {
  echo "Formal E1 requires a clean worktree." >&2; exit 1; }
git merge-base --is-ancestor "${ORIGINAL_G2_BASE_COMMIT}" "${CURRENT_COMMIT}"
test -d /root/autodl-tmp/datasets || { echo "Market1501 root absent" >&2; exit 1; }
test -f /root/autodl-tmp/pretrained/resnet50-19c8e357.pth || { echo "ImageNet weights absent" >&2; exit 1; }
test -f "${SMOKE_OUTPUT}/e1_smoke_analysis/e1_static_dynamic_analysis_manifest.json" || {
  echo "Run the dedicated E1 one-epoch smoke pipeline before formal training." >&2; exit 1; }
test ! -e "${OUTPUT_DIR}" && test ! -e "${CONSOLE_LOG}" || {
  echo "Refusing to reuse formal output or console log." >&2; exit 1; }
python tools/verify_g2_e1_static_dynamic_protocol.py \
  --baseline-config "${BASE_CONFIG}" --candidate-config "${CONFIG}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
START_EPOCH="$(date +%s)"; START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  python tools/train.py --config_file "${CONFIG}"
  python tools/finalize_g2_e1_static_dynamic_experiment.py \
    --config-file "${CONFIG}" --output-dir "${OUTPUT_DIR}"
} 2>&1 | tee "${CONSOLE_LOG}"
END_EPOCH="$(date +%s)"; END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

python tools/recover_g2_e1_static_dynamic_experiment.py \
  --config-file "${CONFIG}" --output-dir "${OUTPUT_DIR}" --console-log "${CONSOLE_LOG}" \
  --started-at-utc "${START_UTC}" --ended-at-utc "${END_UTC}" \
  --runtime-seconds "$((END_EPOCH - START_EPOCH))"
printf 'Formal E1 evidence was registered. Output: %s\n' "${OUTPUT_DIR}"
