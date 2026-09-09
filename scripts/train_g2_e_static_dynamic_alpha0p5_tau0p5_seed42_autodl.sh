#!/usr/bin/env bash
# Formal G2-E: direct alpha=.3 baseline; only alpha changes to 0.5.
set -euo pipefail

EXPECTED_BRANCH="codex/g2-e-static-dynamic-alpha0p5-tau0p5"
G2_A_BASE_COMMIT="63761021a40693694f037d850066deb2237a5c41"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_e_static_dynamic_alpha0p5_tau0p5_autodl.yml"
BASE_CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_e_static_dynamic_alpha0p3_tau0p5_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_e_static_dynamic_alpha0p5_tau0p5_seed42_market1501"
CONSOLE_LOG="${OUTPUT_DIR}.console.log"
SMOKE_OUTPUT="${G2_E_SMOKE_OUTPUT_DIR:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_e_static_dynamic_alpha0p5_tau0p5_seed42_market1501_smoke}"

cd "${REPO_ROOT}"
CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_COMMIT="$(git rev-parse HEAD)"
test "${CURRENT_BRANCH}" = "${EXPECTED_BRANCH}" || {
  echo "Expected ${EXPECTED_BRANCH}, got ${CURRENT_BRANCH}" >&2; exit 1; }
test -z "$(git status --porcelain=v1 --untracked-files=all)" || {
  echo "Formal G2-E requires a clean worktree." >&2; exit 1; }
git merge-base --is-ancestor "${G2_A_BASE_COMMIT}" "${CURRENT_COMMIT}"
test -d /root/autodl-tmp/datasets || { echo "Market1501 root absent" >&2; exit 1; }
test -f /root/autodl-tmp/pretrained/resnet50-19c8e357.pth || { echo "ImageNet weights absent" >&2; exit 1; }
test -f "${SMOKE_OUTPUT}/g2_e_smoke_analysis/g2_e_static_dynamic_alpha0p5_tau0p5_analysis_manifest.json" || {
  echo "Run the dedicated G2-E one-epoch smoke pipeline before formal training." >&2; exit 1; }
test ! -e "${OUTPUT_DIR}" && test ! -e "${CONSOLE_LOG}" || {
  echo "Refusing to reuse formal output or console log." >&2; exit 1; }
python tools/verify_g2_e_static_dynamic_alpha0p5_tau0p5_protocol.py \
  --baseline-config "${BASE_CONFIG}" --candidate-config "${CONFIG}"

python tools/verify_g2_e_alpha_smoke.py --alpha 0.5 \
  --config-file "${CONFIG}" --smoke-output "${SMOKE_OUTPUT}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python tools/train.py --config_file "${CONFIG}" 2>&1 | tee "${CONSOLE_LOG}"
python tools/finish_g2_e_alpha_experiment.py --alpha 0.5 \
  --config-file "${CONFIG}" --output-dir "${OUTPUT_DIR}" --console-log "${CONSOLE_LOG}"
