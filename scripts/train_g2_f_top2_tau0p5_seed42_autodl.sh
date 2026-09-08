#!/usr/bin/env bash
# Formal G2-F: direct G2-A tau=.5 baseline plus hard Top-2 gating only.
set -euo pipefail

EXPECTED_BRANCH="codex/g2-f-top2-tau0p5"
G2_A_BASE_COMMIT="d724a6536e4a819c5d2932412e90b7dea224041b"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_f_top2_tau0p5_autodl.yml"
BASE_CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau0p5_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_f_top2_tau0p5_seed42_market1501"
CONSOLE_LOG="${OUTPUT_DIR}.console.log"
SMOKE_OUTPUT="${G2_F_SMOKE_OUTPUT_DIR:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_f_top2_tau0p5_seed42_market1501_smoke}"

cd "${REPO_ROOT}"
test "$(git branch --show-current)" = "${EXPECTED_BRANCH}" || {
  echo "Expected ${EXPECTED_BRANCH}, got $(git branch --show-current)" >&2; exit 1; }
test -z "$(git status --porcelain=v1 --untracked-files=all)" || {
  echo "Formal G2-F requires a clean worktree." >&2; exit 1; }
git merge-base --is-ancestor "${G2_A_BASE_COMMIT}" "$(git rev-parse HEAD)"
test -d /root/autodl-tmp/datasets || { echo "Market1501 root absent" >&2; exit 1; }
test -f /root/autodl-tmp/pretrained/resnet50-19c8e357.pth || { echo "ImageNet weights absent" >&2; exit 1; }
test -f "${SMOKE_OUTPUT}/g2_f_smoke_analysis/g2_f_top2_tau0p5_analysis_manifest.json" || {
  echo "Run the dedicated G2-F one-epoch smoke pipeline first." >&2; exit 1; }
test ! -e "${OUTPUT_DIR}" && test ! -e "${CONSOLE_LOG}" || {
  echo "Refusing to reuse formal output or console log." >&2; exit 1; }
python tools/verify_g2_f_top2_tau0p5_protocol.py \
  --baseline-config "${BASE_CONFIG}" --candidate-config "${CONFIG}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
START_EPOCH="$(date +%s)"; START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  python tools/train.py --config_file "${CONFIG}"
  python tools/finalize_g2_f_top2_tau0p5_experiment.py \
    --config-file "${CONFIG}" --output-dir "${OUTPUT_DIR}"
  python tools/package_g2_f_top2_tau0p5_result.py --output-dir "${OUTPUT_DIR}"
  bash scripts/export_g2_f_top2_tau0p5_result_autodl.sh "${OUTPUT_DIR}"
} 2>&1 | tee "${CONSOLE_LOG}"
END_EPOCH="$(date +%s)"; END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python tools/recover_g2_f_top2_tau0p5_experiment.py \
  --config-file "${CONFIG}" --output-dir "${OUTPUT_DIR}" --console-log "${CONSOLE_LOG}" \
  --started-at-utc "${START_UTC}" --ended-at-utc "${END_UTC}" \
  --runtime-seconds "$((END_EPOCH - START_EPOCH))"
printf 'Formal G2-F evidence and archive are ready under: %s\n' "${OUTPUT_DIR}"
