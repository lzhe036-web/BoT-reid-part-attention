#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="codex/g2-d1"
BASELINE_COMMIT="d724a6536e4a819c5d2932412e90b7dea224041b"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_diff46_autodl.yml"
BASE_CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau0p5_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_d1_tau0p5_seed42_market1501"
CONSOLE_LOG="${OUTPUT_DIR}.console.log"

cd "${REPO_ROOT}"
CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_COMMIT="$(git rev-parse HEAD)"
REMOTE_COMMIT="$(git rev-parse "origin/${EXPECTED_BRANCH}")"
test "${CURRENT_BRANCH}" = "${EXPECTED_BRANCH}" || { echo "Wrong branch: ${CURRENT_BRANCH}" >&2; exit 1; }
test -z "$(git status --porcelain=v1 --untracked-files=all)" || { echo "Formal training requires a clean worktree." >&2; exit 1; }
test "${CURRENT_COMMIT}" = "${REMOTE_COMMIT}" || { echo "Local HEAD differs from origin/${EXPECTED_BRANCH}." >&2; exit 1; }
git merge-base --is-ancestor "${BASELINE_COMMIT}" "${CURRENT_COMMIT}" || { echo "G2-D1 is not descended from G2-A tau0p5." >&2; exit 1; }
test -d /root/autodl-tmp/datasets || { echo "Dataset root is absent." >&2; exit 1; }
test -f /root/autodl-tmp/pretrained/resnet50-19c8e357.pth || { echo "ImageNet pretraining is absent." >&2; exit 1; }
test ! -e "${OUTPUT_DIR}" || { echo "Refusing to reuse ${OUTPUT_DIR}" >&2; exit 1; }
test ! -e "${CONSOLE_LOG}" || { echo "Refusing to reuse ${CONSOLE_LOG}" >&2; exit 1; }
python tools/verify_g2_diff46_protocol.py --baseline-config "${BASE_CONFIG}" --candidate-config "${CONFIG}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN_STARTED_EPOCH="$(date +%s)"; RUN_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  python tools/train.py --config_file "${CONFIG}"
  python tools/finalize_g2_diff46_experiment.py --config-file "${CONFIG}" --output-dir "${OUTPUT_DIR}"
} 2>&1 | tee "${CONSOLE_LOG}"
RUN_ENDED_EPOCH="$(date +%s)"; RUN_ENDED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python tools/recover_g2_diff46_experiment.py --config-file "${CONFIG}" --output-dir "${OUTPUT_DIR}" \
  --console-log "${CONSOLE_LOG}" --started-at-utc "${RUN_STARTED_UTC}" --ended-at-utc "${RUN_ENDED_UTC}" \
  --runtime-seconds "$((RUN_ENDED_EPOCH - RUN_STARTED_EPOCH))"
printf 'G2-D1 formal evidence and single-version package: %s\n' "${OUTPUT_DIR}"
