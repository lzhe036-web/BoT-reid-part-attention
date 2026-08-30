#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="codex/g2-without-z4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_without_z4_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_without_z4_tau1_seed42_market1501"
CONSOLE_LOG="${OUTPUT_DIR}.console.log"

cd "${REPO_ROOT}"
CURRENT_BRANCH="$(git branch --show-current)"; CURRENT_COMMIT="$(git rev-parse HEAD)"; REMOTE_COMMIT="$(git rev-parse --verify "origin/${EXPECTED_BRANCH}")"
[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] || { echo "Expected branch ${EXPECTED_BRANCH}" >&2; exit 1; }
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || { echo "Formal G2-without-z4 training requires a clean worktree." >&2; exit 1; }
[[ "${CURRENT_COMMIT}" == "${REMOTE_COMMIT}" ]] || { echo "Local commit differs from origin/${EXPECTED_BRANCH}" >&2; exit 1; }
[[ -d /root/autodl-tmp/datasets ]] || { echo "Dataset root is absent." >&2; exit 1; }
[[ -f /root/autodl-tmp/pretrained/resnet50-19c8e357.pth ]] || { echo "ImageNet weights are absent." >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}" && ! -e "${CONSOLE_LOG}" ]] || { echo "Refusing to reuse formal output or console log." >&2; exit 1; }
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN_STARTED_EPOCH="$(date +%s)"; RUN_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
python tools/train.py --config_file "${CONFIG}" 2>&1 | tee "${CONSOLE_LOG}"; TRAIN_STATUS="${PIPESTATUS[0]}"
set -e
[[ "${TRAIN_STATUS}" -eq 0 ]] || { echo "Training failed; preserving output." >&2; exit "${TRAIN_STATUS}"; }
set +e
python tools/finalize_g2_without_z4_experiment.py --config-file "${CONFIG}" --output-dir "${OUTPUT_DIR}" 2>&1 | tee -a "${CONSOLE_LOG}"; FINALIZER_STATUS="${PIPESTATUS[0]}"
set -e
[[ "${FINALIZER_STATUS}" -eq 0 ]] || { echo "Finalizer failed; preserving output." >&2; exit "${FINALIZER_STATUS}"; }
RUN_ENDED_EPOCH="$(date +%s)"; RUN_ENDED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; RUN_RUNTIME_SECONDS="$((RUN_ENDED_EPOCH - RUN_STARTED_EPOCH))"
python tools/recover_g2_without_z4_experiment.py --config-file "${CONFIG}" --output-dir "${OUTPUT_DIR}" --console-log "${CONSOLE_LOG}" --started-at-utc "${RUN_STARTED_UTC}" --ended-at-utc "${RUN_ENDED_UTC}" --runtime-seconds "${RUN_RUNTIME_SECONDS}"
