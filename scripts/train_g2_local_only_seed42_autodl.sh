#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="codex/g2-local-only"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_local_only_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_local_only_tau1_seed42_market1501"
CONSOLE_LOG="${OUTPUT_DIR}.console.log"

cd "${REPO_ROOT}"
CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_COMMIT="$(git rev-parse HEAD)"
REMOTE_COMMIT="$(git rev-parse --verify "origin/${EXPECTED_BRANCH}")"

if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  printf 'Expected branch %s, got %s\n' "${EXPECTED_BRANCH}" "${CURRENT_BRANCH}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  printf 'Formal G2-local-only training requires a clean worktree.\n' >&2
  exit 1
fi
if [[ "${CURRENT_COMMIT}" != "${REMOTE_COMMIT}" ]]; then
  printf 'Local G2-local-only commit does not match origin/%s.\n' "${EXPECTED_BRANCH}" >&2
  exit 1
fi
if [[ ! -d /root/autodl-tmp/datasets ]]; then
  printf 'Dataset root is absent: /root/autodl-tmp/datasets\n' >&2
  exit 1
fi
if [[ ! -f /root/autodl-tmp/pretrained/resnet50-19c8e357.pth ]]; then
  printf 'ImageNet weights are absent.\n' >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'Refusing to reuse formal output directory: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ -e "${CONSOLE_LOG}" ]]; then
  printf 'Refusing to reuse formal console log: %s\n' "${CONSOLE_LOG}" >&2
  exit 1
fi

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN_STARTED_EPOCH="$(date +%s)"
RUN_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e
python tools/train.py --config_file "${CONFIG}" 2>&1 | tee "${CONSOLE_LOG}"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  printf 'G2-local-only training failed; preserving OUTPUT_DIR and console log.\n' >&2
  exit "${TRAIN_STATUS}"
fi

set +e
python tools/finalize_g2_local_only_experiment.py \
  --config-file "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" 2>&1 | tee -a "${CONSOLE_LOG}"
FINALIZER_STATUS="${PIPESTATUS[0]}"
set -e
if [[ "${FINALIZER_STATUS}" -ne 0 ]]; then
  printf 'G2-local-only finalizer failed; preserving OUTPUT_DIR and console log.\n' >&2
  exit "${FINALIZER_STATUS}"
fi

RUN_ENDED_EPOCH="$(date +%s)"
RUN_ENDED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_RUNTIME_SECONDS="$((RUN_ENDED_EPOCH - RUN_STARTED_EPOCH))"

python tools/recover_g2_local_only_experiment.py \
  --config-file "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --console-log "${CONSOLE_LOG}" \
  --started-at-utc "${RUN_STARTED_UTC}" \
  --ended-at-utc "${RUN_ENDED_UTC}" \
  --runtime-seconds "${RUN_RUNTIME_SECONDS}"
