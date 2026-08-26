#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="codex/g2-global-local-gating"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_global_local_tau1_seed42_market1501"

cd "${REPO_ROOT}"
CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_COMMIT="$(git rev-parse HEAD)"
REMOTE_COMMIT="$(git rev-parse "origin/${EXPECTED_BRANCH}")"

if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  printf 'Expected branch %s, got %s\n' "${EXPECTED_BRANCH}" "${CURRENT_BRANCH}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  printf 'Formal G2 training requires a clean worktree.\n' >&2
  exit 1
fi
if [[ "${CURRENT_COMMIT}" != "${REMOTE_COMMIT}" ]]; then
  printf 'Local G2 commit does not match origin/%s.\n' "${EXPECTED_BRANCH}" >&2
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

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python tools/train.py --config_file "${CONFIG}"
python tools/finalize_g2_global_local_experiment.py \
  --config-file "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}"
