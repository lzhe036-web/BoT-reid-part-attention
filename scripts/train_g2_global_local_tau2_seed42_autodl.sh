#!/usr/bin/env bash
# Formal τg=2.0 candidate. It never reuses τg=0.5 or τg=1.0 output paths.
set -euo pipefail

EXPECTED_BRANCH="codex/g2-global-local-gating-tau2"
TAU0P5_BASE_COMMIT="d724a6536e4a819c5d2932412e90b7dea224041b"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau2_autodl.yml"
BASE_CONFIG="${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau0p5_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_global_local_tau2_seed42_market1501"
CONSOLE_LOG="${OUTPUT_DIR}.console.log"

cd "${REPO_ROOT}"
CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  printf 'Expected branch %s, got %s\n' "${EXPECTED_BRANCH}" "${CURRENT_BRANCH}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  printf 'Formal τg=2.0 training requires a clean worktree.\n' >&2
  exit 1
fi
if ! git merge-base --is-ancestor "${TAU0P5_BASE_COMMIT}" "${CURRENT_COMMIT}"; then
  printf 'Candidate commit is not descended from τg=0.5 base %s.\n' "${TAU0P5_BASE_COMMIT}" >&2
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
if [[ -e "${OUTPUT_DIR}" || -e "${CONSOLE_LOG}" ]]; then
  printf 'Refusing to reuse formal output or console path.\n' >&2
  exit 1
fi

python tools/verify_g2_tau0p5_protocol.py \
  --baseline-config "${BASE_CONFIG}" --candidate-config "${CONFIG}" \
  --expected-baseline-tau 0.5 --expected-candidate-tau 2.0

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN_STARTED_EPOCH="$(date +%s)"
RUN_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  python tools/train.py --config_file "${CONFIG}"
  python tools/finalize_g2_global_local_experiment.py \
    --config-file "${CONFIG}" --output-dir "${OUTPUT_DIR}" \
    --expected-branch "${EXPECTED_BRANCH}" --expected-gating-tau 2.0 \
    --experiment-label "G2 global-plus-local Dynamic Gating (tau=2.0)" \
    --result-filename g2_tau2_formal_result.json \
    --analysis-dirname g2_tau2_gating_analysis
} 2>&1 | tee "${CONSOLE_LOG}"
RUN_ENDED_EPOCH="$(date +%s)"
RUN_ENDED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_RUNTIME_SECONDS="$((RUN_ENDED_EPOCH - RUN_STARTED_EPOCH))"

python tools/recover_g2_global_local_tau2_experiment.py \
  --config-file "${CONFIG}" --output-dir "${OUTPUT_DIR}" \
  --console-log "${CONSOLE_LOG}" --started-at-utc "${RUN_STARTED_UTC}" \
  --ended-at-utc "${RUN_ENDED_UTC}" --runtime-seconds "${RUN_RUNTIME_SECONDS}"

printf 'Formal τg=2.0 evidence was registered. Output: %s\n' "${OUTPUT_DIR}"
