#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="exp/c2l03-camera-conditional-part-attention"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Refusing CondPA formal training from branch '${CURRENT_BRANCH}'; expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing CondPA formal training from a dirty worktree." >&2
  git status --short >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_c2l03_camera_conditional_part_attention_autodl.yml"
python tools/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-id C2-L03-CONDPA-S42 \
  --experiment-family c2l03_camera_conditional_part_attention \
  --expected-branch "${EXPECTED_BRANCH}" \
  --method "C2-L03 + Camera-Conditional Part Attention" \
  --baseline-method C2-L03 \
  --baseline-commit 95769d09774c16cd56bf31800a3b6ecb0e1bce3e \
  --notes "Parallel A independent CondPA validation; Seed=42; camera embedding [6,6]; multi-granularity disabled; C2 lambda=0.3; retrieval protocol unchanged."
