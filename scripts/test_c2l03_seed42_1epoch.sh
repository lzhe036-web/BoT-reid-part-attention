#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="exp/c2l03-seed42-reproducible"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Refusing C2-L03 Seed42 smoke test from branch '${CURRENT_BRANCH}'; expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing C2-L03 Seed42 smoke test from a dirty worktree." >&2
  git status --short >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_c2l03_seed42_autodl.yml"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/root/autodl-tmp/experiments/BoT/c2l03_seed42_market1501_smoke_1epoch}"
python tools/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-id C2-L03-S42-SMOKE-1E \
  --experiment-family c2l03_seed42_reproducible_smoke \
  --expected-branch "${EXPECTED_BRANCH}" \
  --method "C2-L03 (Unified Reproducible Protocol, Seed 42)" \
  --baseline-method C2-L03 \
  --baseline-commit dca6dc1dd890d47dbbbaf192de14c9ab5402afb0 \
  --notes "Prepared 1-epoch C2-L03 Seed42 smoke test; not a formal paper result." \
  SOLVER.MAX_EPOCHS 1 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  SOLVER.EVAL_PERIOD 1 \
  OUTPUT_DIR "${SMOKE_OUTPUT_DIR}"
