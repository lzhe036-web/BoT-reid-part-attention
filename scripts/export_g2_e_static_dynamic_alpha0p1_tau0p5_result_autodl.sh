#!/usr/bin/env bash
# Revalidate/recover before building a unique new delivery; never retrain.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
OUTPUT_DIR="${1:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_e_static_dynamic_alpha0p1_tau0p5_seed42_market1501}"
python tools/finish_g2_e_alpha_experiment.py --alpha 0.1 \
  --config-file "${REPO_ROOT}/configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_e_static_dynamic_alpha0p1_tau0p5_autodl.yml" \
  --output-dir "${OUTPUT_DIR}" --console-log "${OUTPUT_DIR}.console.log" \
  --archive-dir "${G2_E_EXPORT_ROOT:-/root/autodl-tmp/exports}"
