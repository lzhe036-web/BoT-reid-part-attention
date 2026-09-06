#!/usr/bin/env bash
# Post-training only: full-query G1/G2/E1 comparison and deliverable package.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
E1_OUTPUT="${E1_OUTPUT:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_e1_static_dynamic_alpha0p5_tau1_seed42_market1501}"
G1_CONFIG="${G1_CONFIG:?Set G1_CONFIG to the formal G1 source/resolved config}"
G1_CHECKPOINT="${G1_CHECKPOINT:?Set G1_CHECKPOINT to the selected formal G1 checkpoint}"
G1_RESULT="${G1_RESULT:?Set G1_RESULT to formal G1 result JSON or registered run_manifest.json}"
G2_CONFIG="${G2_CONFIG:?Set G2_CONFIG to the formal original-G2 source/resolved config}"
G2_CHECKPOINT="${G2_CHECKPOINT:?Set G2_CHECKPOINT to the selected formal G2 checkpoint}"
G2_RESULT="${G2_RESULT:?Set G2_RESULT to formal G2 result JSON or registered run_manifest.json}"
E1_RUN_MANIFEST="${E1_RUN_MANIFEST:?Set E1_RUN_MANIFEST to the registered E1 run_manifest.json}"
OUTPUT_DIR="${E1_COMPARISON_OUTPUT:-${E1_OUTPUT}/g1_g2_e1_static_dynamic_complete_query_comparison}"

cd "${REPO_ROOT}"
test -f "${E1_OUTPUT}/g2_e1_static_dynamic_alpha0p5_formal_result.json"
test -f "${E1_OUTPUT}/config_resolved.yml"
test -f "${G1_CONFIG}" && test -f "${G1_CHECKPOINT}" && test -f "${G1_RESULT}"
test -f "${G2_CONFIG}" && test -f "${G2_CHECKPOINT}" && test -f "${G2_RESULT}"
test -f "${E1_RUN_MANIFEST}"
test ! -e "${OUTPUT_DIR}" || { echo "Refusing to reuse ${OUTPUT_DIR}" >&2; exit 1; }

E1_CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"]["path"])' "${E1_OUTPUT}/g2_e1_static_dynamic_alpha0p5_formal_result.json")"
test -f "${E1_CHECKPOINT}"

python tools/compare_g1_g2_e1_static_dynamic.py \
  --g1-config "${G1_CONFIG}" --g1-checkpoint "${G1_CHECKPOINT}" --g1-result "${G1_RESULT}" \
  --g2-config "${G2_CONFIG}" --g2-checkpoint "${G2_CHECKPOINT}" --g2-result "${G2_RESULT}" \
  --e1-config configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_e1_static_dynamic_alpha0p5_autodl.yml \
  --e1-resolved-config "${E1_OUTPUT}/config_resolved.yml" --e1-checkpoint "${E1_CHECKPOINT}" \
  --e1-result "${E1_OUTPUT}/g2_e1_static_dynamic_alpha0p5_formal_result.json" \
  --e1-run-manifest "${E1_RUN_MANIFEST}" --output-dir "${OUTPUT_DIR}" --device cuda

printf 'E1 full-query G1/G2 comparison package: %s\n' "${OUTPUT_DIR}"
