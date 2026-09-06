#!/usr/bin/env bash
# Post-training only: extract τg=0.5 gates on the pre-frozen G1/G2 sample set.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAU_OUTPUT="${TAU_OUTPUT:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_global_local_tau0p5_seed42_market1501}"
ANALYSIS_ROOT="${G1_G2_ANALYSIS_DIR:?Set G1_G2_ANALYSIS_DIR to the completed G1/G2 fixed-sample analysis directory}"
G1_RUN_MANIFEST="${G1_RUN_MANIFEST:?Set G1_RUN_MANIFEST to the formal G1 run_manifest.json}"
G2_RUN_MANIFEST="${G2_RUN_MANIFEST:?Set G2_RUN_MANIFEST to the formal G2 tau=1.0 run_manifest.json}"
TAU_RUN_MANIFEST="${TAU_RUN_MANIFEST:?Set TAU_RUN_MANIFEST to the registered tau=0.5 run_manifest.json}"
OUTPUT_DIR="${TAU_COMPARISON_OUTPUT:-${TAU_OUTPUT}/g1_g2_tau0p5_fixed_sample_comparison}"

cd "${REPO_ROOT}"
test -f "${TAU_OUTPUT}/g2_tau0p5_formal_result.json"
test -f "${TAU_OUTPUT}/config_resolved.yml"
test -f "${TAU_OUTPUT}/checkpoint_manifest.tsv"
test -f "${ANALYSIS_ROOT}/manifests/fixed_candidate_samples.tsv"
test -f "${ANALYSIS_ROOT}/manifests/g1_fixed_gating_samples.tsv"
test -f "${ANALYSIS_ROOT}/manifests/g2_fixed_gating_samples.tsv"
test -f "${G1_RUN_MANIFEST}" && test -f "${G2_RUN_MANIFEST}" && test -f "${TAU_RUN_MANIFEST}"
test ! -e "${OUTPUT_DIR}" || { echo "Refusing to reuse ${OUTPUT_DIR}" >&2; exit 1; }

CHECKPOINT="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"]["path"])' "${TAU_OUTPUT}/g2_tau0p5_formal_result.json")"
test -f "${CHECKPOINT}"

python tools/compare_g1_g2_tau0p5_gating.py \
  --candidate-manifest "${ANALYSIS_ROOT}/manifests/fixed_candidate_samples.tsv" \
  --g1-fixed-gates "${ANALYSIS_ROOT}/manifests/g1_fixed_gating_samples.tsv" \
  --g2-fixed-gates "${ANALYSIS_ROOT}/manifests/g2_fixed_gating_samples.tsv" \
  --g1-run-manifest "${G1_RUN_MANIFEST}" --g2-run-manifest "${G2_RUN_MANIFEST}" \
  --tau-run-manifest "${TAU_RUN_MANIFEST}" \
  --config-file configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau0p5_autodl.yml \
  --resolved-config "${TAU_OUTPUT}/config_resolved.yml" --checkpoint "${CHECKPOINT}" \
  --dataset-root /root/autodl-tmp/datasets --output-dir "${OUTPUT_DIR}" --device cuda

printf 'Fixed-sample G1/G2/tau=0.5 comparison: %s\n' "${OUTPUT_DIR}"
