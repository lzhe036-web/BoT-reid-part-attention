#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFF_OUTPUT="${G2_D1_OUTPUT:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_d1_tau0p5_seed42_market1501}"
ANALYSIS_ROOT="${G1_G2_ANALYSIS_DIR:?Set G1_G2_ANALYSIS_DIR to the existing completed G1/G2 fixed-sample analysis directory}"
G1_RUN_MANIFEST="${G1_RUN_MANIFEST:?Set formal G1 run_manifest.json}"
G2_RUN_MANIFEST="${G2_RUN_MANIFEST:?Set formal original-G2 run_manifest.json}"
G2A_RUN_MANIFEST="${G2A_RUN_MANIFEST:?Set formal G2-A tau0p5 run_manifest.json}"
DIFF_RUN_MANIFEST="${G2_D1_RUN_MANIFEST:?Set registered G2-D1 run_manifest.json}"
G2A_FIXED_GATES="${G2A_FIXED_GATES:?Set existing G2-A fixed gate TSV on the same candidate manifest}"
OUTPUT_DIR="${G2_D1_COMPARISON_OUTPUT:-${DIFF_OUTPUT}/g1_g2_g2a_g2_d1_fixed_sample_comparison}"

cd "${REPO_ROOT}"
test -f "${DIFF_OUTPUT}/g2_d1_formal_result.json"; test -f "${DIFF_OUTPUT}/config_resolved.yml"
test -f "${ANALYSIS_ROOT}/manifests/fixed_candidate_samples.tsv"; test -f "${ANALYSIS_ROOT}/manifests/g1_fixed_gating_samples.tsv"; test -f "${ANALYSIS_ROOT}/manifests/g2_fixed_gating_samples.tsv"
test -f "${G2A_FIXED_GATES}"; test -f "${G1_RUN_MANIFEST}"; test -f "${G2_RUN_MANIFEST}"; test -f "${G2A_RUN_MANIFEST}"; test -f "${DIFF_RUN_MANIFEST}"
test ! -e "${OUTPUT_DIR}" || { echo "Refusing to reuse ${OUTPUT_DIR}" >&2; exit 1; }
CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"]["path"])' "${DIFF_OUTPUT}/g2_d1_formal_result.json")"
python tools/compare_g1_g2_g2a_diff46_gating.py --candidate-manifest "${ANALYSIS_ROOT}/manifests/fixed_candidate_samples.tsv" \
  --g1-fixed-gates "${ANALYSIS_ROOT}/manifests/g1_fixed_gating_samples.tsv" --g2-fixed-gates "${ANALYSIS_ROOT}/manifests/g2_fixed_gating_samples.tsv" --g2a-fixed-gates "${G2A_FIXED_GATES}" \
  --g1-run-manifest "${G1_RUN_MANIFEST}" --g2-run-manifest "${G2_RUN_MANIFEST}" --g2a-run-manifest "${G2A_RUN_MANIFEST}" --diff46-run-manifest "${DIFF_RUN_MANIFEST}" \
  --config-file configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_diff46_autodl.yml --resolved-config "${DIFF_OUTPUT}/config_resolved.yml" \
  --checkpoint "${CHECKPOINT}" --dataset-root /root/autodl-tmp/datasets --output-dir "${OUTPUT_DIR}" --device cuda
printf 'Four-version fixed-sample comparison including G2-D1: %s\n' "${OUTPUT_DIR}"
