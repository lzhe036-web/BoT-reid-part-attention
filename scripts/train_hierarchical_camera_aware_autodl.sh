#!/usr/bin/env bash
set -e

CONFIG_FILE="configs/softmax_triplet_hierarchical_camera_aware_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/hierarchical_camera_aware_market1501"
EXPERIMENT_ID="HCAL001"
NOTE="Hierarchical camera-aware triplet loss, weighted negatives, lambda=0.5, margin=0.3, AutoDL"

export CUDA_VISIBLE_DEVICES=0

echo "Running Hierarchical Camera-Aware Triplet Loss experiment"
echo "Config file: ${CONFIG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "EXPERIMENTS.md will be updated automatically after training succeeds."

python tools/train.py --config_file "${CONFIG_FILE}"

echo "Training finished successfully. Updating EXPERIMENTS.md..."
python scripts/append_experiment_result.py \
  --config "${CONFIG_FILE}" \
  --experiment-id "${EXPERIMENT_ID}" \
  --section-title "## Hierarchical Camera-Aware Loss Experiments" \
  --note "${NOTE}" \
  --mode update

echo "EXPERIMENTS.md updated."
echo "To save the experiment record to GitHub, run:"
echo "  git add EXPERIMENTS.md"
echo "  git commit -m \"record hierarchical camera aware experiment result\""
echo "  git push"
