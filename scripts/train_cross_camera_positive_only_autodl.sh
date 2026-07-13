#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0

CONFIG="configs/softmax_triplet_cross_camera_positive_only_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/cross_camera_positive_only_market1501"

echo "Running Cross-Camera Positive Only experiment"
echo "Config: ${CONFIG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "EXPERIMENTS.md will be updated automatically after training succeeds."

python tools/train.py --config_file ${CONFIG}

echo "Training finished successfully. Updating EXPERIMENTS.md..."

python scripts/append_experiment_result.py \
  --config ${CONFIG} \
  --experiment-id C2-CCPO-Market \
  --note "Cross-camera positive only, no extra hard negative, Market1501, AutoDL" \
  --mode update

echo "EXPERIMENTS.md updated."
echo "To save the experiment record to GitHub, run:"
echo "git add EXPERIMENTS.md"
echo "git commit -m \"record cross camera positive only experiment result\""
echo "git push"
