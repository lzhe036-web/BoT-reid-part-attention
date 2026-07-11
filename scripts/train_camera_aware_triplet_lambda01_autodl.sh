#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0

CONFIG="configs/softmax_triplet_camera_aware_lambda01_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/experiments/BoT/camera_aware_triplet_lambda01_market1501"

echo "Running BoT + L_camera_triplet lambda=0.1 experiment"
echo "Config: ${CONFIG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "EXPERIMENTS.md will be updated automatically after training succeeds."
python tools/train.py --config_file "${CONFIG}"
echo "Training finished successfully. Updating EXPERIMENTS.md..."
python scripts/append_experiment_result.py --config "${CONFIG}" --experiment-id CAT-L01 --note "BoT + L_camera_triplet, lambda=0.1, camera-aware hard triplet, AutoDL" --mode update
echo "EXPERIMENTS.md updated."
echo "To save the experiment record to GitHub, run:"
echo "git add EXPERIMENTS.md"
echo "git commit -m \"record camera aware triplet lambda sensitivity result\""
echo "git push"
