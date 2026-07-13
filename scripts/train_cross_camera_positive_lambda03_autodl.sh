#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_cross_camera_positive_lambda03_autodl.yml"
python tools/train.py --config_file "${CONFIG}"
python scripts/append_experiment_result.py --config "${CONFIG}" --experiment-id C2-L03 --note "C2 lambda sensitivity, cross-camera positive only, lambda=0.3, no extra hard negative." --mode update
