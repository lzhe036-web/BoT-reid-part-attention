#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_cross_camera_positive_only_repeat_autodl.yml"
python tools/train.py --config_file "${CONFIG}"
python scripts/append_experiment_result.py --config "${CONFIG}" --experiment-id C2-CCPO-Repeat --note "C2 repeat run to verify stability, cross-camera positive only, no extra hard negative." --mode update
