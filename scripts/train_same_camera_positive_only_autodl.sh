#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_same_camera_positive_only_autodl.yml"
python tools/train.py --config_file "${CONFIG}"
python scripts/append_experiment_result.py --config "${CONFIG}" --experiment-id S2-SCPO-Market --note "Same-camera positive only, no negative or hard-negative mining, Market1501, AutoDL." --mode update
