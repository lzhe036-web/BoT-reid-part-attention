#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_c2_baseline_control_autodl.yml"
python tools/train.py --config_file "${CONFIG}"
python scripts/append_experiment_result.py --config "${CONFIG}" --experiment-id C2-Baseline-Control --note "Baseline control under C2 setting, cross-camera positive loss disabled, other settings aligned with C2." --mode update
