#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_c2_l03_duke_baseline_autodl.yml"
python tools/train.py --config_file "${CONFIG}"
python scripts/append_experiment_result.py --config "${CONFIG}" --experiment-id Duke-Baseline-Control --note "DukeMTMC-reID baseline control: BoT + Part Attention, cross-camera positive loss disabled; settings aligned with C2-L03." --mode update
