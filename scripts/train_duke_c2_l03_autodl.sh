#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/softmax_triplet_c2_l03_duke_autodl.yml"
python tools/train.py --config_file "${CONFIG}"
python scripts/append_experiment_result.py --config "${CONFIG}" --experiment-id Duke-C2-L03 --note "DukeMTMC-reID C2-L03: cross-camera positive only, lambda=0.3, mode=mean; no extra hard negative mining/weighting, same-camera positive, or hierarchical difficulty." --mode update
