#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
python tools/train.py --config_file configs/softmax_triplet_normalized_weighted_loss_autodl.yml
