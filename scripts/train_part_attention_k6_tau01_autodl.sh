#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

python tools/train.py --config_file configs/softmax_triplet_part_attention_k6_tau01_autodl.yml
