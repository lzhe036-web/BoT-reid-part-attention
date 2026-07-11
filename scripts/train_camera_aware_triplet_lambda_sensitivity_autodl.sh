#!/usr/bin/env bash
set -e

bash scripts/train_camera_aware_triplet_lambda01_autodl.sh
bash scripts/train_camera_aware_triplet_lambda03_autodl.sh
bash scripts/train_camera_aware_triplet_lambda05_autodl.sh
bash scripts/train_camera_aware_triplet_lambda10_autodl.sh
