#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0

MARKET_CKPT="${MARKET_CKPT:-/root/autodl-tmp/experiments/BoT/softmax_triplet_part_attention_market1501/resnet50_checkpoint_22320.pt}"

if [ ! -f "${MARKET_CKPT}" ]; then
  echo "Market1501 checkpoint not found: ${MARKET_CKPT}" >&2
  echo "Set MARKET_CKPT to an existing Market1501-trained checkpoint." >&2
  exit 1
fi

python tools/test.py --config_file configs/test_market1501_debias_off_autodl.yml TEST.WEIGHT "${MARKET_CKPT}"
python scripts/append_camera_bias_result.py \
  --experiment-id D1-Market-Debias-Off \
  --config configs/test_market1501_debias_off_autodl.yml \
  --checkpoint "${MARKET_CKPT}" \
  --train-set Market1501 \
  --test-set Market1501 \
  --camera-debias False \
  --note "D1 Market1501 same-domain, camera debias off, joint query-gallery camera mean protocol" \
  --mode update

python tools/test.py --config_file configs/test_market1501_debias_on_autodl.yml TEST.WEIGHT "${MARKET_CKPT}"
python scripts/append_camera_bias_result.py \
  --experiment-id D1-Market-Debias-On \
  --config configs/test_market1501_debias_on_autodl.yml \
  --checkpoint "${MARKET_CKPT}" \
  --train-set Market1501 \
  --test-set Market1501 \
  --camera-debias True \
  --note "D1 Market1501 same-domain, camera debias on, joint query-gallery camera mean protocol" \
  --mode update
