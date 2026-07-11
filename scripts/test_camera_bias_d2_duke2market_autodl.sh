#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0

DUKE_CKPT="${DUKE_CKPT:-/root/autodl-tmp/experiments/BoT/dukemtmc_reid_baseline/resnet50_checkpoint_xxx.pt}"

if [ ! -f "${DUKE_CKPT}" ]; then
  echo "DukeMTMC-reID checkpoint not found: ${DUKE_CKPT}" >&2
  echo "Modify DUKE_CKPT in this script or pass an existing Duke-trained checkpoint via the DUKE_CKPT environment variable." >&2
  exit 1
fi

python tools/test.py --config_file configs/test_duke2market_debias_off_autodl.yml TEST.WEIGHT "${DUKE_CKPT}"
python scripts/append_camera_bias_result.py \
  --experiment-id D2-Duke2Market-Debias-Off \
  --config configs/test_duke2market_debias_off_autodl.yml \
  --checkpoint "${DUKE_CKPT}" \
  --train-set DukeMTMC-reID \
  --test-set Market1501 \
  --camera-debias False \
  --note "D2 DukeMTMC-reID to Market1501 cross-domain, camera debias off, joint query-gallery camera mean protocol" \
  --mode update

python tools/test.py --config_file configs/test_duke2market_debias_on_autodl.yml TEST.WEIGHT "${DUKE_CKPT}"
python scripts/append_camera_bias_result.py \
  --experiment-id D2-Duke2Market-Debias-On \
  --config configs/test_duke2market_debias_on_autodl.yml \
  --checkpoint "${DUKE_CKPT}" \
  --train-set DukeMTMC-reID \
  --test-set Market1501 \
  --camera-debias True \
  --note "D2 DukeMTMC-reID to Market1501 cross-domain, camera debias on, joint query-gallery camera mean protocol" \
  --mode update
