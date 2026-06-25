#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Project dir: $(pwd)"
echo "Python: $(which python)"
echo "Commit: $(git rev-parse HEAD)"
echo "Git status:"
git status --short

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("cuda device name:", torch.cuda.get_device_name(0))
PY

nvidia-smi || true

DATA_DIR="/root/autodl-tmp/data"
MARKET_DIR="$DATA_DIR/market1501"
PRETRAIN_PATH="/root/autodl-tmp/pretrained/resnet50-19c8e357.pth"
CONFIG="configs/softmax_triplet_adaptive_hard_autodl.yml"
OUTPUT_DIR="/root/autodl-tmp/log/adaptive_hard_triplet_tau02"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: dataset root not found: $DATA_DIR"
    echo "Please put Market1501 under /root/autodl-tmp/data/market1501"
    exit 1
fi

if [ ! -d "$MARKET_DIR" ]; then
    echo "ERROR: Market1501 directory not found: $MARKET_DIR"
    echo "Expected: $MARKET_DIR/bounding_box_train, $MARKET_DIR/query, $MARKET_DIR/bounding_box_test"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG"
    exit 1
fi

if [ ! -f "$PRETRAIN_PATH" ]; then
    echo "ERROR: pretrained weight not found: $PRETRAIN_PATH"
    echo "Please put resnet50-19c8e357.pth under /root/autodl-tmp/pretrained"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python tools/train.py --config_file "$CONFIG"

echo "Experiment record: $OUTPUT_DIR/experiment_record.csv"
