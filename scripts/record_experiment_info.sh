#!/usr/bin/env bash
set -e

CONFIG_FILE="${1:-configs/softmax_triplet_part_attention_autodl.yml}"

echo "date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "branch: $(git branch --show-current)"
echo "commit id: $(git rev-parse --short HEAD)"
echo "config file: ${CONFIG_FILE}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU:"
  nvidia-smi --query-gpu=name --format=csv,noheader
else
  echo "GPU: nvidia-smi not found"
fi

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY
