#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

CONFIG="${1:-configs/softmax_triplet_normalized_weighted_loss_autodl.yml}"

OUTPUT_DIR="$(python -c "from config import cfg; cfg.merge_from_file('$CONFIG'); print(cfg.OUTPUT_DIR)")"
mkdir -p "$OUTPUT_DIR"

INFO_FILE="$OUTPUT_DIR/run_info.txt"

{
  echo "record_time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "branch: $(git branch --show-current || true)"
  echo "commit_id: $(git rev-parse HEAD || true)"
  echo "git_status:"
  git status --short || true
  echo "config: $CONFIG"
  echo "output_dir: $OUTPUT_DIR"
  echo "python: $(which python || true)"
  echo "nvidia_smi:"
  nvidia-smi || true
} > "$INFO_FILE"

echo "Experiment info written to $INFO_FILE"
