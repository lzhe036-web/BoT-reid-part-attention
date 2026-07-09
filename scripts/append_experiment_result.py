#!/usr/bin/env python
# encoding: utf-8

import argparse
import glob
import os
import re
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import cfg


PENDING = "待填写"
SECTION_TITLE = "## Camera-Aware Triplet Loss Experiments"
HEADER = "| 实验编号 | 日期 | commit id | 分支 | config 文件 | seed | GPU | 数据集 | 运行时间 | best epoch | Rank-1 | mAP | 备注 |"
SEPARATOR = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def run_command(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return PENDING


def get_gpu_name():
    output = run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if output == PENDING:
        return PENDING
    return output.splitlines()[0].strip() if output.splitlines() else PENDING


def normalize_dataset_name(name):
    if isinstance(name, (list, tuple)):
        name = name[0] if name else ""
    mapping = {
        "market1501": "Market1501",
        "dukemtmc": "DukeMTMC-reID",
        "dukemtmc-reid": "DukeMTMC-reID",
    }
    return mapping.get(str(name).lower(), str(name) if name else PENDING)


def collect_config(config_file):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(config_file)
    return {
        "dataset": normalize_dataset_name(local_cfg.DATASETS.NAMES),
        "output_dir": local_cfg.OUTPUT_DIR,
        "seed": str(local_cfg.SEED) if "SEED" in local_cfg else PENDING,
    }


def parse_metrics(output_dir):
    result = {
        "runtime": PENDING,
        "best_epoch": PENDING,
        "rank1": PENDING,
        "map": PENDING,
    }
    if not output_dir or not os.path.isdir(output_dir):
        return result

    log_paths = sorted(
        glob.glob(os.path.join(output_dir, "*.txt")) + glob.glob(os.path.join(output_dir, "*.log")),
        key=os.path.getmtime,
        reverse=True,
    )
    best = None
    for path in log_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()
        except OSError:
            continue

        current_epoch = None
        current_map = None
        for line in lines:
            epoch_match = re.search(r"Validation Results - Epoch:\s*(\d+)", line)
            if epoch_match:
                current_epoch = epoch_match.group(1)
                current_map = None
                continue

            map_match = re.search(r"mAP:\s*([0-9.]+%)", line)
            if map_match and current_epoch is not None:
                current_map = map_match.group(1)
                continue

            rank_match = re.search(r"Rank-1\s*:\s*([0-9.]+%)", line)
            if rank_match and current_epoch is not None:
                rank1 = rank_match.group(1)
                rank_value = float(rank1.rstrip("%"))
                map_value = float(current_map.rstrip("%")) if current_map else -1.0
                candidate = (rank_value, map_value, current_epoch, rank1, current_map or PENDING)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate

    if best is not None:
        result["best_epoch"] = best[2]
        result["rank1"] = best[3]
        result["map"] = best[4]
    return result


def build_row(args):
    config_info = collect_config(args.config)
    metrics = parse_metrics(config_info["output_dir"])
    values = [
        args.experiment_id,
        datetime.now().strftime("%Y-%m-%d"),
        run_command(["git", "rev-parse", "--short", "HEAD"]),
        run_command(["git", "branch", "--show-current"]),
        args.config,
        config_info["seed"],
        get_gpu_name(),
        config_info["dataset"],
        metrics["runtime"],
        metrics["best_epoch"],
        metrics["rank1"],
        metrics["map"],
        args.note,
    ]
    return "| " + " | ".join(values) + " |"


def ensure_section(content):
    if SECTION_TITLE in content:
        return content
    section = "\n\n{}\n\n{}\n{}\n".format(SECTION_TITLE, HEADER, SEPARATOR)
    return content.rstrip() + section


def update_experiments(row, experiment_id, path="EXPERIMENTS.md"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
    else:
        content = "# Experiments\n"

    content = ensure_section(content)
    lines = content.splitlines()
    replaced = False
    section_seen = False
    insert_at = len(lines)

    for index, line in enumerate(lines):
        if line.strip() == SECTION_TITLE:
            section_seen = True
            continue
        if section_seen and line.startswith("## ") and line.strip() != SECTION_TITLE:
            insert_at = index
            break
        if section_seen and line.startswith("| {} |".format(experiment_id)):
            lines[index] = row
            replaced = True
            break
        if section_seen and line.startswith("| "):
            insert_at = index + 1

    if not replaced:
        lines.insert(insert_at, row)

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Append or update experiment results in EXPERIMENTS.md")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--mode", choices=["dry-run", "update"], default="dry-run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    row = build_row(args)
    print(row)

    if args.dry_run or args.mode == "dry-run":
        return

    update_experiments(row, args.experiment_id)


if __name__ == "__main__":
    main()
