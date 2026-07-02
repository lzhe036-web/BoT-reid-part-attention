#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import re
import subprocess
from datetime import datetime

import sys
sys.path.append('.')
from config import cfg


SECTION_TITLE = "## Normalized Weighted Loss Experiments"
TABLE_HEADER = (
    "| 实验编号 | 日期 | commit id | 分支 | config 文件 | seed | GPU | 数据集 | "
    "运行时间 | best epoch | Rank-1 | mAP | 备注 |"
)
TABLE_SEPARATOR = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
UNKNOWN = "待填写"
UNFIXED = "未固定"


def run_command(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT).decode("utf-8").strip()
    except Exception:
        return ""


def format_percent(value):
    if value is None:
        return UNKNOWN
    return "{:.1f}%".format(value * 100.0)


def parse_percent(text):
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", text)
    if not match:
        return None
    return float(match.group(1)) / 100.0


def parse_log(log_path):
    if not os.path.exists(log_path):
        return {
            "runtime": UNKNOWN,
            "best_epoch": UNKNOWN,
            "rank1": UNKNOWN,
            "mAP": UNKNOWN,
        }

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    timestamps = []
    validation_results = []
    current_epoch = None
    current_mAP = None

    for line in lines:
        ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if ts_match:
            try:
                timestamps.append(datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass

        epoch_match = re.search(r"Validation Results - Epoch:\s*(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            current_mAP = None
            continue

        if current_epoch is not None and "mAP:" in line:
            current_mAP = parse_percent(line)
            continue

        rank1_match = re.search(r"CMC curve,\s*Rank-1\s*:\s*([0-9]+(?:\.[0-9]+)?)%", line)
        if current_epoch is not None and rank1_match:
            rank1 = float(rank1_match.group(1)) / 100.0
            validation_results.append({
                "epoch": current_epoch,
                "rank1": rank1,
                "mAP": current_mAP,
            })
            current_epoch = None
            current_mAP = None

    runtime = UNKNOWN
    if len(timestamps) >= 2:
        seconds = int((timestamps[-1] - timestamps[0]).total_seconds())
        if seconds >= 0:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            runtime = "{}h {}m {}s".format(hours, minutes, secs) if hours else "{}m {}s".format(minutes, secs)

    valid_results = [item for item in validation_results if item["mAP"] is not None]
    if not valid_results:
        return {
            "runtime": runtime,
            "best_epoch": UNKNOWN,
            "rank1": UNKNOWN,
            "mAP": UNKNOWN,
        }

    best = max(valid_results, key=lambda item: item["mAP"])
    return {
        "runtime": runtime,
        "best_epoch": str(best["epoch"]),
        "rank1": format_percent(best["rank1"]),
        "mAP": format_percent(best["mAP"]),
    }


def get_gpu():
    gpu = run_command([
        "nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader",
    ])
    if gpu:
        return gpu.splitlines()[0].strip()
    return UNKNOWN


def get_dataset_name(value):
    if isinstance(value, (list, tuple)):
        raw = value[0] if value else ""
    else:
        raw = value
    mapping = {
        "market1501": "Market1501",
        "dukemtmc": "DukeMTMC-reID",
        "msmt17": "MSMT17",
        "cuhk03": "CUHK03",
        "veri": "VeRi",
    }
    return mapping.get(str(raw).strip("'\""), str(raw) if raw else UNKNOWN)


def get_seed():
    try:
        seed = cfg.SOLVER.SEED
    except AttributeError:
        return UNFIXED
    return str(seed) if seed is not None else UNFIXED


def build_row(args):
    cfg.merge_from_file(args.config)
    output_dir = cfg.OUTPUT_DIR
    log_info = parse_log(os.path.join(output_dir, "log.txt"))
    note = args.note if args.note else UNKNOWN

    return [
        args.experiment_id,
        datetime.now().strftime("%Y-%m-%d"),
        run_command(["git", "rev-parse", "--short", "HEAD"]) or UNKNOWN,
        run_command(["git", "branch", "--show-current"]) or UNKNOWN,
        args.config,
        get_seed(),
        get_gpu(),
        get_dataset_name(cfg.DATASETS.NAMES),
        log_info["runtime"],
        log_info["best_epoch"],
        log_info["rank1"],
        log_info["mAP"],
        note,
    ]


def row_to_markdown(row):
    return "| " + " | ".join(row) + " |"


def find_section_bounds(lines):
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == SECTION_TITLE:
            start = idx
            break
    if start is None:
        return None, None

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    return start, end


def update_experiments(content, row, mode):
    lines = content.splitlines()
    start, end = find_section_bounds(lines)
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([SECTION_TITLE, "", TABLE_HEADER, TABLE_SEPARATOR, row_to_markdown(row)])
        return "\n".join(lines) + "\n"

    section = lines[start:end]
    row_line = row_to_markdown(row)
    experiment_id = row[0]
    existing_idx = None
    for idx, line in enumerate(section):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == experiment_id:
            existing_idx = idx
            break

    if existing_idx is not None:
        if mode == "append":
            raise ValueError("experiment id {} already exists; use --mode update".format(experiment_id))
        section[existing_idx] = row_line
    else:
        section.append(row_line)

    return "\n".join(lines[:start] + section + lines[end:]) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Append or update EXPERIMENTS.md from training logs.")
    parser.add_argument("--config", required=True, help="config file used by the experiment")
    parser.add_argument("--experiment-id", required=True, help="experiment id, e.g. NWL001")
    parser.add_argument("--note", default="", help="experiment note")
    parser.add_argument("--mode", choices=["append", "update"], default="update")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiments-file", default="EXPERIMENTS.md")
    args = parser.parse_args()

    row = build_row(args)
    if os.path.exists(args.experiments_file):
        with open(args.experiments_file, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "# Experiments\n"

    new_content = update_experiments(content, row, args.mode)
    print(row_to_markdown(row))

    if args.dry_run:
        print("\n--dry-run enabled; EXPERIMENTS.md was not modified.")
        return

    with open(args.experiments_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)
    print("Updated {}".format(args.experiments_file))


if __name__ == "__main__":
    main()
