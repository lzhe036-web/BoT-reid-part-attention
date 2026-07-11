#!/usr/bin/env python3
"""Update one Camera Bias D1/D2 result in EXPERIMENTS.md."""

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


SECTION = "## Camera Bias Debias Validation Experiments"
HEADER = "| 实验编号 | 日期 | commit id | 分支 | 训练集 | 测试集 | checkpoint | camera debias | config 文件 | Rank-1 | mAP | 备注 |"
SEPARATOR = "|---|---|---|---|---|---|---|---|---|---|---|---|"
EXPERIMENT_IDS = (
    "D1-Market-Debias-Off",
    "D1-Market-Debias-On",
    "D2-Duke2Market-Debias-Off",
    "D2-Duke2Market-Debias-On",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True, choices=EXPERIMENT_IDS)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-set", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--camera-debias", required=True, choices=("True", "False"))
    parser.add_argument("--note", required=True)
    parser.add_argument("--mode", required=True, choices=("update",))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_output(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown"


def output_dir_from_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    output_dir = config.get("OUTPUT_DIR")
    if not output_dir:
        raise ValueError("OUTPUT_DIR is missing from {}".format(config_path))
    return Path(output_dir)


def parse_metrics(log_path):
    if not log_path.is_file():
        return "待填写", "待填写"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rank1_matches = re.findall(r"CMC curve,\s*Rank-1\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", text, re.IGNORECASE)
    map_matches = re.findall(r"\bmAP\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", text, re.IGNORECASE)
    rank1 = rank1_matches[-1] + "%" if rank1_matches else "待填写"
    mean_ap = map_matches[-1] + "%" if map_matches else "待填写"
    return rank1, mean_ap


def escape_cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def template_row(experiment_id):
    is_d1 = experiment_id.startswith("D1-")
    enabled = experiment_id.endswith("-On")
    train_set = "Market1501" if is_d1 else "DukeMTMC-reID"
    config = (
        "configs/test_{}_debias_{}_autodl.yml".format(
            "market1501" if is_d1 else "duke2market", "on" if enabled else "off"
        )
    )
    note = (
        "D1 Market1501 same-domain; joint query-gallery camera mean debias protocol"
        if is_d1 else
        "D2 DukeMTMC-reID to Market1501 cross-domain; joint query-gallery camera mean debias protocol"
    )
    values = [experiment_id, "待填写", "待填写", "待填写", train_set, "Market1501", "待填写", str(enabled), config, "待填写", "待填写", note]
    return "| " + " | ".join(escape_cell(value) for value in values) + " |"


def ensure_section(content):
    if SECTION in content:
        return content
    suffix = "" if not content or content.endswith("\n") else "\n"
    block = "\n" + SECTION + "\n\n" + HEADER + "\n" + SEPARATOR + "\n"
    block += "\n".join(template_row(experiment_id) for experiment_id in EXPERIMENT_IDS) + "\n"
    return content + suffix + block


def update_row(content, experiment_id, row):
    content = ensure_section(content)
    lines = content.splitlines()
    start = lines.index(SECTION)
    end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")), len(lines))
    for index in range(start, end):
        if lines[index].startswith("| {} |".format(experiment_id)):
            lines[index] = row
            break
    else:
        lines.insert(end, row)
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output_dir = output_dir_from_config(args.config)
    rank1, mean_ap = parse_metrics(output_dir / "log.txt")
    values = [
        args.experiment_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        command_output(["git", "rev-parse", "--short", "HEAD"]),
        command_output(["git", "branch", "--show-current"]),
        args.train_set,
        args.test_set,
        args.checkpoint,
        args.camera_debias,
        args.config,
        rank1,
        mean_ap,
        args.note,
    ]
    row = "| " + " | ".join(escape_cell(value) for value in values) + " |"
    if args.dry_run:
        print("OUTPUT_DIR: {}".format(output_dir))
        print("log: {}".format(output_dir / "log.txt"))
        print(row)
        return
    experiments_path = Path(__file__).resolve().parents[1] / "EXPERIMENTS.md"
    content = experiments_path.read_text(encoding="utf-8", errors="replace") if experiments_path.exists() else "# Experiments\n"
    experiments_path.write_text(update_row(content, args.experiment_id, row), encoding="utf-8")
    print("Updated {} in {}".format(args.experiment_id, experiments_path))


if __name__ == "__main__":
    main()
