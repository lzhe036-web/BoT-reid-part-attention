#!/usr/bin/env python
# encoding: utf-8

import argparse
import glob
import json
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
CROSS_CAMERA_POSITIVE_SECTION_TITLE = "## Cross-Camera Positive Only Experiments"
SAME_CAMERA_POSITIVE_SECTION_TITLE = "## Same-Camera Positive Only Ablation"
LAMBDA_SECTION_TITLE = "## C2 Lambda Sensitivity Experiments"
BASELINE_SECTION_TITLE = "## C2 Baseline-Control Experiments"
DUKE_VALIDATION_SECTION_TITLE = "## Duke Validation Experiments"
HEADER = "| 实验编号 | 日期 | commit id | 分支 | 实验类型 | 数据集 | config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | 运行时间 | best epoch | Rank-1 | Rank-5 | Rank-10 | mAP | re-ranking | 备注 |"
SEPARATOR = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
LEGACY_COLUMN_COUNT = 17
PRE_RERANK_COLUMN_COUNT = 19
CURRENT_COLUMN_COUNT = 20


def get_cfg_value(node, key, default=PENDING):
    if node is None:
        return default
    try:
        value = getattr(node, key)
    except (AttributeError, KeyError):
        return default
    return default if value is None else value


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
    model_cfg = get_cfg_value(local_cfg, "MODEL", None)
    return {
        "dataset": normalize_dataset_name(local_cfg.DATASETS.NAMES),
        "output_dir": local_cfg.OUTPUT_DIR,
        # The current code default must never be used to backfill a historical
        # run.  build_row obtains the actual training seed from artifacts in
        # that run's OUTPUT_DIR instead.
        "seed": PENDING,
        "cross_camera_enabled": get_cfg_value(model_cfg, "CROSS_CAMERA_POSITIVE_ONLY", False),
        "cross_camera_lambda": get_cfg_value(model_cfg, "CROSS_CAMERA_POSITIVE_LAMBDA"),
        "same_camera_enabled": get_cfg_value(model_cfg, "SAME_CAMERA_POSITIVE_ONLY", False),
        "same_camera_lambda": get_cfg_value(model_cfg, "SAME_CAMERA_POSITIVE_LAMBDA"),
        "same_camera_mode": get_cfg_value(model_cfg, "SAME_CAMERA_POSITIVE_MODE"),
        "reranking": str(local_cfg.TEST.RE_RANKING),
    }


def parse_metrics(output_dir):
    result = {
        "runtime": PENDING,
        "best_epoch": PENDING,
        "rank1": PENDING,
        "rank5": PENDING,
        "rank10": PENDING,
        "map": PENDING,
        "log_path": PENDING,
        "seed": PENDING,
    }
    if not output_dir or not os.path.isdir(output_dir):
        return result

    metadata_path = os.path.join(output_dir, "reproducibility.json")
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata_seed = json.load(handle).get("seed")
            if (
                    isinstance(metadata_seed, int)
                    and not isinstance(metadata_seed, bool)
                    and 0 <= metadata_seed < 2 ** 32):
                result["seed"] = str(metadata_seed)
        except (OSError, TypeError, ValueError):
            pass

    log_paths = sorted(
        glob.glob(os.path.join(output_dir, "*.txt")) + glob.glob(os.path.join(output_dir, "*.log")),
        key=os.path.getmtime,
        reverse=True,
    )
    candidates = []
    runtimes_by_path = {}
    seeds_by_path = {}
    for path in log_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()
        except OSError:
            continue

        if result["log_path"] == PENDING:
            result["log_path"] = path

        current = None
        recorded_seeds = set()
        first_timestamp = None
        last_timestamp = None
        for line in lines:
            seed_match = re.search(
                r"Reproducibility fixed explicitly:\s*training_seed=(\d+)",
                line,
            )
            if seed_match:
                recorded_seeds.add(seed_match.group(1))
            resolved_seed_match = re.match(r"^SEED:\s*(\d+)\s*$", line.strip())
            if resolved_seed_match:
                recorded_seeds.add(resolved_seed_match.group(1))

            timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if timestamp_match:
                try:
                    timestamp = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                    first_timestamp = timestamp if first_timestamp is None else min(first_timestamp, timestamp)
                    last_timestamp = timestamp if last_timestamp is None else max(last_timestamp, timestamp)
                except ValueError:
                    pass
            epoch_match = re.search(r"Validation Results - Epoch:\s*(\d+)", line)
            if epoch_match:
                if current is not None:
                    candidates.append(current)
                current = {
                    "epoch": epoch_match.group(1),
                    "map": PENDING,
                    "rank1": PENDING,
                    "rank5": PENDING,
                    "rank10": PENDING,
                    "log_path": path,
                }
                continue

            map_match = re.search(r"mAP:\s*([0-9.]+%)", line)
            if map_match and current is not None:
                current["map"] = map_match.group(1)
                continue

            rank_match = re.search(r"Rank-(1|5|10)\s*:\s*([0-9.]+%)", line)
            if rank_match and current is not None:
                current["rank{}".format(rank_match.group(1))] = rank_match.group(2)
        if current is not None:
            candidates.append(current)
        if first_timestamp is not None and last_timestamp is not None:
            runtimes_by_path[path] = str(last_timestamp - first_timestamp)
        if recorded_seeds:
            seeds_by_path[path] = recorded_seeds

    valid_candidates = [
        candidate for candidate in candidates if candidate["rank1"] != PENDING
    ]
    if valid_candidates:
        best = max(
            valid_candidates,
            key=lambda candidate: (
                float(candidate["rank1"].rstrip("%")),
                float(candidate["map"].rstrip("%"))
                if candidate["map"] != PENDING
                else -1.0,
            ),
        )
        result["best_epoch"] = best["epoch"]
        result["rank1"] = best["rank1"]
        result["rank5"] = best["rank5"]
        result["rank10"] = best["rank10"]
        result["map"] = best["map"]
        result["log_path"] = best["log_path"]
        result["runtime"] = runtimes_by_path.get(best["log_path"], PENDING)
    if result["seed"] == PENDING:
        selected_seeds = seeds_by_path.get(result["log_path"], set())
        if len(selected_seeds) == 1:
            result["seed"] = next(iter(selected_seeds))
    return result


def experiment_type_and_lambda(config_info):
    if config_info["same_camera_enabled"]:
        return "Same-camera positive only", str(config_info["same_camera_lambda"])
    if config_info["cross_camera_enabled"]:
        return "Cross-camera positive only", str(config_info["cross_camera_lambda"])
    return "Baseline control", "0 (disabled)"


def build_row(args):
    config_info = collect_config(args.config)
    metrics = parse_metrics(config_info["output_dir"])
    experiment_type, loss_lambda = experiment_type_and_lambda(config_info)
    values = [
        args.experiment_id,
        datetime.now().strftime("%Y-%m-%d"),
        run_command(["git", "rev-parse", "--short", "HEAD"]),
        run_command(["git", "branch", "--show-current"]),
        experiment_type,
        config_info["dataset"],
        args.config,
        config_info["output_dir"],
        metrics["log_path"],
        get_gpu_name(),
        metrics["seed"],
        loss_lambda,
        metrics["runtime"],
        metrics["best_epoch"],
        metrics["rank1"],
        metrics["rank5"],
        metrics["rank10"],
        metrics["map"],
        config_info["reranking"],
        args.note,
    ]
    return "| " + " | ".join(values) + " |"


def select_section_title(args):
    config_name = os.path.basename(args.config)
    if args.experiment_id.startswith("Duke-"):
        return DUKE_VALIDATION_SECTION_TITLE
    if args.experiment_id.startswith("S2-") or "same_camera_positive" in config_name:
        return SAME_CAMERA_POSITIVE_SECTION_TITLE
    if args.experiment_id.startswith("C2-L") or "positive_lambda" in config_name:
        return LAMBDA_SECTION_TITLE
    if args.experiment_id == "C2-Baseline-Control":
        return BASELINE_SECTION_TITLE
    if args.experiment_id.startswith("C2-") or "cross_camera_positive_only" in config_name:
        return CROSS_CAMERA_POSITIVE_SECTION_TITLE
    return SECTION_TITLE


def ensure_section(content, section_title):
    if section_title not in content:
        section = "\n\n{}\n\n{}\n{}\n".format(section_title, HEADER, SEPARATOR)
        return content.rstrip() + section

    lines = content.splitlines()
    section_seen = False
    for index, line in enumerate(lines):
        if line.strip() == section_title:
            section_seen = True
            continue
        if section_seen and line.startswith("## "):
            break
        if not section_seen or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == "实验编号":
            lines[index] = HEADER
            continue
        if cells and all(cell and set(cell) == {"-"} for cell in cells):
            lines[index] = SEPARATOR
            continue
        if len(cells) == LEGACY_COLUMN_COUNT:
            migrated = (
                cells[:15]
                + [PENDING, PENDING]
                + cells[15:16]
                + [PENDING]
                + cells[16:]
            )
            if len(migrated) != CURRENT_COLUMN_COUNT:
                raise RuntimeError("Failed to migrate legacy experiment row.")
            lines[index] = "| " + " | ".join(migrated) + " |"
            continue
        if len(cells) == PRE_RERANK_COLUMN_COUNT:
            migrated = cells[:18] + [PENDING] + cells[18:]
            if len(migrated) != CURRENT_COLUMN_COUNT:
                raise RuntimeError("Failed to add re-ranking to experiment row.")
            lines[index] = "| " + " | ".join(migrated) + " |"
    return "\n".join(lines)


def update_experiments(row, experiment_id, section_title, path="EXPERIMENTS.md"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
    else:
        content = "# Experiments\n"

    content = ensure_section(content, section_title)
    lines = content.splitlines()
    replaced = False
    section_seen = False
    insert_at = len(lines)

    for index, line in enumerate(lines):
        if line.strip() == section_title:
            section_seen = True
            continue
        if section_seen and line.startswith("## ") and line.strip() != section_title:
            insert_at = index
            break
        if section_seen and line.startswith("| {} |".format(experiment_id)):
            lines[index] = row
            replaced = True
            break
        if section_seen and line.startswith("|"):
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

    update_experiments(row, args.experiment_id, select_section_title(args))


if __name__ == "__main__":
    main()
