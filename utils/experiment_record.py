# encoding: utf-8

import csv
import os
import subprocess

import torch


def _run_git_command(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT).decode('utf-8').strip()
    except Exception:
        return "unknown"


def _format_duration(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return "{}h {}m {}s".format(hours, minutes, secs)
    if minutes > 0:
        return "{}m {}s".format(minutes, secs)
    return "{}s".format(secs)


def _get_seed(cfg):
    try:
        return cfg.SOLVER.SEED
    except AttributeError:
        return None


def _get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def _get_notes(cfg):
    if cfg.MODEL.ADAPTIVE_HARD_TRIPLET:
        return "Adaptive Hard Triplet Loss, tau={}, AutoDL".format(cfg.MODEL.ADAPTIVE_HARD_TRIPLET_TAU)
    return ""


def write_experiment_record(cfg, config_file, start_time, end_time, train_result=None):
    if train_result is None:
        train_result = {}

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    record_path = os.path.join(output_dir, "experiment_record.csv")
    fieldnames = [
        "commit_id",
        "dirty",
        "config",
        "seed",
        "gpu",
        "start_time",
        "end_time",
        "duration",
        "best_epoch",
        "rank1",
        "mAP",
        "notes",
    ]

    commit_id = _run_git_command(["git", "rev-parse", "HEAD"])
    dirty = bool(_run_git_command(["git", "status", "--short"]))
    duration = (end_time - start_time).total_seconds()

    row = {
        "commit_id": commit_id,
        "dirty": dirty,
        "config": config_file,
        "seed": _get_seed(cfg),
        "gpu": _get_gpu_name(),
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": _format_duration(duration),
        "best_epoch": train_result.get("best_epoch"),
        "rank1": train_result.get("rank1"),
        "mAP": train_result.get("mAP"),
        "notes": _get_notes(cfg),
    }

    write_header = not os.path.exists(record_path)
    with open(record_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return record_path
