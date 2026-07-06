# encoding: utf-8
"""
Append test experiment metadata and metrics to EXPERIMENTS.md.
"""

import os
import re
import subprocess
from pathlib import Path


EXPERIMENT_TABLE_HEADER = (
    "| Time | Branch | Commit | Dataset | Config | Seed | GPU | Weight | "
    "Neck feat | Camera debias | Best epoch | Rank-1 | mAP | Runtime | Note |"
)
EXPERIMENT_TABLE_SEPARATOR = (
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def _run_command(command):
    try:
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
        return output.decode("utf-8", errors="replace").strip()
    except Exception:
        return "unknown"


def _git_commit():
    return _run_command(["git", "rev-parse", "--short", "HEAD"])


def _git_branch():
    return _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def _gpu_info():
    gpu = _run_command([
        "nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader",
    ])
    if gpu == "unknown" or not gpu:
        return "unknown"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu = "; ".join(line.strip() for line in gpu.splitlines() if line.strip())
    if visible_devices:
        return "{} (CUDA_VISIBLE_DEVICES={})".format(gpu, visible_devices)
    return gpu


def _cfg_value(cfg, path, default="unknown"):
    current = cfg
    try:
        for part in path.split("."):
            current = getattr(current, part)
        if current is None or current == "":
            return default
        return current
    except Exception:
        return default


def _format_percent(value):
    try:
        return "{:.2f}".format(float(value) * 100.0)
    except Exception:
        return "unknown"


def _format_runtime(seconds):
    try:
        seconds = int(round(float(seconds)))
    except Exception:
        return "unknown"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def _best_epoch_from_weight(weight):
    if not weight:
        return "unknown"
    basename = os.path.basename(str(weight))
    match = re.search(r"(?:epoch|model|checkpoint|ckpt)[_-]?(\d+)", basename, re.IGNORECASE)
    if match:
        return match.group(1)
    numbers = re.findall(r"\d+", basename)
    if numbers:
        return numbers[-1]
    return "unknown"


def _seed(cfg):
    for path in ("SEED", "SOLVER.SEED", "MODEL.SEED"):
        value = _cfg_value(cfg, path)
        if value != "unknown":
            return value
    return "unknown"


def _escape_markdown_cell(value):
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _ensure_experiment_table(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    else:
        content = "# Experiment Records\n\n"

    if EXPERIMENT_TABLE_HEADER in content:
        return content

    if content and not content.endswith("\n"):
        content += "\n"
    if content.strip():
        content += "\n"
    content += "## Auto Test Records\n\n"
    content += EXPERIMENT_TABLE_HEADER + "\n"
    content += EXPERIMENT_TABLE_SEPARATOR + "\n"
    return content


def record_experiment(cfg, config_file, start_time, end_time, cmc=None, mAP=None, note=""):
    project_root = Path(__file__).resolve().parents[1]
    experiments_path = str(project_root / "EXPERIMENTS.md")
    runtime_seconds = (end_time - start_time).total_seconds()
    rank1 = cmc[0] if cmc is not None and len(cmc) > 0 else None
    weight = _cfg_value(cfg, "TEST.WEIGHT")

    row = [
        end_time.strftime("%Y-%m-%d %H:%M:%S"),
        _git_branch(),
        _git_commit(),
        _cfg_value(cfg, "DATASETS.NAMES"),
        config_file or "unknown",
        _seed(cfg),
        _gpu_info(),
        weight,
        _cfg_value(cfg, "TEST.NECK_FEAT"),
        _cfg_value(cfg, "TEST.CAMERA_MEAN_DEBIAS"),
        _best_epoch_from_weight(weight),
        _format_percent(rank1),
        _format_percent(mAP),
        _format_runtime(runtime_seconds),
        note or "unknown",
    ]
    line = "| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |\n"

    content = _ensure_experiment_table(experiments_path)
    with open(experiments_path, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.write(line)
    return experiments_path
