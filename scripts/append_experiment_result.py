#!/usr/bin/env python
"""Append or update experiment results in EXPERIMENTS.md after training."""

import argparse
import datetime as _dt
import re
import subprocess
from pathlib import Path


UNKNOWN = "待确认"
TODO = "待填写"
UNFIXED = "未固定"

TABLE_HEADER = (
    "| 实验编号 | 日期 | commit id | 分支 | config 文件 | K | tau | seed | GPU | 数据集 | "
    "运行时间 | best epoch | Rank-1 | mAP | 备注 |"
)
TABLE_SEPARATOR = "|---|---|---|---|---|---:|---:|---|---|---|---|---|---|---|---|"
SECTION_TITLE = "## Auto Appended Results"


def run_command(args):
    try:
        completed = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def strip_inline_comment(value):
    quote = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in ("'", '"'):
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "#" and quote is None:
            return value[:index].rstrip()
    return value.strip()


def parse_scalar(value):
    value = strip_inline_comment(value).strip()
    if not value:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        items = []
        for item in value[1:-1].split(","):
            parsed = parse_scalar(item.strip())
            if parsed is not None:
                items.append(parsed)
        return items
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_simple_yaml(path):
    values = {}
    stack = []
    key_pattern = re.compile(r"^(\s*)([^:#]+):(?:\s*(.*))?$")

    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return values

    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = key_pattern.match(line)
        if not match:
            continue

        indent = len(match.group(1).replace("\t", "    "))
        key = match.group(2).strip()
        value = (match.group(3) or "").strip()

        while stack and stack[-1][0] >= indent:
            stack.pop()

        path_parts = [item[1] for item in stack] + [key]
        dotted_key = ".".join(path_parts)

        if value:
            values[dotted_key] = parse_scalar(value)
        else:
            stack.append((indent, key))

    return values


def value_to_text(value, fallback):
    if value is None or value == "":
        return fallback
    if isinstance(value, list):
        return ",".join(str(item) for item in value) if value else fallback
    return str(value)


def find_seed(config_values):
    preferred_keys = ("SEED", "SOLVER.SEED", "MODEL.SEED", "DATALOADER.SEED")
    for key in preferred_keys:
        if key in config_values:
            return value_to_text(config_values[key], UNFIXED)
    for key, value in config_values.items():
        if key.upper().endswith(".SEED") or key.upper() == "SEED":
            return value_to_text(value, UNFIXED)
    return UNFIXED


def get_gpu_name():
    output = run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if not output:
        return UNKNOWN
    names = [line.strip() for line in output.splitlines() if line.strip()]
    return "; ".join(names) if names else UNKNOWN


def parse_timestamps(log_text):
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")
    timestamps = []
    for match in pattern.finditer(log_text):
        try:
            timestamps.append(_dt.datetime.strptime(" ".join(match.groups()), "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
    return timestamps


def format_duration(delta):
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return TODO
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_best_epoch(log_text):
    for line in log_text.splitlines():
        if "best" not in line.lower():
            continue
        match = re.search(r"epoch(?:\s*[:=]|\s+)\s*(\d+)", line, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return TODO


def parse_metrics(log_path):
    if not log_path or not log_path.exists():
        return {
            "log_exists": False,
            "runtime": TODO,
            "best_epoch": TODO,
            "rank1": TODO,
            "map": TODO,
        }

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    timestamps = parse_timestamps(log_text)
    runtime = format_duration(timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else TODO

    rank_matches = re.findall(
        r"\bRank-1\b\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        log_text,
        flags=re.IGNORECASE,
    )
    map_matches = re.findall(
        r"\bmAP\b\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        log_text,
        flags=re.IGNORECASE,
    )

    rank1 = f"{rank_matches[-1]}%" if rank_matches else TODO
    map_value = f"{map_matches[-1]}%" if map_matches else TODO

    return {
        "log_exists": True,
        "runtime": runtime,
        "best_epoch": parse_best_epoch(log_text),
        "rank1": rank1,
        "map": map_value,
    }


def join_log_path_text(output_dir):
    if output_dir == UNKNOWN:
        return ""
    return output_dir.rstrip("/\\") + "/log.txt"


def markdown_escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").strip()


def build_markdown_row(result):
    columns = [
        result["experiment_id"],
        result["date"],
        result["commit_id"],
        result["branch"],
        result["config"],
        result["k"],
        result["tau"],
        result["seed"],
        result["gpu"],
        result["dataset"],
        result["runtime"],
        result["best_epoch"],
        result["rank1"],
        result["map"],
        result["note"],
    ]
    return "| " + " | ".join(markdown_escape(item) for item in columns) + " |"


def append_row(content, row):
    lines = content.splitlines()
    if SECTION_TITLE not in lines:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([SECTION_TITLE, "", TABLE_HEADER, TABLE_SEPARATOR, row])
        return "\n".join(lines) + "\n"

    section_index = lines.index(SECTION_TITLE)
    next_heading_index = len(lines)
    for index in range(section_index + 1, len(lines)):
        if lines[index].startswith("## ") and lines[index] != SECTION_TITLE:
            next_heading_index = index
            break

    table_indices = [
        index
        for index in range(section_index + 1, next_heading_index)
        if lines[index].startswith("|")
    ]
    if not table_indices:
        insert_at = section_index + 1
        lines[insert_at:insert_at] = ["", TABLE_HEADER, TABLE_SEPARATOR, row]
    else:
        insert_at = table_indices[-1] + 1
        lines.insert(insert_at, row)
    return "\n".join(lines) + "\n"


def update_row_or_append(content, experiment_id, row):
    lines = content.splitlines()
    pattern = re.compile(r"^\|\s*" + re.escape(experiment_id) + r"\s*\|")
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = row
            return "\n".join(lines) + "\n", "updated"
    return append_row(content, row), "appended"


def write_experiment_file(path, row, mode):
    experiments_path = Path(path)
    content = experiments_path.read_text(encoding="utf-8") if experiments_path.exists() else ""
    if mode == "update":
        return update_row_or_append(content, row.split("|", 2)[1].strip(), row)
    return append_row(content, row), "appended"


def collect_result(args):
    config_path = Path(args.config)
    config_values = parse_simple_yaml(config_path)
    output_dir = value_to_text(config_values.get("OUTPUT_DIR"), UNKNOWN)
    log_path_text = join_log_path_text(output_dir)
    log_path = None if not log_path_text else Path(log_path_text)
    metrics = parse_metrics(log_path)

    k = value_to_text(config_values.get("MODEL.PART_ATTENTION_PARTS"), UNKNOWN)
    tau = value_to_text(config_values.get("MODEL.PART_ATTENTION_TAU"), UNKNOWN)
    dataset = value_to_text(config_values.get("DATASETS.NAMES"), "Market1501")
    note = args.note or f"BoT + Part Attention, K={k}, tau={tau}, AutoDL"

    return {
        "experiment_id": args.experiment_id,
        "date": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit_id": run_command(["git", "rev-parse", "--short", "HEAD"]) or UNKNOWN,
        "branch": run_command(["git", "branch", "--show-current"]) or UNKNOWN,
        "config": args.config,
        "k": k,
        "tau": tau,
        "seed": find_seed(config_values),
        "gpu": get_gpu_name(),
        "dataset": dataset,
        "output_dir": output_dir,
        "log_path": log_path_text,
        "log_exists": metrics["log_exists"],
        "runtime": metrics["runtime"],
        "best_epoch": metrics["best_epoch"],
        "rank1": metrics["rank1"],
        "map": metrics["map"],
        "note": note,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append or update one experiment result row in EXPERIMENTS.md."
    )
    parser.add_argument("--config", required=True, help="Path to the training config file.")
    parser.add_argument("--experiment-id", required=True, help="Experiment id, for example T001.")
    parser.add_argument("--note", default="", help="Optional note for the experiment row.")
    parser.add_argument(
        "--experiments-file",
        default="EXPERIMENTS.md",
        help="Markdown file to update. Defaults to EXPERIMENTS.md.",
    )
    parser.add_argument(
        "--mode",
        choices=("append", "update"),
        default="append",
        help="append adds a row; update replaces a matching experiment id row or appends if missing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the row without writing files.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = collect_result(args)
    row = build_markdown_row(result)

    print(f"Config: {result['config']}")
    print(f"OUTPUT_DIR: {result['output_dir']}")
    print(f"log.txt: {result['log_path'] or TODO}")
    print(f"log.txt exists: {'yes' if result['log_exists'] else 'no'}")
    print(f"Rank-1: {result['rank1']}")
    print(f"mAP: {result['map']}")
    print("Markdown row to write:")
    print(row)

    if args.dry_run:
        print("Dry-run: no file was modified.")
        return

    new_content, action = write_experiment_file(args.experiments_file, row, args.mode)
    Path(args.experiments_file).write_text(new_content, encoding="utf-8")
    print(f"{action.capitalize()} {args.experiments_file}.")


if __name__ == "__main__":
    main()
