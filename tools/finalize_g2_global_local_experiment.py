#!/usr/bin/env python
"""Select the formal G2 checkpoint and export metrics and gate evidence."""

from __future__ import absolute_import

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from tools.analyze_g2_global_local_gating import analyze
from utils.dynamic_gating_evidence import read_gating_epoch_records
from utils.experiment_recording import (
    atomic_write_json,
    build_dynamic_checkpoint_manifest,
    read_validation_history,
    select_dynamic_checkpoint,
    sha256_file,
)


EXPECTED_BRANCH = "codex/g2-global-local-gating"
EXPECTED_EPOCHS = (40, 80, 120)
EXPECTED_GATING_INPUT = "concat_global_local"


def _git(*arguments):
    output = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT)] + list(arguments),
        stderr=subprocess.STDOUT,
    )
    return output.decode("utf-8", errors="replace").strip()


def _load_configuration(config_path, output_dir):
    configuration = cfg.clone()
    configuration.merge_from_file(str(config_path))
    configuration.freeze()
    required = {
        "SEED": (int(configuration.SEED), 42),
        "MODEL.MULTI_GRANULARITY_GATING_INPUT": (
            str(configuration.MODEL.MULTI_GRANULARITY_GATING_INPUT),
            EXPECTED_GATING_INPUT,
        ),
        "SOLVER.MAX_EPOCHS": (int(configuration.SOLVER.MAX_EPOCHS), 120),
        "SOLVER.CHECKPOINT_PERIOD": (
            int(configuration.SOLVER.CHECKPOINT_PERIOD), 40
        ),
        "SOLVER.EVAL_PERIOD": (int(configuration.SOLVER.EVAL_PERIOD), 40),
    }
    for field, (actual, expected) in required.items():
        if actual != expected:
            raise ValueError(
                "Formal G2 protocol mismatch {}: {!r} != {!r}".format(
                    field, actual, expected
                )
            )
    if Path(str(configuration.OUTPUT_DIR)).resolve() != output_dir:
        raise ValueError("Formal G2 output directory does not match the fixed YAML")
    return configuration


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finalize(config_path, output_dir):
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("G2 output directory is absent: {}".format(output_dir))
    configuration = _load_configuration(config_path, output_dir)

    branch = _git("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise ValueError(
            "G2 finalization requires branch {}, got {}".format(
                EXPECTED_BRANCH, branch
            )
        )
    commit = _git("rev-parse", "HEAD")

    validation_path = output_dir / "validation_history.jsonl"
    validation_records = read_validation_history(validation_path)
    observed_epochs = tuple(int(row["epoch"]) for row in validation_records)
    if observed_epochs != EXPECTED_EPOCHS:
        raise ValueError(
            "Formal G2 requires validation epochs {}, got {}".format(
                EXPECTED_EPOCHS, observed_epochs
            )
        )
    checkpoint_rows = build_dynamic_checkpoint_manifest(
        output_dir, validation_records
    )
    selected_checkpoint, selected_validation = select_dynamic_checkpoint(
        checkpoint_rows, validation_records
    )
    checkpoint_path = output_dir / selected_checkpoint["relative_path"]

    epoch_stats_path = output_dir / "dynamic_gating_epoch_stats.jsonl"
    epoch_records = read_gating_epoch_records(epoch_stats_path)
    if tuple(int(row["epoch"]) for row in epoch_records) != tuple(range(1, 121)):
        raise ValueError("Formal G2 requires one gate-statistics record per epoch")
    selected_epoch = int(selected_validation["epoch"])
    selected_gate_rows = [
        row for row in epoch_records if int(row["epoch"]) == selected_epoch
    ]
    if len(selected_gate_rows) != 1:
        raise ValueError("Selected G2 epoch has no unique gate-statistics record")

    analysis_dir = output_dir / "g2_gating_analysis"
    if analysis_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite existing G2 analysis: {}".format(analysis_dir)
        )
    analysis_manifest = analyze(
        config_path,
        checkpoint_path,
        analysis_dir,
        epoch_stats_path,
        sample_limit=256,
    )
    test_weight_rows = _read_csv(
        analysis_dir / "g2_gate_test_weight_summary.csv"
    )
    controller_block_rows = _read_csv(
        analysis_dir / "g2_controller_input_block_norms.csv"
    )

    result = {
        "experiment": "G2 global-plus-local Dynamic Gating",
        "branch": branch,
        "commit": commit,
        "seed": 42,
        "gating_input": "concat([g, z2, z4, z6])",
        "gate_outputs": ["w2", "w4", "w6"],
        "checkpoint_selection_rule": (
            "highest Rank-1; if tied, highest mAP; if still tied, earliest epoch"
        ),
        "selected_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": selected_epoch,
            "global_iteration": int(selected_validation["global_iteration"]),
        },
        "metrics": {
            "rank1_percent": float(selected_validation["rank1_percent"]),
            "rank5_percent": float(selected_validation["rank5_percent"]),
            "rank10_percent": float(selected_validation["rank10_percent"]),
            "map_percent": float(selected_validation["map_percent"]),
        },
        "selected_epoch_gate_statistics": selected_gate_rows[0],
        "test_gate_weight_distribution": test_weight_rows,
        "controller_input_block_coefficient_statistics": controller_block_rows,
        "evidence": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "validation_history": str(validation_path),
            "validation_history_sha256": sha256_file(validation_path),
            "epoch_gate_statistics": str(epoch_stats_path),
            "epoch_gate_statistics_sha256": sha256_file(epoch_stats_path),
            "analysis_manifest": str(analysis_manifest),
            "analysis_manifest_sha256": sha256_file(analysis_manifest),
        },
    }
    result_path = output_dir / "g2_formal_result.json"
    atomic_write_json(result_path, result)
    return result_path, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result_path, result = finalize(args.config_file, args.output_dir)
    print(json.dumps({
        "result_path": str(result_path),
        "metrics": result["metrics"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
