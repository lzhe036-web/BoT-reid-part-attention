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
from tools.g2_dynamic_gating_profiles import G2_GLOBAL_LOCAL_PROFILE
from utils.dynamic_gating_evidence import read_gating_epoch_records
from utils.experiment_recording import (
    atomic_write_json,
    build_dynamic_checkpoint_manifest,
    read_validation_history,
    select_dynamic_checkpoint,
    sha256_file,
)


EXPECTED_BRANCH = G2_GLOBAL_LOCAL_PROFILE.expected_branch
EXPECTED_EPOCHS = (40, 80, 120)
EXPECTED_GATING_INPUT = G2_GLOBAL_LOCAL_PROFILE.gating_input


def _git(*arguments):
    output = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT)] + list(arguments),
        stderr=subprocess.STDOUT,
    )
    return output.decode("utf-8", errors="replace").strip()


def _load_configuration(config_path, output_dir,
                        profile=G2_GLOBAL_LOCAL_PROFILE):
    configuration = cfg.clone()
    configuration.merge_from_file(str(config_path))
    configuration.freeze()
    required = {
        "SEED": (int(configuration.SEED), 42),
        "MODEL.MULTI_GRANULARITY_GATING_INPUT": (
            str(configuration.MODEL.MULTI_GRANULARITY_GATING_INPUT),
            profile.gating_input,
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
                "Formal {} protocol mismatch {}: {!r} != {!r}".format(
                    profile.experiment_label,
                    field, actual, expected
                )
            )
    if Path(str(configuration.OUTPUT_DIR)).resolve() != output_dir:
        raise ValueError(
            "Formal {} output directory does not match the fixed YAML".format(
                profile.experiment_label
            )
        )
    return configuration


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finalize(config_path, output_dir, profile=G2_GLOBAL_LOCAL_PROFILE):
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("G2 output directory is absent: {}".format(output_dir))
    configuration = _load_configuration(config_path, output_dir, profile=profile)

    branch = _git("branch", "--show-current")
    if branch != profile.expected_branch:
        raise ValueError(
            "{} finalization requires branch {}, got {}".format(
                profile.experiment_label, profile.expected_branch, branch
            )
        )
    commit = _git("rev-parse", "HEAD")

    validation_path = output_dir / "validation_history.jsonl"
    validation_records = read_validation_history(validation_path)
    observed_epochs = tuple(int(row["epoch"]) for row in validation_records)
    if observed_epochs != EXPECTED_EPOCHS:
        raise ValueError(
            "Formal {} requires validation epochs {}, got {}".format(
                profile.experiment_label, EXPECTED_EPOCHS, observed_epochs
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
        raise ValueError(
            "Formal {} requires one gate-statistics record per epoch".format(
                profile.experiment_label
            )
        )
    if profile.active_scales == (2, 4):
        forbidden = (
            "p6_mean", "p6_std", "p6_min", "p6_max",
            "applied_w6_mean", "applied_w6_std", "dominant_k6_ratio",
        )
        for row in epoch_records:
            if row.get("gating_scales") != [2, 4]:
                raise ValueError("G2-without-z6 gate scales must be [2, 4]")
            if any(field in row for field in forbidden):
                raise ValueError("G2-without-z6 gate evidence must not record z6")
    selected_epoch = int(selected_validation["epoch"])
    selected_gate_rows = [
        row for row in epoch_records if int(row["epoch"]) == selected_epoch
    ]
    if len(selected_gate_rows) != 1:
        raise ValueError(
            "Selected {} epoch has no unique gate-statistics record".format(
                profile.experiment_label
            )
        )

    analysis_dir = output_dir / profile.analysis_directory_name
    if analysis_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite existing {} analysis: {}".format(
                profile.experiment_label, analysis_dir
            )
        )
    analysis_manifest = analyze(
        config_path,
        checkpoint_path,
        analysis_dir,
        epoch_stats_path,
        sample_limit=256, profile=profile,
    )
    test_weight_rows = _read_csv(
        analysis_dir / "{}_gate_test_weight_summary.csv".format(
            profile.artifact_prefix
        )
    )
    controller_block_rows = _read_csv(
        analysis_dir / "{}_controller_input_block_norms.csv".format(
            profile.artifact_prefix
        )
    )

    result = {
        "experiment": profile.experiment_label,
        "branch": branch,
        "commit": commit,
        "seed": 42,
        "gating_input": profile.gating_input_semantics,
        "gate_outputs": ["w{}".format(scale) for scale in profile.active_scales],
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
    result_path = output_dir / profile.formal_result_filename
    atomic_write_json(result_path, result)
    return result_path, result


def main_for_profile(profile, argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Select and seal a formal {} experiment without retraining."
            .format(profile.experiment_label)
        )
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result_path, result = finalize(
        args.config_file, args.output_dir, profile=profile
    )
    print(json.dumps({
        "result_path": str(result_path),
        "metrics": result["metrics"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv=None):
    return main_for_profile(G2_GLOBAL_LOCAL_PROFILE, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
