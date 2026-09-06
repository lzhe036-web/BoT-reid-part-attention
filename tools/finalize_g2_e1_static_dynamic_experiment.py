#!/usr/bin/env python
"""Select the formal E1 checkpoint and export bound residual-gating evidence."""

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
from tools.analyze_g2_e1_static_dynamic_alpha import analyze
from utils.dynamic_gating_evidence import read_gating_epoch_records
from utils.experiment_recording import (
    atomic_write_json, build_dynamic_checkpoint_manifest, read_validation_history,
    select_dynamic_checkpoint, sha256_file,
)


EXPECTED_BRANCH = "codex/g2-e1-static-dynamic-alpha0p5"
ORIGINAL_G2_BASE_COMMIT = "5a9a2a33e19f3d9dfc3afaf356dc8b9ee3f0f737"
EXPECTED_EPOCHS = (40, 80, 120)
ALPHA = 0.5
RESULT_FILENAME = "g2_e1_static_dynamic_alpha0p5_formal_result.json"


def _git(*arguments):
    output = subprocess.check_output(["git", "-C", str(REPO_ROOT)] + list(arguments),
                                     stderr=subprocess.STDOUT)
    return output.decode("utf-8", errors="replace").strip()


def _load_configuration(config_path, output_dir):
    configuration = cfg.clone()
    configuration.merge_from_file(str(config_path))
    configuration.freeze()
    checks = {
        "SEED": (int(configuration.SEED), 42),
        "MODEL.MULTI_GRANULARITY_GATING_INPUT": (
            str(configuration.MODEL.MULTI_GRANULARITY_GATING_INPUT), "concat_global_local"),
        "MODEL.MULTI_GRANULARITY_GATING_TAU": (
            float(configuration.MODEL.MULTI_GRANULARITY_GATING_TAU), 1.0),
        "MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL": (
            configuration.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL, True),
        "MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA": (
            float(configuration.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA), ALPHA),
        "SOLVER.MAX_EPOCHS": (int(configuration.SOLVER.MAX_EPOCHS), 120),
        "SOLVER.CHECKPOINT_PERIOD": (int(configuration.SOLVER.CHECKPOINT_PERIOD), 40),
        "SOLVER.EVAL_PERIOD": (int(configuration.SOLVER.EVAL_PERIOD), 40),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError("Formal E1 protocol mismatch {}: {!r} != {!r}".format(
                name, actual, expected))
    if Path(str(configuration.OUTPUT_DIR)).resolve() != output_dir:
        raise ValueError("Formal E1 output directory does not match fixed YAML")
    return configuration


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finalize(config_path, output_dir):
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("E1 output directory is absent: {}".format(output_dir))
    _load_configuration(config_path, output_dir)
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise ValueError("E1 finalization requires branch {}".format(EXPECTED_BRANCH))
    commit = _git("rev-parse", "HEAD")
    subprocess.check_call(["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor",
                           ORIGINAL_G2_BASE_COMMIT, commit])

    validation_path = output_dir / "validation_history.jsonl"
    validation = read_validation_history(validation_path)
    if tuple(int(row["epoch"]) for row in validation) != EXPECTED_EPOCHS:
        raise ValueError("Formal E1 requires validation epochs {}".format(EXPECTED_EPOCHS))
    checkpoint_rows = build_dynamic_checkpoint_manifest(output_dir, validation)
    selected_checkpoint, selected_validation = select_dynamic_checkpoint(checkpoint_rows, validation)
    checkpoint = output_dir / selected_checkpoint["relative_path"]

    epoch_stats_path = output_dir / "dynamic_gating_epoch_stats.jsonl"
    epoch_rows = read_gating_epoch_records(epoch_stats_path)
    if tuple(int(row["epoch"]) for row in epoch_rows) != tuple(range(1, 121)):
        raise ValueError("Formal E1 requires one gate-statistics record per epoch")
    selected_epoch = int(selected_validation["epoch"])
    selected_gate = [row for row in epoch_rows if int(row["epoch"]) == selected_epoch]
    if len(selected_gate) != 1:
        raise ValueError("Selected E1 epoch has no unique gate statistics")

    analysis_dir = output_dir / "g2_e1_static_dynamic_alpha0p5_analysis"
    if analysis_dir.exists():
        raise FileExistsError("Refusing to overwrite E1 analysis: {}".format(analysis_dir))
    analysis_manifest = analyze(config_path, checkpoint, analysis_dir, epoch_stats_path,
                                sample_limit=256)
    result = {
        "experiment": "G2 E1 static concatenation plus gated residual (alpha=0.5)",
        "branch": EXPECTED_BRANCH, "commit": commit, "seed": 42,
        "gating_input": "concat([g, z2, z4, z6])", "gating_temperature": 1.0,
        "gate_outputs": ["w2", "w4", "w6"],
        "fusion_mode": "static_concat_plus_gated_residual", "static_dynamic_alpha": ALPHA,
        "fusion_formula": "concat(g,(1+0.5*w2)z2,(1+0.5*w4)z4,(1+0.5*w6)z6)",
        "checkpoint_selection_rule": "highest Rank-1; if tied, highest mAP; if still tied, earliest epoch",
        "selected_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint),
                                "epoch": selected_epoch,
                                "global_iteration": int(selected_validation["global_iteration"])},
        "metrics": {key: float(selected_validation[key]) for key in
                    ("rank1_percent", "rank5_percent", "rank10_percent", "map_percent")},
        "selected_epoch_gate_statistics": selected_gate[0],
        "evidence": {
            "config": str(config_path), "config_sha256": sha256_file(config_path),
            "validation_history": str(validation_path), "validation_history_sha256": sha256_file(validation_path),
            "epoch_gate_statistics": str(epoch_stats_path), "epoch_gate_statistics_sha256": sha256_file(epoch_stats_path),
            "analysis_manifest": str(analysis_manifest), "analysis_manifest_sha256": sha256_file(analysis_manifest),
        },
    }
    result_path = output_dir / RESULT_FILENAME
    atomic_write_json(result_path, result)
    return result_path, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result_path, result = finalize(args.config_file, args.output_dir)
    print(json.dumps({"result_path": str(result_path), "metrics": result["metrics"]},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
