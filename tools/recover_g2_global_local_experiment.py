#!/usr/bin/env python
"""Recover and register a completed formal G2 experiment without retraining.

The G2 training/finalization path predates the unified experiment registry.  A
successful run can therefore contain metrics, checkpoints, and gating plots but
still be absent from ``experiment_records`` and ``EXPERIMENTS.md``.  This tool
validates the existing machine-generated evidence, creates an immutable recovery
bundle, and upserts it through the same registry used by the G1 runner.

It never changes a metric, checkpoint, training log, or gating statistic in the
original output directory.
"""

from __future__ import absolute_import

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dynamic_experiment_registry import (
    DynamicExperimentEvidenceError,
    _refresh_partial_artifact_manifest,
    register_dynamic_run_state,
)
from utils.config_serialization import deserialize_cfg_node_yaml
from utils.dynamic_gating_evidence import read_gating_epoch_records
from utils.experiment_recording import atomic_write_json, read_validation_history, sha256_file
from utils.experiment_schema import (
    GATING_STAT_FIELDS,
    NOT_APPLICABLE,
    NOT_RECORDED,
    SCHEMA_VERSION,
)


EXPERIMENT_ID = "C2-L03-MGDG-G2-GL-T1-S42"
EXPECTED_BRANCH = "codex/g2-global-local-gating"
EXPECTED_PARENT_BRANCH = "exp/c2-l03-multi-granularity-dynamic-gating"
EXPECTED_EPOCHS = (40, 80, 120)
EXPECTED_GATE_EPOCHS = tuple(range(1, 121))
EXPECTED_GATING_INPUT = "concat_global_local"
SELECTION_RULE = "highest Rank-1; if tied, highest mAP; if still tied, earliest epoch"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_autodl.yml"
)


class G2RecoveryError(DynamicExperimentEvidenceError):
    """Raised when completed G2 evidence is absent or internally inconsistent."""


def _read_json(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise G2RecoveryError("Required {} is missing or empty: {}".format(label, path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise G2RecoveryError("Invalid {} {}: {}".format(label, path, error))


def _require_file(path, label):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise G2RecoveryError("Required {} is missing or empty: {}".format(label, path))
    return path


def _file_evidence(path, source_checkpoint_sha256=NOT_APPLICABLE,
                   selection_rule=NOT_APPLICABLE):
    path = _require_file(path, "artifact")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "selection_rule": selection_rule,
    }


def _copy_atomic(source, destination):
    source = _require_file(source, "recovery source")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(".{}.tmp.{}".format(destination.name, uuid.uuid4().hex))
    shutil.copyfile(str(source), str(temporary))
    os.replace(str(temporary), str(destination))
    return destination


def _nested(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise G2RecoveryError("Configuration is missing {}".format(".".join(keys)))
        value = value[key]
    return value


def _validate_configuration(config_path, resolved_path, output_dir, reproducibility):
    config_path = _require_file(config_path, "source config")
    resolved_path = _require_file(resolved_path, "resolved config")
    with config_path.open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle)
    with resolved_path.open("r", encoding="utf-8") as handle:
        resolved = deserialize_cfg_node_yaml(handle.read())
    if not isinstance(source, dict) or not isinstance(resolved, dict):
        raise G2RecoveryError("Source and resolved configs must be YAML mappings")

    checks = (
        (("SEED",), 42),
        (("MODEL", "MULTI_GRANULARITY_DYNAMIC_GATING"), True),
        (("MODEL", "MULTI_GRANULARITY_GATING_INPUT"), EXPECTED_GATING_INPUT),
        (("MODEL", "MULTI_GRANULARITY_GATING_TAU"), 1.0),
        (("MODEL", "MULTI_GRANULARITY_GATING_NORMALIZATION"), "scaled_softmax"),
        (("MODEL", "MULTI_GRANULARITY_PART_SCALES"), [2, 4, 6]),
        (("SOLVER", "MAX_EPOCHS"), 120),
        (("SOLVER", "CHECKPOINT_PERIOD"), 40),
        (("SOLVER", "EVAL_PERIOD"), 40),
    )
    for keys, expected in checks:
        for label, payload in (("source", source), ("resolved", resolved)):
            actual = _nested(payload, *keys)
            if actual != expected:
                raise G2RecoveryError(
                    "G2 {} config mismatch {}: {!r} != {!r}".format(
                        label, ".".join(keys), actual, expected
                    )
                )
    for label, payload in (("source", source), ("resolved", resolved)):
        configured_output = Path(str(_nested(payload, "OUTPUT_DIR"))).resolve()
        if configured_output != output_dir:
            raise G2RecoveryError(
                "G2 {} config OUTPUT_DIR {} does not match {}".format(
                    label, configured_output, output_dir
                )
            )

    configuration_record = reproducibility.get("configuration", {})
    expected_source_sha = configuration_record.get("source_file_sha256")
    expected_resolved_sha = configuration_record.get("resolved_file_sha256")
    source_sha = sha256_file(config_path)
    resolved_sha = sha256_file(resolved_path)
    if expected_source_sha != source_sha:
        raise G2RecoveryError("Source config SHA256 does not match reproducibility.json")
    if expected_resolved_sha != resolved_sha:
        raise G2RecoveryError("Resolved config SHA256 does not match reproducibility.json")
    return source, resolved


def _validate_reproducibility(reproducibility):
    chain = reproducibility.get("seed_chain", {})
    seed_values = {
        "seed": reproducibility.get("seed"),
        "source_config_seed": chain.get("source_config_seed"),
        "resolved_config_seed": chain.get("resolved_config_seed"),
        "applied_training_seed": chain.get("applied_training_seed"),
        "reproducibility_metadata_seed": chain.get("reproducibility_metadata_seed"),
    }
    invalid = {key: value for key, value in seed_values.items() if value != 42}
    if invalid:
        raise G2RecoveryError("Seed=42 evidence chain is incomplete: {}".format(invalid))
    if reproducibility.get("seed_applied_before_data_loading") is not True:
        raise G2RecoveryError("Seed was not recorded before data loading")
    code = reproducibility.get("code", {})
    if code.get("branch") != EXPECTED_BRANCH:
        raise G2RecoveryError(
            "Training branch mismatch: {!r} != {!r}".format(
                code.get("branch"), EXPECTED_BRANCH
            )
        )
    commit = code.get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise G2RecoveryError("Training commit is missing from reproducibility.json")
    if code.get("dirty") is not False:
        raise G2RecoveryError(
            "Formal G2 launch did not record a clean working tree: {!r}".format(
                code.get("dirty")
            )
        )
    return commit


def _read_checkpoint_manifest(path):
    path = _require_file(path, "checkpoint manifest")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "epoch", "global_iteration", "relative_path", "file_size", "sha256", "selected"
    }
    if not rows or not required.issubset(set(rows[0])):
        raise G2RecoveryError("checkpoint_manifest.tsv has an invalid schema")
    return rows


def _best_validation(records):
    return sorted(
        records,
        key=lambda row: (
            float(row["rank1_percent"]),
            float(row["map_percent"]),
            -int(row["epoch"]),
        ),
        reverse=True,
    )[0]


def _same_number(left, right):
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _validate_result(output_dir, result, commit, validation_records,
                     checkpoint_rows, gate_records):
    if result.get("branch") != EXPECTED_BRANCH or result.get("commit") != commit:
        raise G2RecoveryError("G2 result branch/commit does not match training evidence")
    if result.get("seed") != 42:
        raise G2RecoveryError("G2 result does not record Seed=42")
    if result.get("gating_input") != "concat([g, z2, z4, z6])":
        raise G2RecoveryError("G2 result has the wrong controller input")
    if result.get("gate_outputs") != ["w2", "w4", "w6"]:
        raise G2RecoveryError("G2 result has the wrong gate-output semantics")

    observed_validation_epochs = tuple(int(row["epoch"]) for row in validation_records)
    if observed_validation_epochs != EXPECTED_EPOCHS:
        raise G2RecoveryError(
            "Formal G2 validation epochs are {}, expected {}".format(
                observed_validation_epochs, EXPECTED_EPOCHS
            )
        )
    best = _best_validation(validation_records)
    selected = result.get("selected_checkpoint", {})
    metrics = result.get("metrics", {})
    if int(selected.get("epoch", -1)) != int(best["epoch"]):
        raise G2RecoveryError("G2 result selected a checkpoint from the wrong epoch")
    for field in ("rank1_percent", "rank5_percent", "rank10_percent", "map_percent"):
        if not _same_number(metrics.get(field), best[field]):
            raise G2RecoveryError("G2 result metric {} was not machine-selected".format(field))

    selected_rows = [row for row in checkpoint_rows if row.get("selected") == "true"]
    if len(selected_rows) != 1:
        raise G2RecoveryError("Checkpoint manifest must contain exactly one selected row")
    selected_row = selected_rows[0]
    checkpoint_path = output_dir / selected_row["relative_path"]
    checkpoint_path = _require_file(checkpoint_path, "selected checkpoint")
    checkpoint_sha = sha256_file(checkpoint_path)
    if int(selected_row["epoch"]) != int(best["epoch"]):
        raise G2RecoveryError("Selected checkpoint row does not match best validation epoch")
    if selected_row["sha256"] != checkpoint_sha or selected.get("sha256") != checkpoint_sha:
        raise G2RecoveryError("Selected checkpoint SHA256 binding is inconsistent")
    if Path(str(selected.get("path", ""))).resolve() != checkpoint_path:
        raise G2RecoveryError("G2 result selected checkpoint path is inconsistent")
    for row in checkpoint_rows:
        candidate = _require_file(output_dir / row["relative_path"], "checkpoint")
        if sha256_file(candidate) != row["sha256"]:
            raise G2RecoveryError("Checkpoint manifest SHA256 mismatch: {}".format(candidate))
        if int(candidate.stat().st_size) != int(row["file_size"]):
            raise G2RecoveryError("Checkpoint manifest size mismatch: {}".format(candidate))

    observed_gate_epochs = tuple(int(row["epoch"]) for row in gate_records)
    if observed_gate_epochs != EXPECTED_GATE_EPOCHS:
        raise G2RecoveryError("Formal G2 requires one gate-statistics row per epoch")
    selected_gate = [row for row in gate_records if int(row["epoch"]) == int(best["epoch"])]
    if len(selected_gate) != 1:
        raise G2RecoveryError("Selected epoch has no unique gate-statistics row")
    result_gate = result.get("selected_epoch_gate_statistics", {})
    for field in GATING_STAT_FIELDS:
        value = selected_gate[0].get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise G2RecoveryError("Invalid gate statistic {}".format(field))
        if not _same_number(result_gate.get(field), value):
            raise G2RecoveryError("G2 result gate statistic {} was changed".format(field))
    return best, selected_row, selected_gate[0], checkpoint_path, checkpoint_sha


def _validate_analysis(output_dir, result, checkpoint_sha, config_path,
                       validation_path, gate_stats_path):
    evidence = result.get("evidence", {})
    bindings = (
        (config_path, evidence.get("config_sha256"), "result config"),
        (validation_path, evidence.get("validation_history_sha256"), "validation history"),
        (gate_stats_path, evidence.get("epoch_gate_statistics_sha256"), "gate statistics"),
    )
    for path, expected_sha, label in bindings:
        if sha256_file(path) != expected_sha:
            raise G2RecoveryError("{} SHA256 binding is inconsistent".format(label))

    analysis_manifest_path = _require_file(
        evidence.get("analysis_manifest", ""), "G2 analysis manifest"
    )
    if sha256_file(analysis_manifest_path) != evidence.get("analysis_manifest_sha256"):
        raise G2RecoveryError("G2 analysis manifest SHA256 binding is inconsistent")
    analysis = _read_json(analysis_manifest_path, "G2 analysis manifest")
    if analysis.get("checkpoint_sha256") != checkpoint_sha:
        raise G2RecoveryError("G2 analysis is bound to a different checkpoint")
    if analysis.get("config_sha256") != sha256_file(config_path):
        raise G2RecoveryError("G2 analysis is bound to a different config")
    if analysis.get("epoch_statistics_sha256") != sha256_file(gate_stats_path):
        raise G2RecoveryError("G2 analysis is bound to different gate statistics")

    analysis_files = {}
    for artifact_type, item in sorted(analysis.get("files", {}).items()):
        if not isinstance(item, dict):
            raise G2RecoveryError("Invalid analysis artifact {}".format(artifact_type))
        artifact_path = _require_file(item.get("path", ""), "analysis artifact")
        if sha256_file(artifact_path) != item.get("sha256"):
            raise G2RecoveryError("Analysis artifact SHA256 mismatch: {}".format(artifact_path))
        analysis_files[artifact_type] = artifact_path

    required = {
        "controller_block_norms_csv",
        "controller_block_norms_png",
        "test_gate_samples_tsv",
        "test_weight_summary_csv",
        "test_weight_distribution_png",
        "dynamic_gating_summary_json",
    }
    if not required.issubset(set(analysis_files)):
        raise G2RecoveryError(
            "G2 analysis is missing required artifacts: {}".format(
                sorted(required - set(analysis_files))
            )
        )
    summary_path = analysis_files["dynamic_gating_summary_json"]
    samples_path = analysis_files["test_gate_samples_tsv"]
    summary = _read_json(summary_path, "dynamic gating summary")
    if summary.get("source_checkpoint_sha256") != checkpoint_sha:
        raise G2RecoveryError("Dynamic gating summary is bound to a different checkpoint")
    sample_evidence = summary.get("gating_samples", {})
    if sample_evidence.get("sha256") != sha256_file(samples_path):
        raise G2RecoveryError("Gating samples SHA256 binding is inconsistent")
    if int(summary.get("selected_sample_count", 0)) <= 0:
        raise G2RecoveryError("No deterministic G2 gating samples were recorded")
    return analysis_manifest_path, analysis_files, summary_path, samples_path, summary


def _parse_utc(value, label):
    if not isinstance(value, str) or not value:
        raise G2RecoveryError("{} is missing".format(label))
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        raise G2RecoveryError("{} is not ISO-8601: {!r}".format(label, value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _timing(reproducibility, result_path, started_at_utc=None,
            ended_at_utc=None, runtime_seconds=None):
    started = _parse_utc(
        started_at_utc or reproducibility.get("created_at_utc"), "start time"
    )
    if ended_at_utc:
        ended = _parse_utc(ended_at_utc, "end time")
    else:
        ended = dt.datetime.fromtimestamp(
            Path(result_path).stat().st_mtime, tz=dt.timezone.utc
        )
    measured = (ended - started).total_seconds()
    runtime = float(runtime_seconds) if runtime_seconds is not None else measured
    if measured < 0 or runtime < 0 or not math.isfinite(runtime):
        raise G2RecoveryError("Formal G2 runtime evidence is invalid")
    return (
        started.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        ended.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        runtime,
    )


def _lineage(commit):
    """Return only lineage that can be derived from the recorded Git object."""
    import subprocess

    def git(*args):
        try:
            output = subprocess.check_output(
                ["git", "-C", str(REPO_ROOT)] + list(args),
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return NOT_RECORDED
        return output.decode("utf-8", errors="replace").strip() or NOT_RECORDED

    parent_commit = git("merge-base", commit, EXPECTED_PARENT_BRANCH)
    return {
        "parent_branch": EXPECTED_PARENT_BRANCH,
        "parent_commit": parent_commit,
        "merge_base": parent_commit,
    }


def _environment_payload(reproducibility, reproducibility_sha):
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_source": "reproducibility.json recorded before data loading",
        "recorded_at_utc": reproducibility.get("created_at_utc", NOT_RECORDED),
        "source_reproducibility_sha256": reproducibility_sha,
        "code": reproducibility.get("code", {}),
        "environment": reproducibility.get("environment", {}),
    }


def _verify_existing_run(run_dir, output_dir, commit, checkpoint_sha):
    manifest = _read_json(run_dir / "run_manifest.json", "recovered run manifest")
    status = _read_json(run_dir / "run_status.json", "recovered run status")
    expected = {
        "run_id": run_dir.name,
        "status": "success",
        "commit": commit,
        "output_dir": str(output_dir),
        "gating_input": EXPECTED_GATING_INPUT,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise G2RecoveryError(
                "Existing recovery bundle identity mismatch {}: {!r} != {!r}".format(
                    key, manifest.get(key), value
                )
            )
    if status.get("run_id") != run_dir.name or status.get("status") != "success":
        raise G2RecoveryError("Existing recovery run_status.json is inconsistent")
    if manifest.get("selected_checkpoint", {}).get("sha256") != checkpoint_sha:
        raise G2RecoveryError("Existing recovery bundle selected a different checkpoint")
    for artifact_type, evidence in manifest.get("artifacts", {}).items():
        if not isinstance(evidence, dict) or evidence.get("path") in (
                None, "", NOT_RECORDED):
            continue
        path = _require_file(evidence["path"], artifact_type)
        if sha256_file(path) != evidence.get("sha256"):
            raise G2RecoveryError(
                "Existing recovery artifact was changed: {}".format(path)
            )
    return register_dynamic_run_state(run_dir)


def recover(config_path, output_dir, console_log, records_root, experiments_path,
            started_at_utc=None, ended_at_utc=None, runtime_seconds=None):
    config_path = _require_file(config_path, "source config")
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise G2RecoveryError("G2 output directory is absent: {}".format(output_dir))
    console_log = _require_file(console_log, "console log")
    records_root = Path(records_root).resolve()
    experiments_path = Path(experiments_path).resolve()

    training_log = _require_file(output_dir / "log.txt", "training log")
    resolved_path = _require_file(output_dir / "config_resolved.yml", "resolved config")
    reproducibility_path = _require_file(
        output_dir / "reproducibility.json", "reproducibility record"
    )
    validation_path = _require_file(
        output_dir / "validation_history.jsonl", "validation history"
    )
    gate_stats_path = _require_file(
        output_dir / "dynamic_gating_epoch_stats.jsonl", "gate epoch statistics"
    )
    checkpoint_manifest_path = _require_file(
        output_dir / "checkpoint_manifest.tsv", "checkpoint manifest"
    )
    result_path = _require_file(output_dir / "g2_formal_result.json", "G2 formal result")

    reproducibility = _read_json(reproducibility_path, "reproducibility record")
    commit = _validate_reproducibility(reproducibility)
    source_config, _resolved_config = _validate_configuration(
        config_path, resolved_path, output_dir, reproducibility
    )
    validation_records = read_validation_history(validation_path)
    gate_records = read_gating_epoch_records(gate_stats_path)
    checkpoint_rows = _read_checkpoint_manifest(checkpoint_manifest_path)
    result = _read_json(result_path, "G2 formal result")
    best, selected_row, gating_statistics, checkpoint_path, checkpoint_sha = _validate_result(
        output_dir, result, commit, validation_records, checkpoint_rows, gate_records
    )
    analysis_manifest_path, analysis_files, summary_path, samples_path, summary = \
        _validate_analysis(
            output_dir, result, checkpoint_sha, config_path,
            validation_path, gate_stats_path
        )
    started, ended, runtime = _timing(
        reproducibility, result_path, started_at_utc, ended_at_utc, runtime_seconds
    )

    run_id = "{}-{}-{}".format(EXPERIMENT_ID, commit[:10], checkpoint_sha[:10])
    runs_root = records_root / "runs"
    run_dir = runs_root / run_id
    if run_dir.exists():
        row = _verify_existing_run(run_dir, output_dir, commit, checkpoint_sha)
        return run_dir, row, False

    runs_root.mkdir(parents=True, exist_ok=True)
    # Build directly at the deterministic final path so every recorded absolute
    # evidence path remains valid.  The directory is exclusively created here
    # and is removed on failure before it can be registered.
    temporary_dir = run_dir
    temporary_dir.mkdir()
    try:
        source_snapshot = _copy_atomic(config_path, temporary_dir / "config_source.yml")
        resolved_snapshot = _copy_atomic(resolved_path, temporary_dir / "config_resolved.yml")
        console_snapshot = _copy_atomic(console_log, temporary_dir / "console.log")
        checkpoint_manifest_snapshot = _copy_atomic(
            checkpoint_manifest_path, temporary_dir / "checkpoint_manifest.tsv"
        )
        result_snapshot = _copy_atomic(result_path, temporary_dir / "g2_formal_result.json")
        analysis_manifest_snapshot = _copy_atomic(
            analysis_manifest_path, temporary_dir / "g2_gating_analysis_manifest.json"
        )
        summary_snapshot = _copy_atomic(summary_path, temporary_dir / "dynamic_gating_summary.json")
        samples_snapshot = _copy_atomic(samples_path, temporary_dir / "gating_samples.tsv")
        reproducibility_sha = sha256_file(reproducibility_path)
        environment_path = temporary_dir / "environment.json"
        atomic_write_json(
            environment_path,
            _environment_payload(reproducibility, reproducibility_sha),
        )

        lineage = _lineage(commit)
        env = reproducibility.get("environment", {})
        gpu_names = env.get("gpu_names", [])
        gpu = ", ".join(str(item) for item in gpu_names) if gpu_names else NOT_RECORDED
        artifacts = {
            "source_config": _file_evidence(source_snapshot),
            "source_config_origin": _file_evidence(config_path),
            "resolved_config_snapshot": _file_evidence(resolved_snapshot),
            "resolved_config_origin": _file_evidence(resolved_path),
            "console_log": _file_evidence(console_snapshot),
            "training_log": _file_evidence(training_log),
            "reproducibility": _file_evidence(reproducibility_path),
            "environment": _file_evidence(environment_path),
            "validation_history": _file_evidence(validation_path),
            "gating_epoch_statistics": _file_evidence(gate_stats_path),
            "checkpoint_manifest": _file_evidence(checkpoint_manifest_snapshot),
            "selected_checkpoint": _file_evidence(
                checkpoint_path, checkpoint_sha, SELECTION_RULE
            ),
            "g2_formal_result": _file_evidence(result_snapshot),
            "g2_gating_analysis_manifest": _file_evidence(analysis_manifest_snapshot),
            "dynamic_gating_summary": _file_evidence(
                summary_snapshot, checkpoint_sha, summary.get("selection_rule", SELECTION_RULE)
            ),
            "gating_samples": _file_evidence(
                samples_snapshot, checkpoint_sha, summary.get("selection_rule", SELECTION_RULE)
            ),
        }
        for index, row in enumerate(checkpoint_rows):
            checkpoint = output_dir / row["relative_path"]
            artifacts["checkpoint_{:03d}".format(index)] = _file_evidence(
                checkpoint, row["sha256"], NOT_APPLICABLE
            )
        for artifact_type, path in sorted(analysis_files.items()):
            artifacts["g2_analysis_{}".format(artifact_type)] = _file_evidence(
                path, checkpoint_sha, summary.get("selection_rule", SELECTION_RULE)
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "experiment_family": "C2-L03-MULTI-GRANULARITY-DYNAMIC-GATING",
            "evidence_id": run_id,
            "run_id": run_id,
            "run_kind": "formal",
            "status": "success",
            "state_history": [
                {"status": "success", "timestamp_utc": ended, "source": "post-hoc recovery"}
            ],
            "method_family": "multi_granularity_feature",
            "method_variant": "g2_global_local_per_sample_dynamic_gating",
            "method": "C2-L03 + G2 Dynamic Gating [g,z2,z4,z6] -> [w2,w4,w6]",
            "dataset": str(_nested(source_config, "DATASETS", "NAMES")),
            "baseline": "C2-L03 + MGP concat",
            "margin": _nested(source_config, "SOLVER", "MARGIN"),
            "mode": _nested(source_config, "MODEL", "CROSS_CAMERA_POSITIVE_MODE"),
            "lambda": NOT_APPLICABLE,
            "cross_camera_positive_lambda": _nested(
                source_config, "MODEL", "CROSS_CAMERA_POSITIVE_LAMBDA"
            ),
            "branch": EXPECTED_BRANCH,
            "commit": commit,
            "parent_branch": lineage["parent_branch"],
            "parent_commit": lineage["parent_commit"],
            "merge_base": lineage["merge_base"],
            "candidate_protocol_signature_sha256": NOT_RECORDED,
            "implementation_signature_sha256": NOT_RECORDED,
            "feature_reference_commit": NOT_RECORDED,
            "feature_reference_signature_sha256": NOT_RECORDED,
            "current_feature_signature_sha256": NOT_RECORDED,
            "feature_compatibility_status": NOT_RECORDED,
            "gating_signature_sha256": NOT_RECORDED,
            "seed": 42,
            "gpu": gpu,
            "started_at_utc": started,
            "ended_at_utc": ended,
            "runtime_seconds": runtime,
            "return_code": 0,
            "output_dir": str(output_dir),
            "command": reproducibility.get("command", []),
            "alignment_mode": NOT_APPLICABLE,
            "alignment_temperature": NOT_APPLICABLE,
            "gating_mode": "per_sample_dynamic_gating",
            "gating_input": EXPECTED_GATING_INPUT,
            "gating_temperature": 1.0,
            "gating_normalization": "scaled_softmax",
            "scale_order": "2,4,6",
            "gate_outputs": ["w2", "w4", "w6"],
            "metrics": {
                "rank1_percent": float(best["rank1_percent"]),
                "rank5_percent": float(best["rank5_percent"]),
                "rank10_percent": float(best["rank10_percent"]),
                "map_percent": float(best["map_percent"]),
                "best_epoch": int(best["epoch"]),
                "selected_epoch": int(best["epoch"]),
            },
            "gating_statistics": dict(gating_statistics),
            "launch_worktree_clean": True,
            "launch_worktree_clean_evidence": str(reproducibility_path),
            "selected_checkpoint_record": dict(selected_row),
            "notes": (
                "Recovered from existing machine-generated G2 evidence; numeric values "
                "were not edited. Launch commit/worktree state comes from reproducibility.json."
            ),
            "records_root": str(records_root),
            "experiments_path": str(experiments_path),
            "source_config": artifacts["source_config"],
            "source_config_origin": artifacts["source_config_origin"],
            "resolved_config": artifacts["resolved_config_snapshot"],
            "console_log": artifacts["console_log"],
            "training_log": artifacts["training_log"],
            "reproducibility": artifacts["reproducibility"],
            "environment": artifacts["environment"],
            "checkpoint_manifest": artifacts["checkpoint_manifest"],
            "selected_checkpoint": artifacts["selected_checkpoint"],
            "dynamic_gating_summary": artifacts["dynamic_gating_summary"],
            "gating_samples": artifacts["gating_samples"],
            "artifacts": artifacts,
        }
        _refresh_partial_artifact_manifest(temporary_dir, manifest)
        atomic_write_json(temporary_dir / "run_manifest.json", manifest)
        atomic_write_json(temporary_dir / "run_status.json", {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": "success",
            "errors": [],
            "started_at_utc": started,
            "ended_at_utc": ended,
            "return_code": 0,
            "recovery": True,
        })
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(str(temporary_dir))
        raise

    row = register_dynamic_run_state(run_dir)
    return run_dir, row, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--console-log", required=True)
    parser.add_argument(
        "--records-root", default=str(REPO_ROOT / "experiment_records")
    )
    parser.add_argument(
        "--experiments-path", default=str(REPO_ROOT / "EXPERIMENTS.md")
    )
    parser.add_argument("--started-at-utc", default=None)
    parser.add_argument("--ended-at-utc", default=None)
    parser.add_argument("--runtime-seconds", type=float, default=None)
    args = parser.parse_args(argv)
    run_dir, row, created = recover(
        args.config_file,
        args.output_dir,
        args.console_log,
        args.records_root,
        args.experiments_path,
        started_at_utc=args.started_at_utc,
        ended_at_utc=args.ended_at_utc,
        runtime_seconds=args.runtime_seconds,
    )
    print(json.dumps({
        "created": created,
        "run_dir": str(run_dir),
        "run_id": row["run_id"],
        "status": row["status"],
        "rank1_percent": row["rank1_percent"],
        "map_percent": row["map_percent"],
        "selected_epoch": row["selected_epoch"],
        "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
