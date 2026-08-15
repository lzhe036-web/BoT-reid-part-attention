# encoding: utf-8
"""Transactional registry for reproducible Dynamic Gating experiments.

This module is imported and re-exported by :mod:`utils.experiment_recording` so
the project continues to expose one recorder API while the historical C2-MGP
schema remains readable.
"""

from __future__ import absolute_import

import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import yaml

from utils.dynamic_gating_evidence import GATING_STAT_FIELDS, read_gating_epoch_records
from utils.multigranularity_signatures import (
    STATIC_BASELINE_BRANCH,
    STATIC_BASELINE_SHA,
)


UNIFIED_SCHEMA_VERSION = 2
NOT_RECORDED = "not_recorded"
NOT_APPLICABLE = "not_applicable"
MISSING_EVIDENCE = "missing_evidence"
AUTO_RUNS_START = "<!-- AUTO-DYNAMIC-GATING-RUNS:START -->"
AUTO_RUNS_END = "<!-- AUTO-DYNAMIC-GATING-RUNS:END -->"
AUTO_FORMAL_START = "<!-- AUTO-DYNAMIC-GATING-FORMAL:START -->"
AUTO_FORMAL_END = "<!-- AUTO-DYNAMIC-GATING-FORMAL:END -->"
AUTO_CHECKPOINTS_START = "<!-- AUTO-DYNAMIC-GATING-CHECKPOINTS:START -->"
AUTO_CHECKPOINTS_END = "<!-- AUTO-DYNAMIC-GATING-CHECKPOINTS:END -->"


class DynamicExperimentEvidenceError(RuntimeError):
    pass


BASE_RUN_FIELDS = (
    "schema_version", "experiment_id", "experiment_family", "evidence_id",
    "run_id", "run_kind", "status", "method_family", "method_variant",
    "branch", "commit", "parent_branch", "parent_commit", "merge_base",
    "feature_reference_commit", "feature_reference_signature_sha256",
    "current_feature_signature_sha256", "feature_compatibility_status",
    "feature_compatibility_evidence_path",
    "feature_compatibility_evidence_size_bytes",
    "feature_compatibility_evidence_sha256", "gating_signature_sha256",
    "gating_signature_path", "gating_signature_size_bytes",
    "gating_signature_evidence_sha256",
    "seed", "source_config_origin_path", "source_config_origin_size_bytes",
    "source_config_origin_sha256", "source_config_path", "source_config_size_bytes",
    "source_config_sha256", "resolved_config_path",
    "resolved_config_size_bytes", "resolved_config_sha256",
    "training_log_path", "training_log_size_bytes", "training_log_sha256",
    "console_log_path", "console_log_size_bytes", "console_log_sha256",
    "output_dir", "selected_checkpoint_path", "selected_checkpoint_size_bytes",
    "selected_checkpoint_sha256", "checkpoint_manifest_path",
    "checkpoint_manifest_size_bytes", "checkpoint_manifest_sha256",
    "artifact_manifest_path", "artifact_manifest_size_bytes",
    "artifact_manifest_sha256", "run_evidence_manifest_path",
    "run_evidence_manifest_size_bytes", "run_evidence_manifest_sha256",
    "reproducibility_path",
    "reproducibility_size_bytes", "reproducibility_sha256",
    "dataset_manifest_path", "dataset_manifest_sha256", "model_manifest_path",
    "model_manifest_sha256", "environment_path", "environment_sha256", "gpu",
    "start_time", "end_time", "runtime_seconds", "return_code",
    "alignment_mode", "alignment_temperature", "gating_mode",
    "gating_input", "gating_temperature", "gating_normalization", "scale_order",
    "dynamic_gating_summary_path", "dynamic_gating_summary_size_bytes",
    "dynamic_gating_summary_sha256", "dynamic_gating_summary_source_checkpoint_sha256",
    "gating_samples_path", "gating_samples_size_bytes", "gating_samples_sha256",
    "gating_samples_source_checkpoint_sha256", "gating_sample_selection_rule",
)
GATING_RUN_STAT_FIELDS = tuple(
    field for field in GATING_STAT_FIELDS if field not in BASE_RUN_FIELDS
)
RUN_FIELDS = BASE_RUN_FIELDS + GATING_RUN_STAT_FIELDS + (
    "rank1_percent", "rank5_percent", "rank10_percent", "map_percent",
    "best_epoch", "selected_epoch", "notes",
)
EVIDENCE_FIELDS = (
    "schema_version", "run_id", "experiment_id", "run_kind", "status",
    "artifact_type", "path", "size_bytes", "sha256",
    "source_checkpoint_sha256", "selection_rule",
)
CHECKPOINT_FIELDS = (
    "run_id", "experiment_id", "run_kind", "checkpoint_path", "size_bytes",
    "ignite_epoch", "global_iteration", "sha256", "selected",
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}.{}".format(path.name, os.getpid(), uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_write_json(path, payload):
    _atomic_write(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git(repo_root, args):
    try:
        output = subprocess.check_output(
            ["git", "-C", str(Path(repo_root).resolve())] + list(args),
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise DynamicExperimentEvidenceError(
            "Git evidence command failed: {}".format(" ".join(args))
        ) from error
    return output.decode("utf-8", errors="replace").strip()


def validate_dynamic_lineage(repo_root, expected_commit=None):
    branch = _git(repo_root, ["branch", "--show-current"])
    commit = _git(repo_root, ["rev-parse", "HEAD"])
    parent_local = _git(repo_root, ["rev-parse", STATIC_BASELINE_BRANCH])
    parent_remote = _git(
        repo_root, ["rev-parse", "origin/{}".format(STATIC_BASELINE_BRANCH)]
    )
    remote_ref = "refs/heads/{}".format(STATIC_BASELINE_BRANCH)
    remote_rows = _git(
        repo_root, ["ls-remote", "--heads", "origin", STATIC_BASELINE_BRANCH]
    ).splitlines()
    parsed_remote_rows = [
        row.split() for row in remote_rows if row.strip()
    ]
    if len(parsed_remote_rows) != 1 or len(parsed_remote_rows[0]) != 2 \
            or parsed_remote_rows[0][1] != remote_ref:
        raise DynamicExperimentEvidenceError(
            "Unable to resolve the unique Static baseline branch on origin"
        )
    parent_remote_actual = parsed_remote_rows[0][0]
    merge_base = _git(repo_root, ["merge-base", STATIC_BASELINE_BRANCH, "HEAD"])
    status = _git(repo_root, ["status", "--porcelain=v1", "--untracked-files=all"])
    expected_branch = "exp/c2-l03-multi-granularity-dynamic-gating"
    if branch != expected_branch:
        raise DynamicExperimentEvidenceError(
            "Dynamic experiment requires branch {}, got {}".format(expected_branch, branch)
        )
    if expected_commit is not None and commit != expected_commit:
        raise DynamicExperimentEvidenceError("Current commit changed during preflight")
    if (parent_local != STATIC_BASELINE_SHA
            or parent_remote != STATIC_BASELINE_SHA
            or parent_remote_actual != STATIC_BASELINE_SHA):
        raise DynamicExperimentEvidenceError(
            "Static baseline local/remote tracking SHA mismatch"
        )
    if merge_base != STATIC_BASELINE_SHA:
        raise DynamicExperimentEvidenceError(
            "Dynamic branch merge-base is not the fixed Static baseline"
        )
    if status:
        raise DynamicExperimentEvidenceError("Experiment launch requires a clean worktree")
    return {
        "branch": branch,
        "commit": commit,
        "parent_branch": STATIC_BASELINE_BRANCH,
        "parent_commit": STATIC_BASELINE_SHA,
        "parent_local_commit": parent_local,
        "parent_remote_tracking_commit": parent_remote,
        "parent_remote_actual_commit": parent_remote_actual,
        "merge_base": merge_base,
        "dirty": False,
    }


def validate_dynamic_runtime_worktree(repo_root, run_dir, output_dir):
    """Allow only recorder-managed evidence mutations after initialization."""
    repo = Path(repo_root).resolve()
    allowed_files = {
        "EXPERIMENTS.md", "experiment_records/runs.csv",
        "experiment_records/evidence_manifest.tsv",
        "experiment_records/tables/main_results.csv",
    }
    allowed_prefixes = [
        Path(run_dir).resolve(),
        (repo / "experiment_records" / "tables").resolve(),
    ]
    output = Path(output_dir).resolve()
    if output == repo or repo in output.parents:
        allowed_prefixes.append(output)
    unexpected = []
    status = _git(repo, ["status", "--porcelain=v1", "--untracked-files=all"])
    for line in status.splitlines():
        relative = line[3:].strip().strip('"').replace("\\", "/")
        if " -> " in relative:
            relative = relative.split(" -> ", 1)[1]
        candidate = (repo / relative).resolve()
        if relative in allowed_files:
            continue
        if any(prefix == candidate or prefix in candidate.parents for prefix in allowed_prefixes):
            continue
        unexpected.append(relative)
    if unexpected:
        raise DynamicExperimentEvidenceError(
            "Non-evidence worktree changes appeared during training: {}".format(
                sorted(unexpected)
            )
        )
    return True


def _nested(mapping, dotted):
    value = mapping
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise DynamicExperimentEvidenceError("Missing config field {}".format(dotted))
        value = value[part]
    return value


def validate_dynamic_configuration(configuration, static_configuration,
                                   run_kind="formal"):
    required = {
        "SEED": 42,
        "MODEL.NAME": "resnet50",
        "MODEL.MULTI_GRANULARITY_PART": True,
        "MODEL.MULTI_GRANULARITY_PART_SCALES": [2, 4, 6],
        "MODEL.MULTI_GRANULARITY_PART_DIM": 256,
        "MODEL.MULTI_GRANULARITY_PART_AGGREGATION": "mean",
        "MODEL.MULTI_GRANULARITY_PART_FUSION": "concat",
        "MODEL.MULTI_GRANULARITY_DYNAMIC_GATING": True,
        "MODEL.MULTI_GRANULARITY_GATING_INPUT": "global",
        "MODEL.MULTI_GRANULARITY_GATING_TAU": 1.0,
        "MODEL.MULTI_GRANULARITY_GATING_NORMALIZATION": "scaled_softmax",
        "MODEL.CROSS_CAMERA_POSITIVE_ONLY": True,
        "MODEL.CROSS_CAMERA_POSITIVE_LAMBDA": 0.3,
        "MODEL.PART_ATTENTION": False,
        "SOLVER.MAX_EPOCHS": 120 if run_kind == "formal" else 1,
        "TEST.NECK_FEAT": "after", "TEST.FEAT_NORM": "yes",
        "TEST.RE_RANKING": "no",
    }
    for dotted, expected in required.items():
        actual = _nested(configuration, dotted)
        if actual != expected:
            raise DynamicExperimentEvidenceError(
                "Resolved protocol mismatch {}: {!r} != {!r}".format(
                    dotted, actual, expected
                )
            )
    tau = _nested(configuration, "MODEL.MULTI_GRANULARITY_GATING_TAU")
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) \
            or not math.isfinite(float(tau)) or float(tau) <= 0.0:
        raise DynamicExperimentEvidenceError("Gating temperature must be finite and positive")

    def flatten(value, prefix=""):
        rows = {}
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = "{}.{}".format(prefix, key) if prefix else str(key)
                rows.update(flatten(child, child_path))
        else:
            rows[prefix] = value
        return rows

    static_flat = flatten(static_configuration)
    dynamic_flat = flatten(configuration)
    allowed = {
        "MODEL.MULTI_GRANULARITY_DYNAMIC_GATING",
        "MODEL.MULTI_GRANULARITY_GATING_INPUT",
        "MODEL.MULTI_GRANULARITY_GATING_TAU",
        "MODEL.MULTI_GRANULARITY_GATING_NORMALIZATION",
        "OUTPUT_DIR",
    }
    if run_kind == "smoke":
        allowed.update({
            "SOLVER.MAX_EPOCHS", "SOLVER.CHECKPOINT_PERIOD", "SOLVER.EVAL_PERIOD"
        })
    differences = sorted(
        key for key in set(static_flat) | set(dynamic_flat)
        if static_flat.get(key, NOT_RECORDED) != dynamic_flat.get(key, NOT_RECORDED)
    )
    unexpected = sorted(set(differences) - allowed)
    if unexpected:
        raise DynamicExperimentEvidenceError(
            "Static/Dynamic protocol differs outside gating controls: {}".format(unexpected)
        )
    required_differences = {
        "MODEL.MULTI_GRANULARITY_DYNAMIC_GATING", "OUTPUT_DIR"
    }
    if not required_differences.issubset(differences):
        raise DynamicExperimentEvidenceError(
            "Dynamic config does not declare the required experiment differences"
        )
    return differences


def generate_run_id(experiment_id, commit, when=None):
    stamp = (when or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return "{}-{}-{}-{}".format(
        experiment_id, stamp, str(commit)[:10], uuid.uuid4().hex[:8]
    )


def _file_evidence(path, missing=NOT_RECORDED):
    path = Path(path)
    if not path.is_file():
        return {"path": missing, "size_bytes": missing, "sha256": missing}
    return {
        "path": str(path.resolve()), "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _copy_atomic(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(".{}.tmp.{}".format(destination.name, uuid.uuid4().hex))
    shutil.copyfile(str(source), str(temporary))
    os.replace(str(temporary), str(destination))


def _refresh_partial_artifact_manifest(run_dir, manifest):
    run_dir = Path(run_dir)
    artifact_path = run_dir / "artifact_manifest.json"
    indexed = {
        key: value for key, value in manifest.get("artifacts", {}).items()
        if key not in ("artifact_manifest", "run_evidence_manifest")
    }
    per_run_evidence_path = run_dir / "evidence_manifest.tsv"
    evidence_rows = []
    for artifact_type, evidence in sorted(indexed.items()):
        evidence_rows.append({
            "schema_version": UNIFIED_SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "experiment_id": manifest["experiment_id"],
            "run_kind": manifest["run_kind"], "status": manifest["status"],
            "artifact_type": artifact_type,
            "path": evidence.get("path", NOT_RECORDED),
            "size_bytes": evidence.get("size_bytes", NOT_RECORDED),
            "sha256": evidence.get("sha256", NOT_RECORDED),
            "source_checkpoint_sha256": evidence.get(
                "source_checkpoint_sha256", NOT_APPLICABLE
            ),
            "selection_rule": evidence.get(
                "selection_rule", NOT_APPLICABLE
            ),
        })
    _atomic_write(
        per_run_evidence_path,
        _render_table(evidence_rows, EVIDENCE_FIELDS, "\t"),
    )
    per_run_evidence = _file_evidence(per_run_evidence_path)
    per_run_evidence["source_checkpoint_sha256"] = NOT_APPLICABLE
    per_run_evidence["selection_rule"] = NOT_APPLICABLE
    indexed["run_evidence_manifest"] = per_run_evidence
    atomic_write_json(artifact_path, {
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "run_id": manifest["run_id"], "status": manifest["status"],
        "artifacts": indexed,
    })
    evidence = _file_evidence(artifact_path)
    evidence["source_checkpoint_sha256"] = NOT_APPLICABLE
    evidence["selection_rule"] = NOT_APPLICABLE
    manifest["artifact_manifest"] = evidence
    manifest["run_evidence_manifest"] = per_run_evidence
    manifest.setdefault("artifacts", {})[
        "run_evidence_manifest"
    ] = dict(per_run_evidence)
    manifest.setdefault("artifacts", {})["artifact_manifest"] = dict(evidence)
    return evidence


def initialize_dynamic_run(records_root, experiments_path, experiment_id,
                           run_kind, source_config, resolved_config_text,
                           output_dir, lineage, feature_evidence, command,
                           started_at_utc=None):
    records_root = Path(records_root)
    records_root.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_id(experiment_id, lineage["commit"])
    run_dir = records_root / "runs" / run_id
    if run_dir.exists():
        raise DynamicExperimentEvidenceError("Generated run_id already exists")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise DynamicExperimentEvidenceError("OUTPUT_DIR exists and is non-empty")
    run_dir.mkdir(parents=True)
    _copy_atomic(source_config, run_dir / "config_source.yml")
    _atomic_write(run_dir / "config_resolved.yml", resolved_config_text)
    atomic_write_json(run_dir / "shared_feature_compatibility.json", feature_evidence)
    gating_payload = feature_evidence["fusion_gating_signature"]
    atomic_write_json(run_dir / "fusion_gating_signature.json", gating_payload)
    started = started_at_utc or utc_now()
    manifest = {
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_family": "C2-L03-MULTI-GRANULARITY-DYNAMIC-GATING",
        "evidence_id": run_id,
        "run_id": run_id,
        "run_kind": run_kind,
        "status": "initialized",
        "state_history": [{"status": "initialized", "timestamp_utc": started}],
        "method_family": "multi_granularity_feature",
        "method_variant": "per_sample_dynamic_gating",
        "branch": lineage["branch"], "commit": lineage["commit"],
        "parent_branch": lineage["parent_branch"],
        "parent_commit": lineage["parent_commit"],
        "parent_local_commit": lineage.get(
            "parent_local_commit", lineage["parent_commit"]
        ),
        "parent_remote_tracking_commit": lineage.get(
            "parent_remote_tracking_commit", lineage["parent_commit"]
        ),
        "parent_remote_actual_commit": lineage.get(
            "parent_remote_actual_commit", lineage["parent_commit"]
        ),
        "merge_base": lineage["merge_base"], "seed": 42,
        "feature_reference_commit": feature_evidence["feature_reference_commit"],
        "feature_reference_signature_sha256": feature_evidence["feature_reference_signature_sha256"],
        "current_feature_signature_sha256": feature_evidence["current_feature_signature_sha256"],
        "feature_compatibility_status": feature_evidence["feature_compatibility_status"],
        "gating_signature_sha256": feature_evidence["fusion_gating_signature"]["current_sha256"],
        "source_config": _file_evidence(run_dir / "config_source.yml"),
        "source_config_origin": _file_evidence(source_config),
        "resolved_config": _file_evidence(run_dir / "config_resolved.yml"),
        "feature_compatibility": _file_evidence(run_dir / "shared_feature_compatibility.json"),
        "fusion_gating_signature": _file_evidence(run_dir / "fusion_gating_signature.json"),
        "command": list(command), "output_dir": str(output.resolve()),
        "console_log": _file_evidence(run_dir / "console.log"),
        "training_log": {"path": str((output / "log.txt").resolve()), "size_bytes": NOT_RECORDED, "sha256": NOT_RECORDED},
        "started_at_utc": started, "ended_at_utc": NOT_RECORDED,
        "runtime_seconds": NOT_RECORDED, "return_code": NOT_RECORDED,
        "alignment_mode": NOT_APPLICABLE,
        "alignment_temperature": NOT_APPLICABLE,
        "gating_mode": "per_sample_dynamic_gating", "gating_input": "global",
        "gating_temperature": 1.0, "gating_normalization": "scaled_softmax",
        "scale_order": "2,4,6", "metrics": {}, "notes": "tau=1.0 initial candidate",
        "records_root": str(records_root.resolve()),
        "experiments_path": str(Path(experiments_path).resolve()),
    }
    manifest["artifacts"] = {
        "source_config": dict(manifest["source_config"]),
        "source_config_origin": dict(manifest["source_config_origin"]),
        "resolved_config_snapshot": dict(manifest["resolved_config"]),
        "feature_compatibility": dict(manifest["feature_compatibility"]),
        "fusion_gating_signature": dict(manifest["fusion_gating_signature"]),
    }
    _refresh_partial_artifact_manifest(run_dir, manifest)
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "run_status.json", {
        "schema_version": UNIFIED_SCHEMA_VERSION, "run_id": run_id,
        "status": "initialized", "errors": [], "started_at_utc": started,
        "ended_at_utc": NOT_RECORDED, "return_code": NOT_RECORDED,
    })
    register_dynamic_run_state(run_dir)
    return run_dir, manifest


def transition_dynamic_run(run_dir, status, return_code=NOT_RECORDED,
                           runtime_seconds=NOT_RECORDED, error=None):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "run_manifest.json")
    status_payload = read_json(run_dir / "run_status.json")
    timestamp = utc_now()
    manifest["status"] = status
    manifest.setdefault("state_history", []).append({
        "status": status, "timestamp_utc": timestamp,
        "error": str(error) if error is not None else NOT_APPLICABLE,
    })
    status_payload["status"] = status
    if return_code != NOT_RECORDED:
        manifest["return_code"] = int(return_code)
        status_payload["return_code"] = int(return_code)
    if runtime_seconds != NOT_RECORDED:
        manifest["runtime_seconds"] = float(runtime_seconds)
        status_payload["runtime_seconds"] = float(runtime_seconds)
    if status in ("success", "failed", "incomplete", "interrupted", "training_complete"):
        manifest["ended_at_utc"] = timestamp
        status_payload["ended_at_utc"] = timestamp
    if error is not None:
        status_payload.setdefault("errors", []).append(str(error))
        manifest["notes"] = str(error)
    console = run_dir / "console.log"
    if console.is_file() and console.stat().st_size > 0:
        manifest["console_log"] = _file_evidence(console)
        manifest.setdefault("artifacts", {})["console_log"] = dict(
            manifest["console_log"]
        )
    output = Path(manifest["output_dir"])
    for artifact_type, filename in (
            ("training_log", "log.txt"),
            ("reproducibility", "reproducibility.json"),
            ("gating_epoch_statistics", "dynamic_gating_epoch_stats.jsonl")):
        candidate = output / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            manifest[artifact_type] = _file_evidence(candidate)
            manifest.setdefault("artifacts", {})[artifact_type] = dict(
                manifest[artifact_type]
            )
    for index, checkpoint in enumerate(sorted(output.glob("resnet50_checkpoint_*.pt"))):
        if checkpoint.is_file():
            manifest.setdefault("artifacts", {})[
                "incomplete_checkpoint_{:03d}".format(index)
            ] = _file_evidence(checkpoint)
    _refresh_partial_artifact_manifest(run_dir, manifest)
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "run_status.json", status_payload)
    register_dynamic_run_state(run_dir)
    return manifest


def _normalize_row(row, fields):
    return {
        field: (
            NOT_RECORDED
            if row.get(field, NOT_RECORDED) in (None, "")
            else row.get(field, NOT_RECORDED)
        )
        for field in fields
    }


def _read_table(path, delimiter):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader), list(reader.fieldnames or [])


def _render_table(rows, fields, delimiter):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), delimiter=delimiter,
                            lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(_normalize_row(row, fields))
    return output.getvalue()


def _upsert(rows, row, key="run_id"):
    value = str(row[key])
    result = [existing for existing in rows if str(existing.get(key)) != value]
    result.append(row)
    return sorted(result, key=lambda item: (str(item.get("start_time", "")), str(item.get(key, ""))))


def _artifact_rows(manifest):
    rows = []
    for artifact_type, evidence in manifest.get("artifacts", {}).items():
        if not isinstance(evidence, dict) or evidence.get("path") in (None, NOT_RECORDED, MISSING_EVIDENCE):
            continue
        rows.append({
            "schema_version": UNIFIED_SCHEMA_VERSION,
            "run_id": manifest["run_id"], "experiment_id": manifest["experiment_id"],
            "run_kind": manifest["run_kind"], "status": manifest["status"],
            "artifact_type": artifact_type, "path": evidence.get("path", NOT_RECORDED),
            "size_bytes": evidence.get("size_bytes", NOT_RECORDED),
            "sha256": evidence.get("sha256", NOT_RECORDED),
            "source_checkpoint_sha256": evidence.get("source_checkpoint_sha256", NOT_APPLICABLE),
            "selection_rule": evidence.get("selection_rule", NOT_APPLICABLE),
        })
    return rows


def _manifest_run_row(manifest):
    def evidence(name):
        return manifest.get(name, {}) if isinstance(manifest.get(name), dict) else {}
    source = evidence("source_config")
    source_origin = evidence("source_config_origin")
    resolved = evidence("resolved_config")
    console = evidence("console_log")
    training = evidence("training_log")
    feature = evidence("feature_compatibility")
    fusion_signature = evidence("fusion_gating_signature")
    checkpoint = evidence("selected_checkpoint")
    checkpoint_manifest = evidence("checkpoint_manifest")
    artifact_manifest = evidence("artifact_manifest")
    run_evidence_manifest = evidence("run_evidence_manifest")
    reproducibility = evidence("reproducibility")
    dataset = evidence("dataset_manifest")
    model = evidence("model_manifest")
    environment = evidence("environment")
    summary = evidence("dynamic_gating_summary")
    samples = evidence("gating_samples")
    metrics = manifest.get("metrics", {})
    statistics = manifest.get("gating_statistics", {})
    row = {
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "experiment_family": manifest.get("experiment_family", NOT_RECORDED),
        "evidence_id": manifest.get("evidence_id", manifest["run_id"]),
        "run_id": manifest["run_id"], "run_kind": manifest["run_kind"],
        "status": manifest["status"], "method_family": manifest["method_family"],
        "method_variant": manifest["method_variant"], "branch": manifest["branch"],
        "commit": manifest["commit"], "parent_branch": manifest["parent_branch"],
        "parent_commit": manifest["parent_commit"], "merge_base": manifest["merge_base"],
        "feature_reference_commit": manifest["feature_reference_commit"],
        "feature_reference_signature_sha256": manifest["feature_reference_signature_sha256"],
        "current_feature_signature_sha256": manifest["current_feature_signature_sha256"],
        "feature_compatibility_status": manifest["feature_compatibility_status"],
        "feature_compatibility_evidence_path": feature.get("path", NOT_RECORDED),
        "feature_compatibility_evidence_size_bytes": feature.get("size_bytes", NOT_RECORDED),
        "feature_compatibility_evidence_sha256": feature.get("sha256", NOT_RECORDED),
        "gating_signature_sha256": manifest.get("gating_signature_sha256", NOT_RECORDED),
        "gating_signature_path": fusion_signature.get("path", NOT_RECORDED),
        "gating_signature_size_bytes": fusion_signature.get("size_bytes", NOT_RECORDED),
        "gating_signature_evidence_sha256": fusion_signature.get("sha256", NOT_RECORDED),
        "seed": manifest["seed"], "output_dir": manifest["output_dir"],
        "source_config_origin_path": source_origin.get("path", NOT_RECORDED),
        "source_config_origin_size_bytes": source_origin.get("size_bytes", NOT_RECORDED),
        "source_config_origin_sha256": source_origin.get("sha256", NOT_RECORDED),
        "gpu": manifest.get("gpu", NOT_RECORDED),
        "start_time": manifest.get("started_at_utc", NOT_RECORDED),
        "end_time": manifest.get("ended_at_utc", NOT_RECORDED),
        "runtime_seconds": manifest.get("runtime_seconds", NOT_RECORDED),
        "return_code": manifest.get("return_code", NOT_RECORDED),
        "alignment_mode": manifest.get("alignment_mode", NOT_APPLICABLE),
        "alignment_temperature": manifest.get("alignment_temperature", NOT_APPLICABLE),
        "gating_mode": manifest["gating_mode"], "gating_input": manifest["gating_input"],
        "gating_temperature": manifest["gating_temperature"],
        "gating_normalization": manifest["gating_normalization"],
        "scale_order": manifest["scale_order"],
        "rank1_percent": metrics.get("rank1_percent", NOT_RECORDED),
        "rank5_percent": metrics.get("rank5_percent", NOT_RECORDED),
        "rank10_percent": metrics.get("rank10_percent", NOT_RECORDED),
        "map_percent": metrics.get("map_percent", NOT_RECORDED),
        "best_epoch": metrics.get("best_epoch", NOT_RECORDED),
        "selected_epoch": metrics.get("selected_epoch", NOT_RECORDED),
        "notes": manifest.get("notes", NOT_RECORDED),
    }
    for prefix, item in (
            ("source_config", source), ("resolved_config", resolved),
            ("training_log", training), ("console_log", console),
            ("selected_checkpoint", checkpoint), ("checkpoint_manifest", checkpoint_manifest),
            ("artifact_manifest", artifact_manifest), ("reproducibility", reproducibility),
            ("run_evidence_manifest", run_evidence_manifest),
            ("dynamic_gating_summary", summary), ("gating_samples", samples)):
        row["{}_path".format(prefix)] = item.get("path", NOT_RECORDED)
        row["{}_size_bytes".format(prefix)] = item.get("size_bytes", NOT_RECORDED)
        row["{}_sha256".format(prefix)] = item.get("sha256", NOT_RECORDED)
    for prefix, item in (("dataset_manifest", dataset), ("model_manifest", model), ("environment", environment)):
        row["{}_path".format(prefix)] = item.get("path", NOT_RECORDED)
        row["{}_sha256".format(prefix)] = item.get("sha256", NOT_RECORDED)
    row["dynamic_gating_summary_source_checkpoint_sha256"] = summary.get("source_checkpoint_sha256", NOT_RECORDED)
    row["gating_samples_source_checkpoint_sha256"] = samples.get("source_checkpoint_sha256", NOT_RECORDED)
    row["gating_sample_selection_rule"] = samples.get("selection_rule", NOT_RECORDED)
    for field in GATING_STAT_FIELDS:
        row[field] = statistics.get(field, NOT_RECORDED)
    return _normalize_row(row, RUN_FIELDS)


def _markdown_escape(value):
    text = str(value if value is not None else NOT_RECORDED)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _markdown_section(start, end, title, fields, rows):
    lines = [start, "## {}".format(title), "", "| " + " | ".join(fields) + " |",
             "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(row.get(field, NOT_RECORDED)) for field in fields) + " |")
    lines.extend([end, ""])
    return "\n".join(lines)


def _replace_section(content, start, end, replacement):
    if start in content or end in content:
        if content.count(start) != 1 or content.count(end) != 1:
            raise DynamicExperimentEvidenceError("EXPERIMENTS.md generated markers are malformed")
        before, rest = content.split(start, 1)
        _, after = rest.split(end, 1)
        return before.rstrip() + "\n\n" + replacement.rstrip() + "\n" + after.lstrip("\r\n")
    return content.rstrip() + "\n\n" + replacement


def _checkpoint_rows(records_root, run_rows):
    rows = []
    for run in run_rows:
        path = run.get("checkpoint_manifest_path")
        if path in (None, "", NOT_RECORDED, MISSING_EVIDENCE):
            continue
        checkpoint_rows, _ = _read_table(path, "\t")
        for checkpoint in checkpoint_rows:
            rows.append({
                "run_id": run["run_id"], "experiment_id": run["experiment_id"],
                "run_kind": run["run_kind"],
                "checkpoint_path": checkpoint.get("relative_path", checkpoint.get("path", NOT_RECORDED)),
                "size_bytes": checkpoint.get("file_size", checkpoint.get("size_bytes", NOT_RECORDED)),
                "ignite_epoch": checkpoint.get("epoch", NOT_RECORDED),
                "global_iteration": checkpoint.get("global_iteration", NOT_RECORDED),
                "sha256": checkpoint.get("sha256", NOT_RECORDED),
                "selected": checkpoint.get("selected", "false"),
            })
    unique = {(row["run_id"], row["checkpoint_path"]): row for row in rows}
    return [unique[key] for key in sorted(unique)]


def _markdown_content(experiments_path, run_rows, formal_rows, checkpoint_rows):
    path = Path(experiments_path)
    content = path.read_text(encoding="utf-8") if path.is_file() else "# Experiments\n"
    sections = (
        (AUTO_RUNS_START, AUTO_RUNS_END, "Run Registry / All Recorded Runs", RUN_FIELDS, run_rows),
        (AUTO_FORMAL_START, AUTO_FORMAL_END, "Formal Results", RUN_FIELDS, formal_rows),
        (AUTO_CHECKPOINTS_START, AUTO_CHECKPOINTS_END, "Checkpoint Evidence", CHECKPOINT_FIELDS, checkpoint_rows),
    )
    for start, end, title, fields, rows in sections:
        content = _replace_section(content, start, end, _markdown_section(start, end, title, fields, rows))
    return content


def _transactional_write(targets):
    previous = {}
    staged = {}
    try:
        for path, text in targets.items():
            path = Path(path)
            previous[path] = path.read_bytes() if path.is_file() else None
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(".{}.stage.{}".format(path.name, uuid.uuid4().hex))
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            staged[path] = temporary
        for path, temporary in staged.items():
            os.replace(str(temporary), str(path))
    except BaseException:
        for temporary in staged.values():
            if temporary.exists():
                temporary.unlink()
        for path, payload in previous.items():
            try:
                if payload is None:
                    if path.exists():
                        path.unlink()
                else:
                    temporary = path.with_name(".{}.rollback.{}".format(path.name, uuid.uuid4().hex))
                    temporary.write_bytes(payload)
                    os.replace(str(temporary), str(path))
            except BaseException:
                pass
        raise


def register_dynamic_run_state(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "run_manifest.json")
    records_root = Path(manifest["records_root"])
    experiments_path = Path(manifest["experiments_path"])
    runs_path = records_root / "runs.csv"
    formal_path = records_root / "tables" / "main_results.csv"
    evidence_path = records_root / "evidence_manifest.tsv"
    run_rows, old_run_fields = _read_table(runs_path, ",")
    formal_rows, old_formal_fields = _read_table(formal_path, ",")
    evidence_rows, old_evidence_fields = _read_table(evidence_path, "\t")
    persisted_run_fields = tuple(RUN_FIELDS) + tuple(
        field for field in old_run_fields if field not in RUN_FIELDS
    )
    persisted_formal_fields = tuple(RUN_FIELDS) + tuple(
        field for field in old_formal_fields if field not in RUN_FIELDS
    )
    persisted_evidence_fields = tuple(EVIDENCE_FIELDS) + tuple(
        field for field in old_evidence_fields if field not in EVIDENCE_FIELDS
    )
    row = _manifest_run_row(manifest)
    run_rows = _upsert(run_rows, row)
    formal_rows = [item for item in formal_rows if item.get("run_id") != row["run_id"]]
    if row["run_kind"] == "formal" and row["status"] == "success":
        formal_rows = _upsert(formal_rows, row)
    evidence_rows = [item for item in evidence_rows if item.get("run_id") != row["run_id"]]
    evidence_rows.extend(_artifact_rows(manifest))
    evidence_rows = sorted(evidence_rows, key=lambda item: (str(item.get("run_id", "")), str(item.get("artifact_type", "")), str(item.get("path", ""))))
    checkpoints = _checkpoint_rows(records_root, run_rows)
    markdown = _markdown_content(experiments_path, run_rows, formal_rows, checkpoints)
    _transactional_write({
        runs_path: _render_table(run_rows, persisted_run_fields, ","),
        formal_path: _render_table(formal_rows, persisted_formal_fields, ","),
        evidence_path: _render_table(
            evidence_rows, persisted_evidence_fields, "\t"
        ),
        experiments_path: markdown,
    })
    return row


def refresh_experiments_markdown(experiments_path, records_root):
    records_root = Path(records_root)
    run_rows, _ = _read_table(records_root / "runs.csv", ",")
    formal_rows, _ = _read_table(records_root / "tables" / "main_results.csv", ",")
    checkpoints = _checkpoint_rows(records_root, run_rows)
    content = _markdown_content(experiments_path, run_rows, formal_rows, checkpoints)
    _atomic_write(experiments_path, content)
    return content


def migrate_unified_schema(path, fields, delimiter=","):
    rows, old_fields = _read_table(path, delimiter)
    # Unknown historical columns are retained after the current schema rather
    # than discarded. Missing new evidence remains explicitly not_recorded.
    merged_fields = list(fields) + [field for field in old_fields if field not in fields]
    _atomic_write(path, _render_table(rows, merged_fields, delimiter))
    return merged_fields, rows


def record_file_artifact(manifest, artifact_type, path,
                         source_checkpoint_sha256=NOT_APPLICABLE,
                         selection_rule=NOT_APPLICABLE, required=False):
    path = Path(path)
    if not path.is_file() or (required and path.stat().st_size <= 0):
        if required:
            raise DynamicExperimentEvidenceError(
                "Required artifact is missing or empty: {}".format(path)
            )
        evidence = {"path": NOT_RECORDED, "size_bytes": NOT_RECORDED, "sha256": NOT_RECORDED}
    else:
        evidence = _file_evidence(path)
    evidence["source_checkpoint_sha256"] = source_checkpoint_sha256
    evidence["selection_rule"] = selection_rule
    manifest.setdefault("artifacts", {})[artifact_type] = evidence
    return evidence


def validate_seed42_reproducibility(reproducibility, environment,
                                    source_sha256, resolved_sha256):
    if reproducibility.get("seed") != 42:
        raise DynamicExperimentEvidenceError("Applied reproducibility seed is not 42")
    chain = reproducibility.get("seed_chain", {})
    for field in (
            "source_config_seed", "resolved_config_seed",
            "applied_training_seed", "reproducibility_metadata_seed"):
        if chain.get(field) != 42:
            raise DynamicExperimentEvidenceError(
                "Seed42 evidence chain mismatch: {}".format(field)
            )
    random_state = reproducibility.get("random_state", {})
    for field in (
            "seed", "python_random_seed", "numpy_seed", "torch_cpu_seed",
            "torch_cuda_manual_seed_all_seed"):
        if random_state.get(field) != 42:
            raise DynamicExperimentEvidenceError(
                "Applied random seed evidence mismatch: {}".format(field)
            )
    for field in (
            "python_random_seeded", "numpy_seeded", "torch_cpu_seeded",
            "cuda_available", "torch_cuda_manual_seed_all_called",
            "torch_cuda_all_seeded", "cudnn_deterministic"):
        if random_state.get(field) is not True:
            raise DynamicExperimentEvidenceError(
                "Applied deterministic seed evidence is missing: {}".format(field)
            )
    if random_state.get("cudnn_benchmark") is not False:
        raise DynamicExperimentEvidenceError("cudnn.benchmark must be false")
    for payload, label in ((random_state, "random_state"), (environment, "environment")):
        if str(payload.get("pythonhashseed")) != "42":
            raise DynamicExperimentEvidenceError(
                "{} PYTHONHASHSEED must be 42".format(label)
            )
        if payload.get("cublas_workspace_config") != ":4096:8":
            raise DynamicExperimentEvidenceError(
                "{} CUBLAS_WORKSPACE_CONFIG mismatch".format(label)
            )
    if int(environment.get("gpu_count", 0)) <= 0:
        raise DynamicExperimentEvidenceError("CUDA/GPU environment evidence is missing")
    worker = reproducibility.get("data_loader_worker_seeding", {})
    if (worker.get("enabled") is not True or worker.get("num_workers") != 8
            or "torch.initial_seed()" not in str(worker.get("scheme", ""))):
        raise DynamicExperimentEvidenceError("DataLoader worker Seed42 evidence mismatch")
    sampler = reproducibility.get("random_identity_sampler", {})
    if sampler.get("base_seed") != 42:
        raise DynamicExperimentEvidenceError("Sampler Seed42 evidence mismatch")
    generators = reproducibility.get("data_loader_generators", {})
    if generators.get("stream_seeds") != {
            "train": 42, "query": 43, "gallery": 44}:
        raise DynamicExperimentEvidenceError("DataLoader generator Seed42 evidence mismatch")
    configuration = reproducibility.get("configuration", {})
    if configuration.get("source_file_sha256") != source_sha256:
        raise DynamicExperimentEvidenceError("Reproducibility source config SHA mismatch")
    if configuration.get("resolved_file_sha256") != resolved_sha256:
        raise DynamicExperimentEvidenceError("Reproducibility resolved config SHA mismatch")
    return True


_DYNAMIC_CHECKPOINT_RE = re.compile(
    r"^resnet50_checkpoint_(?P<global_iteration>\d+)\.pt$"
)
_EPOCH_EVIDENCE_RE = re.compile(
    r"EPOCH_EVIDENCE epoch=(?P<epoch>\d+) "
    r"global_iteration=(?P<global_iteration>\d+) "
    r"epoch_length=(?P<epoch_length>\d+)"
)


def build_dynamic_checkpoint_manifest(output_dir, validation_records):
    """Bind checkpoints only through Ignite EPOCH_EVIDENCE observations."""
    output = Path(output_dir)
    log_path = output / "log.txt"
    if not log_path.is_file():
        raise DynamicExperimentEvidenceError("Training log is missing")
    mappings = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EPOCH_EVIDENCE_RE.search(line)
        if match is None:
            continue
        row = {key: int(value) for key, value in match.groupdict().items()}
        epoch = row["epoch"]
        if epoch in mappings and mappings[epoch] != row:
            raise DynamicExperimentEvidenceError("Conflicting Ignite epoch evidence")
        mappings[epoch] = row
    if not mappings:
        raise DynamicExperimentEvidenceError("No authoritative EPOCH_EVIDENCE was recorded")
    validation_by_epoch = {int(row["epoch"]): row for row in validation_records}
    rows = []
    for checkpoint in sorted(output.glob("resnet50_checkpoint_*.pt")):
        match = _DYNAMIC_CHECKPOINT_RE.fullmatch(checkpoint.name)
        if match is None:
            continue
        iteration = int(match.group("global_iteration"))
        matching = [item for item in mappings.values() if item["global_iteration"] == iteration]
        if len(matching) != 1:
            raise DynamicExperimentEvidenceError(
                "Checkpoint lacks unique authoritative Ignite epoch evidence: {}".format(
                    checkpoint.name
                )
            )
        evidence = matching[0]
        epoch = evidence["epoch"]
        validation = validation_by_epoch.get(epoch)
        if validation is None or int(validation["global_iteration"]) != iteration:
            raise DynamicExperimentEvidenceError(
                "Checkpoint/validation/Ignite evidence mismatch at epoch {}".format(epoch)
            )
        rows.append({
            "epoch": epoch, "global_iteration": iteration,
            "epoch_length": evidence["epoch_length"], "filename": checkpoint.name,
            "path": str(checkpoint.resolve()), "size_bytes": checkpoint.stat().st_size,
            "artifact_type": "model_checkpoint", "relative_path": checkpoint.name,
            "file_size": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint),
            "selected": "false",
        })
    if not rows:
        raise DynamicExperimentEvidenceError("No model checkpoints were recorded")
    fields = (
        "epoch", "global_iteration", "epoch_length", "filename", "path",
        "size_bytes", "sha256", "selected", "artifact_type", "relative_path",
        "file_size",
    )
    _atomic_write(output / "checkpoint_manifest.tsv", _render_table(rows, fields, "\t"))
    return rows


def select_dynamic_checkpoint(checkpoint_rows, validation_records):
    if not validation_records:
        raise DynamicExperimentEvidenceError("No validation records were recorded")
    selected_validation = sorted(
        validation_records,
        key=lambda row: (
            float(row["rank1_percent"]), float(row["map_percent"]),
            -int(row["epoch"]),
        ), reverse=True,
    )[0]
    selected_epoch = int(selected_validation["epoch"])
    matching = [row for row in checkpoint_rows if int(row["epoch"]) == selected_epoch]
    if len(matching) != 1:
        raise DynamicExperimentEvidenceError(
            "Selected validation epoch must bind to exactly one checkpoint"
        )
    for row in checkpoint_rows:
        row["selected"] = "true" if row is matching[0] else "false"
    output = Path(matching[0]["path"]).parent
    fields = (
        "epoch", "global_iteration", "epoch_length", "filename", "path",
        "size_bytes", "sha256", "selected", "artifact_type", "relative_path",
        "file_size",
    )
    _atomic_write(output / "checkpoint_manifest.tsv", _render_table(checkpoint_rows, fields, "\t"))
    return matching[0], selected_validation


def seal_dynamic_run_evidence(run_dir, environment, dataset_manifest,
                              model_manifest, checkpoint_rows,
                              selected_checkpoint, metrics,
                              dynamic_summary_path, gating_samples_path,
                              gating_statistics, runtime_seconds, return_code=0):
    """Validate and transactionally publish a successful run."""
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "run_manifest.json")
    output = Path(manifest["output_dir"])
    if manifest.get("feature_compatibility_status") != "compatible":
        raise DynamicExperimentEvidenceError("Shared feature compatibility is not compatible")
    console = record_file_artifact(manifest, "console_log", run_dir / "console.log", required=True)
    training = record_file_artifact(manifest, "training_log", output / "log.txt", required=True)
    source = record_file_artifact(manifest, "source_config", run_dir / "config_source.yml", required=True)
    source_origin_path = Path(manifest.get("source_config_origin", {}).get("path", ""))
    source_origin = record_file_artifact(
        manifest, "source_config_origin", source_origin_path, required=True
    )
    if source_origin["sha256"] != source["sha256"]:
        raise DynamicExperimentEvidenceError("Source config origin/snapshot SHA mismatch")
    resolved_output = output / "config_resolved.yml"
    resolved = record_file_artifact(manifest, "resolved_config", resolved_output, required=True)
    snapshot_resolved = run_dir / "config_resolved.yml"
    if sha256_file(snapshot_resolved) != resolved["sha256"]:
        raise DynamicExperimentEvidenceError("Resolved config snapshot SHA256 mismatch")
    feature = record_file_artifact(manifest, "feature_compatibility", run_dir / "shared_feature_compatibility.json", required=True)
    reproducibility = record_file_artifact(manifest, "reproducibility", output / "reproducibility.json", required=True)
    reproducibility_payload = read_json(output / "reproducibility.json")
    validate_seed42_reproducibility(
        reproducibility_payload, environment, source["sha256"], resolved["sha256"]
    )
    env_path = run_dir / "environment.json"
    dataset_path = run_dir / "dataset_manifest.json"
    model_path = run_dir / "model_manifest.json"
    atomic_write_json(env_path, environment)
    atomic_write_json(dataset_path, dataset_manifest)
    atomic_write_json(model_path, model_manifest)
    env_evidence = record_file_artifact(manifest, "environment", env_path, required=True)
    dataset_evidence = record_file_artifact(manifest, "dataset_manifest", dataset_path, required=True)
    model_evidence = record_file_artifact(manifest, "model_manifest", model_path, required=True)
    checkpoint_manifest = output / "checkpoint_manifest.tsv"
    checkpoint_evidence = record_file_artifact(manifest, "checkpoint_manifest", checkpoint_manifest, required=True)
    selected_path = output / selected_checkpoint["relative_path"]
    selected = record_file_artifact(manifest, "selected_checkpoint", selected_path, required=True)
    if selected["sha256"] != selected_checkpoint["sha256"]:
        raise DynamicExperimentEvidenceError("Selected checkpoint SHA256 mismatch")
    selection_rule = (
        "sha256(stable_sample_key) ascending; first 256 query+gallery samples"
    )
    summary = record_file_artifact(
        manifest, "dynamic_gating_summary", dynamic_summary_path,
        source_checkpoint_sha256=selected["sha256"],
        selection_rule=selection_rule, required=True,
    )
    samples = record_file_artifact(
        manifest, "gating_samples", gating_samples_path,
        source_checkpoint_sha256=selected["sha256"],
        selection_rule=selection_rule, required=True,
    )
    if read_json(dynamic_summary_path).get("source_checkpoint_sha256") != selected["sha256"]:
        raise DynamicExperimentEvidenceError("Gating summary checkpoint binding mismatch")
    for field in GATING_STAT_FIELDS:
        value = gating_statistics.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise DynamicExperimentEvidenceError("Missing/non-finite gating statistic {}".format(field))
    if int(gating_statistics["gating_sample_count"]) <= 0:
        raise DynamicExperimentEvidenceError("Successful run must record gating samples")
    manifest.update({
        "console_log": console, "training_log": training,
        "source_config": source, "source_config_origin": source_origin,
        "resolved_config": resolved,
        "feature_compatibility": feature, "reproducibility": reproducibility,
        "environment": env_evidence, "dataset_manifest": dataset_evidence,
        "model_manifest": model_evidence, "checkpoint_manifest": checkpoint_evidence,
        "selected_checkpoint": selected, "dynamic_gating_summary": summary,
        "gating_samples": samples, "gating_statistics": gating_statistics,
        "metrics": metrics, "gpu": environment.get("gpus", NOT_RECORDED),
        "runtime_seconds": float(runtime_seconds), "return_code": int(return_code),
    })
    # Every checkpoint is registered independently and must match its actual file.
    for index, row in enumerate(checkpoint_rows):
        checkpoint_path = output / row["relative_path"]
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != row["sha256"]:
            raise DynamicExperimentEvidenceError("Checkpoint manifest SHA256 mismatch")
        record_file_artifact(
            manifest, "checkpoint_{:03d}".format(index), checkpoint_path,
            source_checkpoint_sha256=row["sha256"],
        )
    # Artifact manifest intentionally excludes its own hash to avoid recursion.
    manifest["status"] = "success"
    ended = utc_now()
    manifest["ended_at_utc"] = ended
    manifest.setdefault("state_history", []).append({"status": "success", "timestamp_utc": ended})
    artifact_manifest = _refresh_partial_artifact_manifest(run_dir, manifest)
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    status = read_json(run_dir / "run_status.json")
    status.update({"status": "success", "ended_at_utc": ended, "return_code": int(return_code)})
    atomic_write_json(run_dir / "run_status.json", status)
    try:
        register_dynamic_run_state(run_dir)
    except BaseException as error:
        # Success is forbidden if the canonical Markdown/registry transaction fails.
        manifest["status"] = "incomplete"
        manifest["notes"] = "EXPERIMENTS.md/registry update failed: {}".format(error)
        manifest.setdefault("state_history", []).append({
            "status": "incomplete", "timestamp_utc": utc_now(),
            "error": manifest["notes"],
        })
        _refresh_partial_artifact_manifest(run_dir, manifest)
        atomic_write_json(run_dir / "run_manifest.json", manifest)
        status["status"] = "incomplete"
        status.setdefault("errors", []).append(manifest["notes"])
        atomic_write_json(run_dir / "run_status.json", status)
        try:
            register_dynamic_run_state(run_dir)
        except BaseException:
            pass
        raise
    return manifest
