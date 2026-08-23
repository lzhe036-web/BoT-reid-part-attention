# encoding: utf-8
"""Fail-closed evidence gate for formal experiments that require a smoke run."""

from __future__ import absolute_import

import csv
import json
import subprocess
from pathlib import Path

import yaml

from utils.experiment_recording import sha256_file


PURE_EVIDENCE_PATHS = ("EXPERIMENTS.md", "experiment_records/")


class SmokeGateError(RuntimeError):
    """Raised when no recorded smoke run safely authorizes a formal run."""


def _require(condition, message):
    if not condition:
        raise SmokeGateError(message)


def _read_json(path):
    path = Path(path)
    _require(path.is_file(), "missing {}".format(path.name))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise SmokeGateError(
            "invalid {}: {}".format(path.name, error)
        )


def _read_yaml(path):
    path = Path(path)
    _require(path.is_file(), "missing {}".format(path.name))
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as error:
        raise SmokeGateError(
            "invalid {}: {}".format(path.name, error)
        )
    _require(isinstance(value, dict), "{} is not a mapping".format(path.name))
    return value


def _changed_leaf_paths(left, right, prefix=""):
    paths = set()
    for key in set(left) | set(right):
        path = "{}.{}".format(prefix, key) if prefix else str(key)
        if key not in left or key not in right:
            paths.add(path)
            continue
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            paths.update(_changed_leaf_paths(left_value, right_value, path))
        elif left_value != right_value:
            paths.add(path)
    return paths


def _require_file_hash(path, expected, label):
    path = Path(path)
    _require(path.is_file(), "{} is missing: {}".format(label, path))
    _require(
        sha256_file(path) == expected,
        "{} SHA256 differs".format(label),
    )


def _git_is_ancestor(repo_root, ancestor, descendant):
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise SmokeGateError(
            "cannot validate smoke commit ancestry: {}".format(
                completed.stderr.strip()
            )
        )
    return completed.returncode == 0


def _changed_commit_paths(repo_root, ancestor, descendant):
    output = subprocess.check_output(
        [
            "git", "-c", "core.quotepath=false", "diff", "--name-only",
            "{}..{}".format(ancestor, descendant),
        ],
        cwd=str(repo_root), text=True,
    )
    return [line.strip().replace("\\", "/") for line in output.splitlines()
            if line.strip()]


def _is_pure_evidence_path(path):
    normalized = str(path).replace("\\", "/")
    return (
        normalized == PURE_EVIDENCE_PATHS[0]
        or normalized.startswith(PURE_EVIDENCE_PATHS[1])
    )


def _validate_checkpoint_evidence(run_dir, status):
    checkpoint_manifest = run_dir / "checkpoint_manifest.tsv"
    _require(checkpoint_manifest.is_file(), "missing checkpoint_manifest.tsv")
    with checkpoint_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    _require(rows, "smoke checkpoint manifest is empty")
    _require(
        {int(row["epoch"]) for row in rows} == {1},
        "smoke checkpoint manifest is not exactly epoch 1",
    )
    selected = [row for row in rows if row.get("selected") == "True"]
    _require(len(selected) == 1, "smoke must have one selected checkpoint")
    selected_row = selected[0]
    _require(
        int(selected_row.get("global_iteration", "0")) > 0,
        "smoke selected checkpoint lacks real iteration evidence",
    )
    checkpoint_path = Path(selected_row["path"])
    _require_file_hash(
        checkpoint_path, selected_row["sha256"], "smoke selected checkpoint"
    )
    _require(status.get("selected_epoch") == 1, "smoke selected epoch is not 1")

    metrics = _read_json(run_dir / "metrics_summary.json")
    _require(metrics.get("selected_epoch") == 1, "smoke metrics epoch is not 1")
    _require(
        metrics.get("checkpoint_sha256") == selected_row["sha256"],
        "smoke selected checkpoint evidence does not match metrics",
    )


def _validate_artifact_manifest(run_dir, manifest):
    path = run_dir / "artifact_hashes.tsv"
    _require(path.is_file(), "missing artifact_hashes.tsv")
    _require(
        path.stat().st_size == manifest.get("artifact_manifest_size_bytes"),
        "smoke artifact manifest size differs",
    )
    _require(
        sha256_file(path) == manifest.get("artifact_manifest_sha256"),
        "smoke artifact manifest SHA256 differs",
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_type = {row.get("artifact_type"): row for row in rows}
    required = {
        "source_config", "resolved_config", "console_log",
        "dataset_manifest", "environment", "environment_packages",
        "feature_compatibility", "metrics_summary", "model_manifest",
        "reproducibility", "checkpoint_manifest", "run_status",
        "training_log", "selected_checkpoint",
    }
    _require(
        not (required - set(by_type)),
        "smoke artifact manifest lacks {}".format(
            sorted(required - set(by_type))
        ),
    )
    for artifact_type in required:
        row = by_type[artifact_type]
        artifact_path = Path(row["path"])
        if not artifact_path.is_absolute():
            artifact_path = run_dir / artifact_path
        _require_file_hash(
            artifact_path, row.get("sha256"),
            "smoke {}".format(artifact_type),
        )
        _require(
            artifact_path.stat().st_size == int(row.get("size_bytes", -1)),
            "smoke {} size differs".format(artifact_type),
        )


def _validate_candidate(
        repo_root, run_dir, formal_config_path, formal_configuration,
        current_commit, expected_branch, expected_experiment_id,
        expected_experiment_family, feature_compatibility,
        expected_parent_branch=None, expected_parent_commit=None,
        expected_merge_base=None, expected_protocol_signature=None,
        expected_implementation_signature=None,
        expected_dataset_manifest=None):
    manifest = _read_json(run_dir / "run_manifest.json")
    status = _read_json(run_dir / "run_status.json")
    _require(manifest.get("run_kind") == "smoke", "run_kind is not smoke")
    _require(status.get("status") == "success", "smoke status is not success")
    _require(status.get("phase") == "complete", "smoke phase is not complete")
    _require(status.get("training_exit_code") == 0, "smoke training did not exit 0")
    _require(
        float(status.get("training_runtime_seconds", 0.0)) > 0.0,
        "smoke lacks positive training runtime evidence",
    )
    _require(
        manifest.get("experiment_id") == expected_experiment_id,
        "smoke experiment_id differs",
    )
    _require(
        manifest.get("experiment_family") == expected_experiment_family,
        "smoke experiment_family differs",
    )
    _require(manifest.get("branch") == expected_branch, "smoke branch differs")
    _require(
        manifest.get("expected_branch") == expected_branch,
        "smoke expected_branch differs",
    )
    if expected_parent_branch is not None:
        _require(
            manifest.get("parent_branch") == expected_parent_branch,
            "smoke parent branch differs",
        )
    if expected_parent_commit is not None:
        _require(
            manifest.get("parent_commit") == expected_parent_commit,
            "smoke parent commit differs",
        )
    if expected_merge_base is not None:
        _require(
            manifest.get("merge_base") == expected_merge_base,
            "smoke merge-base differs",
        )

    source_copy = run_dir / "config_source.yml"
    resolved_copy = run_dir / "config_resolved.yml"
    formal_source_sha = sha256_file(formal_config_path)
    _require(
        manifest.get("config_source_sha256") == formal_source_sha,
        "smoke source config signature differs from formal",
    )
    _require_file_hash(
        source_copy, manifest.get("config_source_sha256"),
        "smoke source config",
    )
    _require_file_hash(
        resolved_copy, manifest.get("config_resolved_sha256"),
        "smoke resolved config",
    )
    for path, field, label in (
            (source_copy, "config_source_size_bytes", "source config"),
            (resolved_copy, "config_resolved_size_bytes", "resolved config")):
        if expected_protocol_signature is not None:
            _require(
                path.stat().st_size == manifest.get(field),
                "smoke {} size differs".format(label),
            )
    smoke_configuration = _read_yaml(resolved_copy)
    protocol_differences = _changed_leaf_paths(
        formal_configuration, smoke_configuration
    )
    _require(
        protocol_differences == {
            "SOLVER.MAX_EPOCHS", "SOLVER.CHECKPOINT_PERIOD",
            "SOLVER.EVAL_PERIOD", "OUTPUT_DIR",
        },
        "smoke/formal protocol signature differs: {}".format(
            sorted(protocol_differences)
        ),
    )
    solver = smoke_configuration.get("SOLVER", {})
    _require(
        solver.get("MAX_EPOCHS") == 1
        and solver.get("CHECKPOINT_PERIOD") == 1
        and solver.get("EVAL_PERIOD") == 1,
        "smoke resolved config is not strictly one epoch",
    )
    _require(
        smoke_configuration.get("OUTPUT_DIR")
        != formal_configuration.get("OUTPUT_DIR"),
        "smoke and formal OUTPUT_DIR must differ",
    )
    if expected_protocol_signature is not None:
        _require(
            smoke_configuration.get("OUTPUT_DIR") == "{}_smoke".format(
                str(formal_configuration.get("OUTPUT_DIR")).rstrip("/")
            ),
            "smoke OUTPUT_DIR is not the deterministic _smoke directory",
        )
        _require(
            manifest.get("protocol_signature_sha256")
            == expected_protocol_signature,
            "smoke protocol signature differs",
        )
        _require(
            manifest.get("implementation_signature_sha256")
            == expected_implementation_signature,
            "smoke implementation signature differs",
        )
        _require(
            manifest.get("git_preflight_clean") is True,
            "smoke clean-worktree preflight is missing",
        )
        _require(
            manifest.get("git_status_preflight") == [],
            "smoke preflight Git status is not empty",
        )

    model = formal_configuration.get("MODEL", {})
    expected_fields = {
        "seed": formal_configuration.get("SEED"),
        "pcc_lambda": model.get("PCC_LAMBDA"),
        "pcc_parts": model.get("PCC_PARTS"),
        "pcc_mode": model.get("PCC_MODE"),
        "alignment_temperature": model.get("PCC_SOFTMIN_TAU"),
    }
    if expected_protocol_signature is not None:
        expected_fields["cross_camera_positive_lambda"] = model.get(
            "CROSS_CAMERA_POSITIVE_LAMBDA"
        )
    mismatched = {
        key: (manifest.get(key), value)
        for key, value in expected_fields.items()
        if manifest.get(key) != value
    }
    _require(
        not mismatched,
        "smoke hyperparameter evidence differs: {}".format(mismatched),
    )
    for field in (
            "feature_reference_commit",
            "feature_reference_signature_sha256",
            "current_feature_signature_sha256",
            "feature_compatibility_status"):
        _require(
            manifest.get(field) == feature_compatibility.get(field),
            "smoke {} differs from formal".format(field),
        )
    _require(
        manifest.get("feature_compatibility_status") == "compatible",
        "smoke feature signature is not compatible",
    )

    console = run_dir / "console.log"
    _require_file_hash(
        console, manifest.get("console_log_sha256"), "smoke console log"
    )
    training_log = Path(manifest.get("training_log_path", ""))
    _require_file_hash(
        training_log, manifest.get("training_log_sha256"),
        "smoke training log",
    )
    _validate_checkpoint_evidence(run_dir, status)
    if expected_dataset_manifest is not None:
        dataset = _read_json(run_dir / "dataset_manifest.json")
        expected_dataset_sha = expected_dataset_manifest.get(
            "dataset_manifest_sha256"
        )
        _require(
            dataset.get("dataset_manifest_sha256") == expected_dataset_sha,
            "smoke dataset manifest signature differs",
        )
        _require(
            manifest.get("dataset_manifest_sha256") == expected_dataset_sha,
            "smoke manifest dataset signature differs",
        )
        for field in ("dataset", "sampler", "batch_size", "num_instance"):
            _require(
                dataset.get(field) == expected_dataset_manifest.get(field),
                "smoke dataset/protocol {} differs".format(field),
            )
        _validate_artifact_manifest(run_dir, manifest)
        environment = _read_json(run_dir / "environment.json")
        _require(environment.get("gpus"), "smoke GPU evidence is missing")
        _require(
            environment.get("git_branch") == manifest.get("branch")
            and environment.get("git_commit") == manifest.get("commit_id"),
            "smoke environment Git evidence differs",
        )
        reproducibility = _read_json(run_dir / "reproducibility.json")
        _require(
            reproducibility.get("status") == "complete"
            and reproducibility.get("applied_seed")
            == formal_configuration.get("SEED")
            and reproducibility.get("cudnn_deterministic") is True
            and reproducibility.get("cudnn_benchmark") is False,
            "smoke reproducibility/seed evidence is incomplete",
        )

    smoke_commit = manifest.get("commit_id", "")
    _require(len(smoke_commit) == 40, "smoke commit_id is not a full SHA")
    _require(
        _git_is_ancestor(repo_root, smoke_commit, current_commit),
        "formal commit is not the smoke commit or its descendant",
    )
    changed_paths = _changed_commit_paths(
        repo_root, smoke_commit, current_commit
    )
    forbidden = [path for path in changed_paths
                 if not _is_pure_evidence_path(path)]
    _require(
        not forbidden,
        "non-evidence files changed after smoke: {}".format(forbidden),
    )
    return {
        "run_id": manifest.get("run_id"),
        "smoke_commit": smoke_commit,
        "formal_commit": current_commit,
        "post_smoke_changed_paths": changed_paths,
        "config_source_sha256": formal_source_sha,
        "selected_epoch": 1,
        "feature_reference_commit": manifest.get("feature_reference_commit"),
        "protocol_signature_sha256": manifest.get(
            "protocol_signature_sha256"
        ),
        "implementation_signature_sha256": manifest.get(
            "implementation_signature_sha256"
        ),
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
    }


def validate_formal_smoke_gate(
        repo_root, records_root, formal_config_path, formal_configuration,
        current_commit, expected_branch, expected_experiment_id,
        expected_experiment_family, feature_compatibility,
        expected_parent_branch=None, expected_parent_commit=None,
        expected_merge_base=None, expected_protocol_signature=None,
        expected_implementation_signature=None,
        expected_dataset_manifest=None):
    """Return matching smoke evidence or reject the formal run fail-closed."""
    repo_root = Path(repo_root)
    records_root = Path(records_root)
    formal_config_path = Path(formal_config_path)
    runs_root = records_root / "runs"
    _require(runs_root.is_dir(), "experiment run evidence directory is missing")
    candidates = []
    for run_dir in runs_root.iterdir():
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = _read_json(manifest_path)
        except SmokeGateError:
            continue
        if manifest.get("experiment_id") == expected_experiment_id:
            candidates.append(run_dir)
    _require(
        candidates,
        "no smoke evidence found for {}".format(expected_experiment_id),
    )
    failures = []
    for run_dir in sorted(candidates, key=lambda path: path.name, reverse=True):
        try:
            return _validate_candidate(
                repo_root, run_dir, formal_config_path, formal_configuration,
                current_commit, expected_branch, expected_experiment_id,
                expected_experiment_family, feature_compatibility,
                expected_parent_branch=expected_parent_branch,
                expected_parent_commit=expected_parent_commit,
                expected_merge_base=expected_merge_base,
                expected_protocol_signature=expected_protocol_signature,
                expected_implementation_signature=(
                    expected_implementation_signature
                ),
                expected_dataset_manifest=expected_dataset_manifest,
            )
        except (SmokeGateError, OSError, ValueError, subprocess.SubprocessError) as error:
            failures.append("{}: {}".format(run_dir.name, error))
    raise SmokeGateError(
        "no matching successful one-epoch smoke passed the formal gate: {}"
        .format("; ".join(failures))
    )
