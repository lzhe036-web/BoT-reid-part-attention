# encoding: utf-8
"""Unified, fail-closed experiment evidence recording.

CSV files are the machine-readable source of truth.  Markdown files and the
generated section in ``EXPERIMENTS.md`` are always derived from those CSVs.
"""

from __future__ import absolute_import

import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import yaml

from utils.reproducibility import (
    derive_data_loader_seed,
    resolved_config_text,
    validate_seed,
    validate_seed_evidence_chain,
)
from utils.multigranular_signature import (
    canonical_multigranular_feature_signature,
)


SCHEMA_VERSION = 5
NOT_RECORDED = "not_recorded"
MISSING_EVIDENCE = "missing_evidence"
NOT_APPLICABLE = "not_applicable"
AUTO_RESULTS_START = "<!-- AUTO-EXPERIMENT-RESULTS:START -->"
AUTO_RESULTS_END = "<!-- AUTO-EXPERIMENT-RESULTS:END -->"
AUTO_RUNS_START = "<!-- AUTO-EXPERIMENT-RUNS:START -->"
AUTO_RUNS_END = "<!-- AUTO-EXPERIMENT-RUNS:END -->"
AUTO_CHECKPOINTS_START = "<!-- AUTO-CHECKPOINT-EVIDENCE:START -->"
AUTO_CHECKPOINTS_END = "<!-- AUTO-CHECKPOINT-EVIDENCE:END -->"
SOFT_LAMBDA_SWEEP_FAMILY = (
    "c2l03_soft_min_alignment_lambda_sweep_tau0p2"
)
WINDOWED_SOFT_ALIGNMENT_FAMILY = (
    "c2l03_windowed_soft_min_alignment_tau0p2_lambda0p05"
)
STRICT_EVIDENCE_FAMILIES = {
    SOFT_LAMBDA_SWEEP_FAMILY,
    WINDOWED_SOFT_ALIGNMENT_FAMILY,
}
STRICT_MANIFEST_REQUIRED_FIELDS = (
    "protocol_signature_sha256",
    "implementation_signature_sha256", "merge_base",
    "commit_tree", "has_upstream", "upstream",
    "config_source_size_bytes", "config_resolved_size_bytes",
    "dataset_manifest_sha256", "commit_time", "commit_parents",
    "git_status_porcelain_raw", "git_staged_diff_empty",
    "git_unstaged_diff_empty", "git_operations_in_progress",
    "git_preflight_checked_at_utc",
)


class EvidenceError(RuntimeError):
    pass


def _strict_manifest_missing_fields(manifest):
    """Return strict fields that are absent or hold invalid sentinels."""
    missing = []
    for field in STRICT_MANIFEST_REQUIRED_FIELDS:
        if field not in manifest:
            missing.append(field)
            continue
        value = manifest[field]
        # Existing upstream sentinels explicitly represent "no upstream".
        if field == "upstream":
            continue
        # Empty porcelain is the recorded proof of a clean Git worktree.
        invalid = ((None, NOT_RECORDED, MISSING_EVIDENCE)
                   if field == "git_status_porcelain_raw"
                   else (None, "", NOT_RECORDED, MISSING_EVIDENCE))
        if value in invalid:
            missing.append(field)
    return missing


def validate_strict_manifest_preflight(manifest):
    """Validate strict evidence shared by lambda and windowed sweeps."""
    missing = _strict_manifest_missing_fields(manifest)
    if missing:
        raise EvidenceError("Strict experiment manifest lacks {}".format(missing))
    if manifest.get("git_preflight_clean") is not True:
        raise EvidenceError("Strict experiment preflight was not clean")
    if manifest.get("git_status_preflight") != []:
        raise EvidenceError("Strict preflight status is not empty")
    if manifest.get("git_status_porcelain_raw") != "":
        raise EvidenceError("Raw preflight porcelain is not empty")
    if manifest.get("git_staged_diff_empty") is not True:
        raise EvidenceError("Staged diff was not empty at preflight")
    if manifest.get("git_unstaged_diff_empty") is not True:
        raise EvidenceError("Unstaged diff was not empty at preflight")
    if manifest.get("git_operations_in_progress") != []:
        raise EvidenceError("Git operation was active at preflight")


MAIN_FIELDS = (
    "schema_version", "run_id", "experiment_id", "experiment_family",
    "run_kind", "method", "method_family", "method_variant", "dataset",
    "branch", "commit", "parent_branch", "parent_commit", "seed", "lambda",
    "cross_camera_positive_lambda",
    "pcc_lambda", "pcc_enabled", "pcc_parts", "pcc_mode",
    "alignment_strategy", "alignment_mode", "alignment_temperature",
    "alignment_window",
    "gating_mode", "gating_temperature",
    "multigranular_feature_signature",
    "multigranular_feature_signature_sha256",
    "feature_reference_commit", "feature_reference_signature_sha256",
    "current_feature_signature_sha256", "feature_compatibility_status",
    "feature_compatibility_evidence_path",
    "feature_compatibility_evidence_size_bytes",
    "feature_compatibility_evidence_sha256",
    "baseline", "margin", "mode", "best_epoch",
    "selected_epoch", "rank1", "rank5", "rank10", "map", "checkpoint",
    "checkpoint_sha256", "runtime_seconds", "gpu", "config",
    "source_config_path", "source_config_sha256", "resolved_config_path",
    "resolved_config_sha256", "log_path", "training_log_size_bytes",
    "log_sha256", "console_log_path", "console_log_size_bytes",
    "console_log_sha256", "artifact_manifest_path",
    "artifact_manifest_size_bytes", "artifact_manifest_sha256",
    "output_dir", "valid_pcc_pair_count",
    "mean_fixed_index_part_distance", "hard_alignment_loss",
    "valid_alignment_pair_count", "mean_hard_path_cost",
    "mean_path_absolute_offset", "soft_alignment_loss",
    "mean_soft_path_cost", "status", "notes",
)
LAMBDA_FIELDS = (
    "run_id", "experiment_id", "method", "dataset", "lambda", "seed",
    "best_epoch", "rank1", "map", "runtime", "checkpoint", "commit",
)
SAME_CAMERA_FIELDS = (
    "run_id", "experiment_id", "variant", "positive_relation", "lambda",
    "rank1", "map", "best_epoch", "seed", "checkpoint", "commit",
)
CAAT_FIELDS = (
    "run_id", "experiment_id", "baseline", "camera_aware_triplet",
    "cross_camera_positive", "same_camera_positive", "hierarchical",
    "weighted", "multi_granularity", "lambda", "rank1", "map", "Params",
    "FLOPs", "runtime", "seed", "commit",
)
DISTANCE_FIELDS = (
    "run_id", "experiment_id", "dataset", "checkpoint",
    "same_id_same_camera_mean", "same_id_same_camera_std",
    "same_id_cross_camera_mean", "same_id_cross_camera_std",
    "different_id_mean", "different_id_std", "cross_camera_gap", "seed",
    "commit",
)
ANCHOR_FIELDS = (
    "run_id", "experiment_id", "dataset", "seed", "total_anchors",
    "valid_cross_camera_anchors", "invalid_cross_camera_anchors",
    "coverage_percent", "cross_camera_positive_count",
    "same_camera_positive_count", "commit",
)
PCC_FIELDS = (
    "schema_version", "run_id", "experiment_id", "run_kind", "baseline",
    "method_family", "method_variant", "pcc_enabled",
    "alignment_strategy", "alignment_temperature", "alignment_window", "gating_mode",
    "gating_temperature", "pcc_parts", "cross_camera_positive_lambda",
    "pcc_lambda", "valid_pcc_pair_count",
    "mean_fixed_index_part_distance", "hard_alignment_loss",
    "valid_alignment_pair_count", "mean_hard_path_cost",
    "mean_path_absolute_offset", "soft_alignment_loss",
    "mean_soft_path_cost", "best_epoch", "rank1", "map", "runtime",
    "seed", "commit",
)
ALIGNMENT_FIELDS = (
    "schema_version", "run_id", "experiment_id", "run_kind", "baseline",
    "method_family", "method_variant", "alignment_mode",
    "alignment_temperature", "alignment_window", "gating_mode", "gating_temperature",
    "multigranular_feature_signature_sha256", "parent_branch",
    "parent_commit", "parts",
    "cross_camera_positive_lambda", "alignment_lambda",
    "valid_alignment_pair_count", "hard_alignment_loss",
    "mean_hard_path_cost", "mean_path_absolute_offset",
    "soft_alignment_loss", "mean_soft_path_cost", "best_epoch",
    "rank1", "map", "runtime", "seed", "commit",
)
SOFT_ALIGNMENT_LAMBDA_FIELDS = (
    "schema_version", "run_id", "experiment_id", "run_kind", "status",
    "alignment_mode", "alignment_temperature", "pcc_lambda",
    "alignment_lambda", "parts", "seed", "rank1", "rank5", "rank10",
    "map", "best_epoch", "runtime", "checkpoint", "checkpoint_sha256",
    "commit", "output_dir",
)
WINDOWED_SOFT_ALIGNMENT_FIELDS = (
    "schema_version", "run_id", "experiment_id", "run_kind", "status",
    "dataset", "method_variant", "alignment_mode", "alignment_window",
    "alignment_temperature", "pcc_lambda", "parts", "seed", "rank1",
    "rank5", "rank10", "map", "best_epoch", "runtime", "checkpoint",
    "checkpoint_sha256", "commit", "output_dir",
)
RUN_FIELDS = (
    "schema_version", "run_id", "experiment_id", "experiment_family",
    "run_kind", "method", "method_family", "method_variant", "dataset",
    "branch", "commit_id", "parent_branch", "parent_commit", "config_file",
    "seed", "lambda",
    "cross_camera_positive_lambda", "pcc_lambda", "pcc_enabled",
    "pcc_parts", "pcc_mode", "alignment_strategy", "alignment_mode",
    "alignment_temperature", "alignment_window", "gating_mode", "gating_temperature",
    "multigranular_feature_signature",
    "multigranular_feature_signature_sha256",
    "feature_reference_commit", "feature_reference_signature_sha256",
    "current_feature_signature_sha256", "feature_compatibility_status",
    "feature_compatibility_evidence_path",
    "feature_compatibility_evidence_size_bytes",
    "feature_compatibility_evidence_sha256", "baseline", "margin",
    "mode", "GPU", "start_time", "end_time", "runtime", "best_epoch",
    "selected_epoch", "Rank-1", "Rank-5", "Rank-10", "mAP", "checkpoint",
    "checkpoint_sha256", "source_config_path", "source_config_sha256",
    "resolved_config_path", "resolved_config_sha256", "log_path",
    "training_log_size_bytes", "log_sha256", "console_log_path",
    "console_log_size_bytes", "console_log_sha256",
    "artifact_manifest_path", "artifact_manifest_size_bytes",
    "artifact_manifest_sha256", "output_dir", "status",
    "valid_pcc_pair_count", "mean_fixed_index_part_distance",
    "hard_alignment_loss", "valid_alignment_pair_count",
    "mean_hard_path_cost", "mean_path_absolute_offset",
    "soft_alignment_loss", "mean_soft_path_cost", "notes",
)
EVIDENCE_FIELDS = (
    "schema_version", "run_id", "run_kind", "artifact_type", "path",
    "size_bytes", "sha256",
)

TABLE_SCHEMAS = {
    "main_results": MAIN_FIELDS,
    "lambda_sensitivity": LAMBDA_FIELDS,
    "same_camera_positive_ablation": SAME_CAMERA_FIELDS,
    "caat_ablation": CAAT_FIELDS,
    "distance_distribution": DISTANCE_FIELDS,
    "anchor_coverage": ANCHOR_FIELDS,
    "pcc_ablation": PCC_FIELDS,
    "alignment_ablation": ALIGNMENT_FIELDS,
    "soft_alignment_lambda_sensitivity": SOFT_ALIGNMENT_LAMBDA_FIELDS,
    "windowed_soft_alignment_sensitivity": WINDOWED_SOFT_ALIGNMENT_FIELDS,
}

PURE_EVIDENCE_PATHS = ("EXPERIMENTS.md", "experiment_records/")


def utc_now():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalized_text(value):
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def atomic_write_text(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("{}.tmp.{}".format(target.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_normalized_text(value))
    os.replace(str(temporary), str(target))
    return target


def json_text(payload):
    return json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"


def atomic_write_json(path, payload):
    return atomic_write_text(path, json_text(payload))


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_protocol_signature(configuration):
    """Hash the semantic experiment protocol, excluding only OUTPUT_DIR."""
    payload = json.loads(json.dumps(configuration))
    payload.pop("OUTPUT_DIR", None)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_implementation_signature(repo_root, revision="HEAD"):
    """Hash the tracked non-evidence tree for smoke/formal equivalence."""
    try:
        output = subprocess.check_output(
            [
                "git", "-C", str(repo_root), "ls-tree", "-r", "--full-tree",
                str(revision),
            ],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace")
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceError(
            "Cannot compute implementation signature: {}".format(error)
        )
    retained = []
    for line in output.splitlines():
        if "\t" not in line:
            raise EvidenceError("Cannot parse Git tree entry")
        _, path = line.split("\t", 1)
        path = normalized_path(path)
        if path == PURE_EVIDENCE_PATHS[0] or path.startswith(
                PURE_EVIDENCE_PATHS[1]):
            continue
        retained.append(line)
    return hashlib.sha256(
        ("\n".join(retained) + "\n").encode("utf-8")
    ).hexdigest()


def normalized_path(path):
    return str(path).replace("\\", "/")


def copy_file_atomic(source, destination):
    source = Path(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("{}.tmp.{}".format(target.name, os.getpid()))
    shutil.copyfile(str(source), str(temporary))
    os.replace(str(temporary), str(target))
    return target


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise EvidenceError("Config is not a YAML mapping: {}".format(path))
    return value


def nested_value(mapping, dotted_path, default=NOT_RECORDED):
    current = mapping
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _first_config_value(mapping, paths, default=NOT_RECORDED):
    for path in paths:
        value = nested_value(mapping, path, NOT_RECORDED)
        if value != NOT_RECORDED:
            return value
    return default


def config_modules(configuration):
    """Infer ablation flags only from resolved configuration leaves."""
    def enabled(paths):
        value = _first_config_value(configuration, paths, False)
        if value in (
                None, "", NOT_RECORDED, NOT_APPLICABLE, MISSING_EVIDENCE,
                False, 0, "false", "False", "off", "no"):
            return False
        return value is True or value == 1 or value in (
            "true", "True", "on", "yes"
        )

    model_name = str(nested_value(configuration, "MODEL.NAME", ""))
    return {
        "baseline": bool(model_name),
        "camera_aware_triplet": enabled(("MODEL.CAMERA_AWARE_TRIPLET",)),
        "cross_camera_positive": enabled(("MODEL.CROSS_CAMERA_POSITIVE_ONLY",)),
        "same_camera_positive": enabled(("MODEL.SAME_CAMERA_POSITIVE_ONLY",)),
        "hierarchical": enabled((
            "MODEL.HIERARCHICAL_CAMERA_AWARE_LOSS",
            "MODEL.HIERARCHICAL_CAMERA_AWARE",
            "MODEL.HIERARCHICAL",
        )),
        "weighted": enabled((
            "MODEL.NORMALIZED_WEIGHTED_LOSS",
            "MODEL.WEIGHTED_LOSS",
            "MODEL.HARD_NEGATIVE_WEIGHTING",
        )),
        "multi_granularity": enabled((
            "MODEL.MULTI_GRANULARITY_PART",
            "MODEL.MULTI_GRANULARITY",
        )),
        "part_correspondence_consistency": enabled((
            "MODEL.PART_CORRESPONDENCE_CONSISTENCY",
        )),
    }


def experiment_identity(configuration):
    modules = config_modules(configuration)
    if modules["same_camera_positive"]:
        method = "Same-camera positive only"
        variant = "same_camera_positive"
        relation = "same_camera"
        loss_lambda = nested_value(
            configuration, "MODEL.SAME_CAMERA_POSITIVE_LAMBDA"
        )
        mode = nested_value(configuration, "MODEL.SAME_CAMERA_POSITIVE_MODE")
    elif modules["camera_aware_triplet"]:
        method = "Camera-aware triplet"
        variant = "camera_aware_triplet"
        relation = "cross_camera_with_negative"
        loss_lambda = nested_value(
            configuration, "MODEL.CAMERA_AWARE_TRIPLET_LAMBDA"
        )
        mode = nested_value(configuration, "MODEL.CAMERA_AWARE_TRIPLET_MODE")
    elif modules["cross_camera_positive"]:
        method = "Cross-camera positive only"
        variant = "cross_camera_positive"
        relation = "cross_camera"
        loss_lambda = nested_value(
            configuration, "MODEL.CROSS_CAMERA_POSITIVE_LAMBDA"
        )
        mode = nested_value(configuration, "MODEL.CROSS_CAMERA_POSITIVE_MODE")
    else:
        method = "Baseline"
        variant = "baseline"
        relation = "none"
        loss_lambda = NOT_RECORDED
        mode = NOT_RECORDED
    pcc_enabled = modules["part_correspondence_consistency"]
    pcc_mode = nested_value(configuration, "MODEL.PCC_MODE")
    if pcc_enabled:
        if pcc_mode == "fixed_index":
            method = "C2-L03 + Fixed-Index Part Correspondence Consistency"
            variant = "fixed_index_pcc"
            method_variant = "fixed_index"
            relation = "same_pid_different_camera_same_index"
        elif pcc_mode == "hard_shortest_path":
            method = "C2-L03 + Hard Shortest-Path Part Alignment"
            variant = "hard_shortest_path"
            method_variant = "hard_shortest_path"
            relation = "same_pid_different_camera_monotonic_path"
        elif pcc_mode == "soft_min":
            method = "C2-L03 + Soft-Min Part Alignment"
            variant = "soft_min"
            method_variant = "soft_min"
            relation = "same_pid_different_camera_soft_monotonic_paths"
        elif pcc_mode == "windowed_soft_min":
            method = "C2-L03 + Windowed Soft-Min Part Alignment"
            variant = "windowed_soft_min"
            method_variant = "windowed_soft_min"
            relation = "same_pid_different_camera_windowed_soft_paths"
        else:
            raise EvidenceError(
                "Unsupported enabled PCC_MODE: {!r}".format(pcc_mode)
            )
        method_family = "part_alignment"
        alignment_mode = pcc_mode
        if pcc_mode in ("soft_min", "windowed_soft_min"):
            from layers.part_correspondence_consistency import (
                validate_softmin_tau,
            )
            alignment_temperature = validate_softmin_tau(
                nested_value(configuration, "MODEL.PCC_SOFTMIN_TAU")
            )
        else:
            alignment_temperature = NOT_APPLICABLE
        if pcc_mode == "windowed_soft_min":
            from layers.part_correspondence_consistency import (
                validate_softmin_window,
            )
            alignment_window = validate_softmin_window(
                nested_value(configuration, "MODEL.PCC_SOFTMIN_WINDOW"),
                int(nested_value(configuration, "MODEL.PCC_PARTS")),
            )
        else:
            alignment_window = NOT_APPLICABLE
        gating_mode = NOT_APPLICABLE
        gating_temperature = NOT_APPLICABLE
    else:
        method_family = variant
        method_variant = variant
        alignment_mode = NOT_APPLICABLE
        alignment_temperature = NOT_APPLICABLE
        alignment_window = NOT_APPLICABLE
        gating_mode = NOT_APPLICABLE
        gating_temperature = NOT_APPLICABLE
    dataset = nested_value(configuration, "DATASETS.NAMES")
    if isinstance(dataset, (list, tuple)):
        dataset = dataset[0] if dataset else NOT_RECORDED
    return {
        "method": method,
        "variant": variant,
        "method_family": method_family,
        "method_variant": method_variant,
        "positive_relation": relation,
        "dataset": str(dataset),
        "lambda": loss_lambda,
        "margin": nested_value(configuration, "SOLVER.MARGIN"),
        "mode": mode,
        "modules": modules,
        "baseline": "C2-L03" if pcc_enabled else NOT_RECORDED,
        "cross_camera_positive_lambda": nested_value(
            configuration, "MODEL.CROSS_CAMERA_POSITIVE_LAMBDA"
        ) if modules["cross_camera_positive"] else NOT_RECORDED,
        "pcc_enabled": pcc_enabled,
        "pcc_parts": nested_value(configuration, "MODEL.PCC_PARTS")
        if pcc_enabled else NOT_RECORDED,
        "pcc_lambda": nested_value(configuration, "MODEL.PCC_LAMBDA")
        if pcc_enabled else NOT_RECORDED,
        "pcc_mode": pcc_mode if pcc_enabled else NOT_RECORDED,
        "alignment_strategy": pcc_mode if pcc_enabled else NOT_RECORDED,
        "alignment_mode": alignment_mode,
        "alignment_temperature": alignment_temperature,
        "alignment_window": alignment_window,
        "gating_mode": gating_mode,
        "gating_temperature": gating_temperature,
    }


def _git_output(repo_root, args):
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root)] + list(args),
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceError("Cannot read Git metadata: {}".format(error))
    return output.decode("utf-8", errors="replace").strip()


def _git_status_entries(repo_root):
    try:
        output = subprocess.check_output(
            [
                "git", "-C", str(repo_root), "status", "--porcelain=v1",
                "--untracked-files=all", "-z",
            ],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceError("Cannot read Git worktree status: {}".format(error))
    entries = []
    for raw_entry in output.split(b"\0"):
        if not raw_entry:
            continue
        entry = raw_entry.decode("utf-8", errors="surrogateescape")
        if len(entry) < 4 or entry[2] != " ":
            raise EvidenceError("Cannot parse Git worktree status entry")
        entries.append((entry[:2], normalized_path(entry[3:])))
    return entries


def _git_status_porcelain_raw(repo_root):
    try:
        output = subprocess.check_output(
            [
                "git", "-C", str(repo_root), "status", "--porcelain=v1",
                "--untracked-files=all",
            ],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceError("Cannot read raw Git worktree status: {}".format(error))
    return output.decode("utf-8", errors="surrogateescape")


def _git_diff_empty(repo_root, cached=False):
    command = ["git", "-C", str(repo_root), "diff", "--quiet"]
    if cached:
        command.insert(-1, "--cached")
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode not in (0, 1):
        raise EvidenceError(
            "Cannot inspect {} Git diff".format(
                "staged" if cached else "unstaged"
            )
        )
    return completed.returncode == 0


def _git_operations_in_progress(repo_root):
    operations = []
    for operation, git_path in {
            "merge": "MERGE_HEAD", "cherry_pick": "CHERRY_PICK_HEAD",
            "revert": "REVERT_HEAD", "rebase_merge": "rebase-merge",
            "rebase_apply": "rebase-apply",
    }.items():
        resolved = _git_output(repo_root, ["rev-parse", "--git-path", git_path])
        candidate = Path(resolved)
        if not candidate.is_absolute():
            candidate = Path(repo_root) / candidate
        if candidate.exists():
            operations.append(operation)
    return operations


def git_metadata(repo_root):
    commit = _git_output(repo_root, ["rev-parse", "HEAD"])
    branch = _git_output(repo_root, ["branch", "--show-current"])
    status_entries = _git_status_entries(repo_root)
    status_porcelain_raw = _git_status_porcelain_raw(repo_root)
    staged_diff_empty = _git_diff_empty(repo_root, cached=True)
    unstaged_diff_empty = _git_diff_empty(repo_root, cached=False)
    operations_in_progress = _git_operations_in_progress(repo_root)
    dirty = bool(status_entries)
    if not re.match(r"^[0-9a-fA-F]{40}$", commit):
        raise EvidenceError("Git commit is not a full SHA: {}".format(commit))
    if not branch:
        raise EvidenceError("Detached HEAD is not allowed for formal training")
    upstream_process = subprocess.run(
        [
            "git", "-C", str(repo_root), "rev-parse", "--abbrev-ref",
            "--symbolic-full-name", "@{upstream}",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    upstream = (
        upstream_process.stdout.strip()
        if upstream_process.returncode == 0 else NOT_RECORDED
    )
    return {
        "commit": commit.lower(),
        "branch": branch,
        "dirty": dirty,
        "status_porcelain": [
            "{} {}".format(status, path) for status, path in status_entries
        ],
        "status_porcelain_raw": status_porcelain_raw,
        "staged_diff_empty": staged_diff_empty,
        "unstaged_diff_empty": unstaged_diff_empty,
        "operations_in_progress": operations_in_progress,
        "preflight_checked_at_utc": utc_now(),
        "commit_time": _git_output(
            repo_root, ["show", "-s", "--format=%cI", commit]
        ),
        "commit_parents": _git_output(
            repo_root, ["show", "-s", "--format=%P", commit]
        ).split(),
        "tree": _git_output(repo_root, ["rev-parse", "HEAD^{tree}"]).lower(),
        "has_upstream": upstream != NOT_RECORDED,
        "upstream": upstream,
    }


def validate_git_preflight(repo_root, expected_branch, expected_commit=None):
    metadata = git_metadata(repo_root)
    if (metadata["dirty"] or metadata["status_porcelain_raw"]
            or not metadata["staged_diff_empty"]
            or not metadata["unstaged_diff_empty"]):
        raise EvidenceError("Formal training requires a clean Git worktree")
    if metadata["operations_in_progress"]:
        raise EvidenceError(
            "Formal training forbids Git operations in progress: {}".format(
                metadata["operations_in_progress"]
            )
        )
    if metadata["branch"] != expected_branch:
        raise EvidenceError(
            "Formal branch mismatch: expected {}, got {}".format(
                expected_branch, metadata["branch"]
            )
        )
    if expected_commit and metadata["commit"] != expected_commit.lower():
        raise EvidenceError(
            "Formal commit mismatch: expected {}, got {}".format(
                expected_commit, metadata["commit"]
            )
        )
    return metadata


def validate_parent_lineage(repo_root, parent_branch, parent_commit,
                            child_commit="HEAD"):
    """Fail closed unless the named parent tip is the child's exact base."""
    if parent_branch in (None, "", NOT_RECORDED, MISSING_EVIDENCE):
        raise EvidenceError("Parent branch evidence is missing")
    if not re.match(r"^[0-9a-fA-F]{40}$", str(parent_commit)):
        raise EvidenceError("Parent commit must be a full SHA")
    resolved_parent = _git_output(
        repo_root, ["rev-parse", str(parent_branch)]
    ).lower()
    expected_parent = str(parent_commit).lower()
    if resolved_parent != expected_parent:
        raise EvidenceError(
            "Parent branch tip differs from recorded parent commit"
        )
    merge_base = _git_output(
        repo_root, ["merge-base", str(parent_branch), str(child_commit)]
    ).lower()
    if merge_base != expected_parent:
        raise EvidenceError(
            "Child is not directly based on the recorded parent commit"
        )
    return {
        "parent_branch": str(parent_branch),
        "parent_commit": expected_parent,
        "merge_base": merge_base,
    }


def validate_git_runtime_state(repo_root, run_dir, expected_branch=None,
                               expected_commit=None):
    """Allow only recorder-owned evidence changes during a run."""
    repo_root = Path(repo_root).resolve()
    run_dir = Path(run_dir).resolve()
    try:
        allowed_relative = normalized_path(run_dir.relative_to(repo_root))
    except ValueError:
        raise EvidenceError("Controlled run_dir must be inside the Git worktree")
    records_relative = normalized_path(
        run_dir.parent.parent.relative_to(repo_root)
    )
    controlled_files = {
        "EXPERIMENTS.md",
        records_relative + "/runs.csv",
        records_relative + "/evidence_manifest.tsv",
    }
    controlled_table_prefix = records_relative + "/tables/"
    unexpected = []
    for status_code, path in _git_status_entries(repo_root):
        under_current_run = (
            path == allowed_relative or path.startswith(allowed_relative + "/")
        )
        controlled_table = (
            path.startswith(controlled_table_prefix)
            and (path.endswith(".csv") or path.endswith(".md"))
        )
        controlled_registry = path in controlled_files or controlled_table
        if not (under_current_run or controlled_registry):
            unexpected.append("{} {}".format(status_code, path))
    if unexpected:
        raise EvidenceError(
            "Git worktree contains changes outside controlled run evidence: {}"
            .format(", ".join(unexpected))
        )
    metadata = git_metadata(repo_root)
    if expected_branch and metadata["branch"] != expected_branch:
        raise EvidenceError("Branch changed after formal preflight")
    if expected_commit and metadata["commit"] != expected_commit.lower():
        raise EvidenceError("Commit changed after formal preflight")
    metadata["controlled_evidence_dir"] = allowed_relative
    metadata["controlled_evidence_only"] = True
    metadata["controlled_registry_files"] = sorted(controlled_files)
    return metadata


def collect_environment(run_dir, repo_root, training_pythonhashseed=None,
                        expected_branch=None, expected_commit=None):
    import torch

    run_dir = Path(run_dir)
    try:
        packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace")
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceError("Cannot collect Python packages: {}".format(error))
    packages_path = atomic_write_text(
        run_dir / "environment_packages.txt", packages.rstrip() + "\n"
    )
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(properties.total_memory),
            })
    driver = NOT_RECORDED
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace").splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        pass
    metadata = validate_git_runtime_state(
        repo_root,
        run_dir,
        expected_branch=expected_branch,
        expected_commit=expected_commit,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None) or NOT_RECORDED,
        "cudnn_version": torch.backends.cudnn.version() or NOT_RECORDED,
        "nvidia_driver": driver,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "CUDA_VISIBLE_DEVICES": os.environ.get(
            "CUDA_VISIBLE_DEVICES", NOT_RECORDED
        ),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", NOT_RECORDED),
        "training_subprocess_PYTHONHASHSEED": (
            str(training_pythonhashseed)
            if training_pythonhashseed is not None else NOT_RECORDED
        ),
        "git_branch": metadata["branch"],
        "git_commit": metadata["commit"],
        "git_commit_time": metadata["commit_time"],
        "git_commit_parents": metadata["commit_parents"],
        "git_status_porcelain_raw": metadata["status_porcelain_raw"],
        "git_staged_diff_empty": metadata["staged_diff_empty"],
        "git_unstaged_diff_empty": metadata["unstaged_diff_empty"],
        "git_operations_in_progress": metadata["operations_in_progress"],
        "git_controlled_evidence_dir": metadata["controlled_evidence_dir"],
        "git_controlled_evidence_only": metadata["controlled_evidence_only"],
        "environment_packages_path": normalized_path(packages_path),
        "environment_packages_sha256": sha256_file(packages_path),
    }


def build_dataset_manifest(dataset, configuration, data_root):
    split_payload = {}
    digest = hashlib.sha256()
    for split_name in ("train", "query", "gallery"):
        samples = sorted(
            getattr(dataset, split_name), key=lambda item: normalized_path(item[0])
        )
        pids = set()
        camids = set()
        split_digest = hashlib.sha256()
        for image_path, pid, camid in samples:
            absolute = Path(image_path).resolve()
            try:
                relative = absolute.relative_to(Path(data_root).resolve())
                path_value = normalized_path(relative)
            except ValueError:
                path_value = normalized_path(absolute)
            size = absolute.stat().st_size
            line = "{}\t{}\t{}\t{}\n".format(path_value, pid, camid, size)
            split_digest.update(line.encode("utf-8"))
            digest.update((split_name + "\t" + line).encode("utf-8"))
            pids.add(int(pid))
            camids.add(int(camid))
        split_payload[split_name] = {
            "image_count": len(samples),
            "pid_count": len(pids),
            "camera_count": len(camids),
            "manifest_sha256": split_digest.hexdigest(),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": utc_now(),
        "dataset": str(nested_value(configuration, "DATASETS.NAMES")),
        "data_root": normalized_path(Path(data_root).resolve()),
        "splits": split_payload,
        "dataset_manifest_sha256": digest.hexdigest(),
        "hash_basis": "sorted relative path, pid, camid, size_bytes",
        "sampler": nested_value(configuration, "DATALOADER.SAMPLER"),
        "batch_size": nested_value(configuration, "SOLVER.IMS_PER_BATCH"),
        "num_instance": nested_value(configuration, "DATALOADER.NUM_INSTANCE"),
        "num_workers": nested_value(configuration, "DATALOADER.NUM_WORKERS"),
    }


def generate_run_id(experiment_id, commit, seed, when=None):
    when = when or dt.datetime.utcnow()
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment_id).strip("-")
    return "{}-{}-{}-s{}".format(
        safe_id,
        when.strftime("%Y%m%dT%H%M%S%fZ"),
        str(commit)[:12],
        seed if seed != NOT_RECORDED else NOT_RECORDED,
    )


def initialize_run(records_root, experiment_id, experiment_family, run_id,
                   config_file, resolved_cfg, output_dir, git_info, notes,
                   command, expected_branch, method=None,
                   baseline_method=NOT_RECORDED,
                   baseline_commit=NOT_RECORDED, run_kind="formal",
                   parent_branch=NOT_RECORDED,
                   parent_commit=NOT_RECORDED,
                   feature_compatibility=None, experiments_path=None,
                   protocol_signature_sha256=None,
                   implementation_signature_sha256=None,
                   dataset_manifest=None):
    if run_kind not in ("formal", "smoke"):
        raise EvidenceError("run_kind must be 'formal' or 'smoke'")
    records_root = ensure_record_layout(records_root)
    run_dir = records_root / "runs" / run_id
    if run_dir.exists():
        raise EvidenceError("run_id already exists: {}".format(run_id))
    run_dir.mkdir(parents=True)
    source_copy = copy_file_atomic(config_file, run_dir / "config_source.yml")
    resolved_copy = atomic_write_text(
        run_dir / "config_resolved.yml", resolved_config_text(resolved_cfg)
    )
    configuration = load_yaml(resolved_copy)
    identity = experiment_identity(configuration)
    feature_signature, feature_signature_sha256 = (
        canonical_multigranular_feature_signature(configuration)
    )
    seed_value = nested_value(configuration, "SEED")
    seed = validate_seed(seed_value) if seed_value != NOT_RECORDED else NOT_RECORDED
    feature_path = run_dir / "feature_compatibility.json"
    feature_reference_commit = NOT_RECORDED
    feature_reference_signature = NOT_RECORDED
    current_feature_signature = NOT_RECORDED
    feature_status = NOT_RECORDED
    feature_path_value = NOT_RECORDED
    feature_size = NOT_RECORDED
    feature_sha256 = NOT_RECORDED
    if feature_compatibility is not None:
        feature_reference_commit = feature_compatibility.get(
            "feature_reference_commit", MISSING_EVIDENCE
        )
        feature_reference_signature = feature_compatibility.get(
            "feature_reference_signature_sha256", MISSING_EVIDENCE
        )
        current_feature_signature = feature_compatibility.get(
            "current_feature_signature_sha256", MISSING_EVIDENCE
        )
        feature_status = feature_compatibility.get(
            "feature_compatibility_status", MISSING_EVIDENCE
        )
        if feature_status != "compatible":
            raise EvidenceError(
                "Shared multigranular features are not compatible: {}".format(
                    feature_compatibility.get("mismatched_components", [])
                )
            )
        atomic_write_json(feature_path, feature_compatibility)
        feature_path_value = normalized_path(feature_path.resolve())
        feature_size = feature_path.stat().st_size
        feature_sha256 = sha256_file(feature_path)
    elif identity["alignment_mode"] in ("soft_min", "windowed_soft_min"):
        raise EvidenceError(
            "Soft-Min alignment runs require parent-bound feature compatibility evidence"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "experiment_family": experiment_family,
        "run_kind": run_kind,
        "method": method or identity["method"],
        "method_family": identity["method_family"],
        "method_variant": identity["method_variant"],
        "baseline_method": baseline_method,
        "baseline_commit": baseline_commit,
        "dataset": identity["dataset"],
        "branch": git_info["branch"],
        "commit_id": git_info["commit"],
        "commit_tree": git_info.get("tree", NOT_RECORDED),
        "git_preflight_clean": not git_info.get("dirty", True),
        "git_status_preflight": list(git_info.get("status_porcelain", [])),
        "git_status_porcelain_raw": git_info.get(
            "status_porcelain_raw", MISSING_EVIDENCE
        ),
        "git_staged_diff_empty": git_info.get(
            "staged_diff_empty", MISSING_EVIDENCE
        ),
        "git_unstaged_diff_empty": git_info.get(
            "unstaged_diff_empty", MISSING_EVIDENCE
        ),
        "git_operations_in_progress": git_info.get(
            "operations_in_progress", [MISSING_EVIDENCE]
        ),
        "git_preflight_checked_at_utc": git_info.get(
            "preflight_checked_at_utc", NOT_RECORDED
        ),
        "commit_time": git_info.get("commit_time", NOT_RECORDED),
        "commit_parents": git_info.get("commit_parents", NOT_RECORDED),
        "has_upstream": bool(git_info.get("has_upstream", False)),
        "upstream": git_info.get("upstream", NOT_RECORDED),
        "parent_branch": parent_branch,
        "parent_commit": parent_commit,
        "merge_base": git_info.get("merge_base", NOT_RECORDED),
        "expected_branch": expected_branch,
        "config_file": normalized_path(Path(config_file).resolve()),
        "config_source": "config_source.yml",
        "config_source_size_bytes": source_copy.stat().st_size,
        "config_source_sha256": sha256_file(source_copy),
        "config_resolved": "config_resolved.yml",
        "config_resolved_size_bytes": resolved_copy.stat().st_size,
        "config_resolved_sha256": sha256_file(resolved_copy),
        "protocol_signature_sha256": (
            protocol_signature_sha256
            or config_protocol_signature(configuration)
        ),
        "implementation_signature_sha256": (
            implementation_signature_sha256 or NOT_RECORDED
        ),
        "dataset_manifest_sha256": (
            (dataset_manifest or {}).get(
                "dataset_manifest_sha256", NOT_RECORDED
            )
        ),
        "seed": seed,
        "lambda": identity["lambda"],
        "cross_camera_positive_lambda": identity["cross_camera_positive_lambda"],
        "pcc_enabled": identity["pcc_enabled"],
        "pcc_parts": identity["pcc_parts"],
        "pcc_lambda": identity["pcc_lambda"],
        "pcc_mode": identity["pcc_mode"],
        "alignment_strategy": identity["alignment_strategy"],
        "alignment_mode": identity["alignment_mode"],
        "alignment_temperature": identity["alignment_temperature"],
        "alignment_window": identity["alignment_window"],
        "gating_mode": identity["gating_mode"],
        "gating_temperature": identity["gating_temperature"],
        "multigranular_feature_signature": feature_signature,
        "multigranular_feature_signature_sha256": (
            feature_signature_sha256
        ),
        "feature_reference_commit": feature_reference_commit,
        "feature_reference_signature_sha256": feature_reference_signature,
        "current_feature_signature_sha256": current_feature_signature,
        "feature_compatibility_status": feature_status,
        "feature_compatibility_evidence_path": feature_path_value,
        "feature_compatibility_evidence_size_bytes": feature_size,
        "feature_compatibility_evidence_sha256": feature_sha256,
        "baseline": identity["baseline"],
        "margin": identity["margin"],
        "mode": identity["mode"],
        "modules": identity["modules"],
        "required_global_iteration_source": (
            "ignite_engine_epoch_evidence"
        ),
        "output_dir": normalized_path(Path(output_dir).resolve()),
        "training_log_path": normalized_path(
            (Path(output_dir).resolve() / "log.txt")
        ),
        "training_log_sha256": NOT_RECORDED,
        "console_log_path": normalized_path(
            (run_dir / "console.log").resolve()
        ),
        "console_log_sha256": NOT_RECORDED,
        "console_log_streams": "stdout_stderr_combined_in_process_order",
        "console_log_encoding": "utf-8",
        "console_log_decode_errors": "replace",
        "console_log_flush_policy": "every_stream_chunk",
        "artifact_manifest_path": normalized_path(
            (run_dir / "artifact_hashes.tsv").resolve()
        ),
        "artifact_manifest_sha256": NOT_RECORDED,
        "experiments_path": normalized_path(Path(
            experiments_path or records_root.parent / "EXPERIMENTS.md"
        ).resolve()),
        "start_time": utc_now(),
        "command": list(command),
        "notes": notes,
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    status_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "phase": "initialized",
        "training_exit_code": NOT_RECORDED,
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(run_dir / "run_status.json", status_payload)
    _register_run_state(run_dir, status_payload)
    try:
        update_experiments_markdown(
            experiments_path or records_root.parent / "EXPERIMENTS.md",
            records_root,
        )
    except BaseException as error:
        status_payload.update({
            "status": "incomplete",
            "phase": "recording",
            "error_type": type(error).__name__,
            "error": str(error),
            "updated_at_utc": utc_now(),
        })
        atomic_write_json(run_dir / "run_status.json", status_payload)
        _register_run_state(
            run_dir, status_payload, write_artifact_manifest=True
        )
        raise
    return run_dir, manifest


def record_training_exit(run_dir, exit_code, runtime_seconds, end_time=None):
    run_dir = Path(run_dir)
    status = read_json(run_dir / "run_status.json")
    status.update({
        "training_exit_code": int(exit_code),
        "training_runtime_seconds": float(runtime_seconds),
        "training_end_time": end_time or utc_now(),
        "status": "training_complete" if int(exit_code) == 0 else "failed",
        "phase": "awaiting_finalization" if int(exit_code) == 0 else "training",
        "updated_at_utc": utc_now(),
    })
    atomic_write_json(run_dir / "run_status.json", status)
    _register_run_state(run_dir, status)
    update_experiments_markdown(
        _experiments_path_for_run(run_dir), Path(run_dir).parent.parent
    )
    return status


def record_console_log_evidence(run_dir):
    """Hash a closed, non-empty console tee and refresh run evidence."""
    run_dir = Path(run_dir)
    console_path = run_dir / "console.log"
    if not console_path.is_file() or console_path.stat().st_size <= 0:
        raise EvidenceError("Console log is missing or empty")
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    manifest["console_log_path"] = normalized_path(console_path.resolve())
    manifest["console_log_size_bytes"] = console_path.stat().st_size
    manifest["console_log_sha256"] = sha256_file(console_path)
    atomic_write_json(manifest_path, manifest)
    status = read_json(run_dir / "run_status.json")
    _register_run_state(run_dir, status)
    update_experiments_markdown(
        _experiments_path_for_run(run_dir), run_dir.parent.parent
    )
    return {
        "path": manifest["console_log_path"],
        "size_bytes": manifest["console_log_size_bytes"],
        "sha256": manifest["console_log_sha256"],
    }


def record_run_failure(run_dir, error, status="incomplete"):
    run_dir = Path(run_dir)
    path = run_dir / "run_status.json"
    payload = read_json(path) if path.is_file() else {"schema_version": SCHEMA_VERSION}
    payload.update({
        "status": status,
        "phase": (
            "interrupted" if status == "interrupted"
            else "failed" if status == "failed" else "finalization"
        ),
        "error_type": type(error).__name__,
        "error": str(error),
        "updated_at_utc": utc_now(),
    })
    atomic_write_json(path, payload)
    _register_run_state(run_dir, payload, write_artifact_manifest=True)
    update_experiments_markdown(
        _experiments_path_for_run(run_dir), Path(run_dir).parent.parent
    )
    return payload


def _experiments_path_for_run(run_dir):
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        value = read_json(manifest_path).get("experiments_path")
        if value not in (None, "", NOT_RECORDED, MISSING_EVIDENCE):
            return Path(value)
    return run_dir.parent.parent.parent / "EXPERIMENTS.md"


def _selected_checkpoint_from_manifest(run_dir):
    path = Path(run_dir) / "checkpoint_manifest.tsv"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [
        row for row in rows
        if str(row.get("selected", "")).lower() == "true"
    ]
    return selected[0] if len(selected) == 1 else None


def _artifact_type_for_path(path, selected_checkpoint=None):
    path = Path(path)
    if path.name == "console.log":
        return "console_log"
    if path.name == "log.txt":
        return "training_log"
    if path.name == "config_source.yml":
        return "source_config"
    if path.name == "config_resolved.yml":
        return "resolved_config"
    if path.name == "feature_compatibility.json":
        return "feature_compatibility"
    if path.name == "artifact_hashes.tsv":
        return "artifact_manifest"
    if path.suffix == ".pt":
        if selected_checkpoint and normalized_path(path.resolve()) == normalized_path(
                Path(selected_checkpoint).resolve()):
            return "selected_checkpoint"
        return "checkpoint"
    return path.stem


def _run_row_from_manifest(run_dir, manifest, status_payload):
    run_dir = Path(run_dir)
    selected = _selected_checkpoint_from_manifest(run_dir)
    metrics_path = run_dir / "metrics_summary.json"
    metrics = read_json(metrics_path) if metrics_path.is_file() else {}
    environment_path = run_dir / "environment.json"
    environment = read_json(environment_path) if environment_path.is_file() else {}
    console_path = Path(manifest.get(
        "console_log_path", run_dir / "console.log"
    ))
    output_value = manifest.get("output_dir", NOT_RECORDED)
    training_path = Path(manifest.get(
        "training_log_path",
        Path(output_value) / "log.txt" if output_value not in (
            None, "", NOT_RECORDED, MISSING_EVIDENCE
        ) else NOT_RECORDED,
    ))
    artifact_path = Path(manifest.get(
        "artifact_manifest_path", run_dir / "artifact_hashes.tsv"
    ))
    source_path = run_dir / "config_source.yml"
    resolved_path = run_dir / "config_resolved.yml"
    actual_feature_path = run_dir / "feature_compatibility.json"
    feature_path = (
        actual_feature_path if actual_feature_path.is_file()
        else Path(manifest.get(
            "feature_compatibility_evidence_path", NOT_RECORDED
        ))
    )

    def evidence(path, manifest_sha_key=None):
        if str(path) in ("", NOT_RECORDED, MISSING_EVIDENCE):
            return NOT_RECORDED, NOT_RECORDED
        value = normalized_path(path.resolve())
        if path.is_file():
            return value, sha256_file(path)
        recorded = manifest.get(manifest_sha_key, NOT_RECORDED) if manifest_sha_key else NOT_RECORDED
        return value, recorded

    console_value, console_sha = evidence(console_path, "console_log_sha256")
    training_value, training_sha = evidence(training_path, "training_log_sha256")
    artifact_value, artifact_sha = evidence(
        artifact_path, "artifact_manifest_sha256"
    )
    feature_value, feature_sha = evidence(
        feature_path, "feature_compatibility_evidence_sha256"
    )
    source_value, source_sha = evidence(source_path, "config_source_sha256")
    resolved_value, resolved_sha = evidence(
        resolved_path, "config_resolved_sha256"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest.get("run_id", run_dir.name),
        "experiment_id": manifest.get("experiment_id", NOT_RECORDED),
        "experiment_family": manifest.get("experiment_family", NOT_RECORDED),
        "run_kind": manifest.get("run_kind", NOT_RECORDED),
        "method": manifest.get("method", NOT_RECORDED),
        "method_family": manifest.get("method_family", NOT_RECORDED),
        "method_variant": manifest.get("method_variant", NOT_RECORDED),
        "dataset": manifest.get("dataset", NOT_RECORDED),
        "branch": manifest.get("branch", NOT_RECORDED),
        "commit_id": manifest.get("commit_id", NOT_RECORDED),
        "parent_branch": manifest.get("parent_branch", NOT_RECORDED),
        "parent_commit": manifest.get("parent_commit", NOT_RECORDED),
        "config_file": manifest.get("config_file", NOT_RECORDED),
        "source_config_path": source_value,
        "source_config_sha256": source_sha,
        "resolved_config_path": resolved_value,
        "resolved_config_sha256": resolved_sha,
        "seed": manifest.get("seed", NOT_RECORDED),
        "lambda": manifest.get("lambda", NOT_RECORDED),
        "cross_camera_positive_lambda": manifest.get(
            "cross_camera_positive_lambda", NOT_RECORDED
        ),
        "pcc_lambda": manifest.get("pcc_lambda", NOT_RECORDED),
        "pcc_enabled": manifest.get("pcc_enabled", NOT_RECORDED),
        "pcc_parts": manifest.get("pcc_parts", NOT_RECORDED),
        "pcc_mode": manifest.get("pcc_mode", NOT_RECORDED),
        "alignment_strategy": manifest.get("alignment_strategy", NOT_RECORDED),
        "alignment_mode": manifest.get("alignment_mode", NOT_RECORDED),
        "alignment_temperature": manifest.get("alignment_temperature", NOT_RECORDED),
        "alignment_window": manifest.get("alignment_window", NOT_RECORDED),
        "gating_mode": manifest.get("gating_mode", NOT_RECORDED),
        "gating_temperature": manifest.get("gating_temperature", NOT_RECORDED),
        "multigranular_feature_signature": manifest.get(
            "multigranular_feature_signature", NOT_RECORDED
        ),
        "multigranular_feature_signature_sha256": manifest.get(
            "multigranular_feature_signature_sha256", NOT_RECORDED
        ),
        "feature_reference_commit": manifest.get(
            "feature_reference_commit", NOT_RECORDED
        ),
        "feature_reference_signature_sha256": manifest.get(
            "feature_reference_signature_sha256", NOT_RECORDED
        ),
        "current_feature_signature_sha256": manifest.get(
            "current_feature_signature_sha256", NOT_RECORDED
        ),
        "feature_compatibility_status": manifest.get(
            "feature_compatibility_status", NOT_RECORDED
        ),
        "feature_compatibility_evidence_path": feature_value,
        "feature_compatibility_evidence_size_bytes": (
            feature_path.stat().st_size if feature_path.is_file()
            else manifest.get(
                "feature_compatibility_evidence_size_bytes", NOT_RECORDED
            )
        ),
        "feature_compatibility_evidence_sha256": feature_sha,
        "baseline": manifest.get("baseline", NOT_RECORDED),
        "margin": manifest.get("margin", NOT_RECORDED),
        "mode": manifest.get("mode", NOT_RECORDED),
        "GPU": _gpu_label(environment),
        "start_time": manifest.get("start_time", NOT_RECORDED),
        "end_time": status_payload.get(
            "training_end_time", status_payload.get("updated_at_utc", NOT_RECORDED)
        ),
        "runtime": status_payload.get("training_runtime_seconds", NOT_RECORDED),
        "best_epoch": metrics.get("best_epoch", NOT_RECORDED),
        "selected_epoch": metrics.get("selected_epoch", NOT_RECORDED),
        "Rank-1": metrics.get("rank1_percent", NOT_RECORDED),
        "Rank-5": metrics.get("rank5_percent", NOT_RECORDED),
        "Rank-10": metrics.get("rank10_percent", NOT_RECORDED),
        "mAP": metrics.get("map_percent", NOT_RECORDED),
        "checkpoint": (
            selected.get("path", NOT_RECORDED) if selected else NOT_RECORDED
        ),
        "checkpoint_sha256": (
            selected.get("sha256", NOT_RECORDED) if selected else NOT_RECORDED
        ),
        "log_path": training_value,
        "training_log_size_bytes": (
            training_path.stat().st_size if training_path.is_file()
            else NOT_RECORDED
        ),
        "log_sha256": training_sha,
        "console_log_path": console_value,
        "console_log_size_bytes": (
            console_path.stat().st_size if console_path.is_file()
            else manifest.get("console_log_size_bytes", NOT_RECORDED)
        ),
        "console_log_sha256": console_sha,
        "artifact_manifest_path": artifact_value,
        "artifact_manifest_size_bytes": (
            artifact_path.stat().st_size if artifact_path.is_file()
            else NOT_RECORDED
        ),
        "artifact_manifest_sha256": artifact_sha,
        "output_dir": manifest.get("output_dir", NOT_RECORDED),
        "status": status_payload.get("status", NOT_RECORDED),
        "valid_pcc_pair_count": metrics.get("valid_pcc_pair_count", NOT_RECORDED),
        "mean_fixed_index_part_distance": metrics.get(
            "mean_fixed_index_part_distance", NOT_RECORDED
        ),
        "hard_alignment_loss": metrics.get("hard_alignment_loss", NOT_RECORDED),
        "valid_alignment_pair_count": metrics.get(
            "valid_alignment_pair_count", NOT_RECORDED
        ),
        "mean_hard_path_cost": metrics.get("mean_hard_path_cost", NOT_RECORDED),
        "mean_path_absolute_offset": metrics.get(
            "mean_path_absolute_offset", NOT_RECORDED
        ),
        "soft_alignment_loss": metrics.get("soft_alignment_loss", NOT_RECORDED),
        "mean_soft_path_cost": metrics.get("mean_soft_path_cost", NOT_RECORDED),
        "notes": manifest.get("notes", NOT_RECORDED),
    }


def _write_partial_artifact_manifest(run_dir, manifest):
    run_dir = Path(run_dir)
    paths = [
        path for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name not in (
            "artifact_hashes.tsv", "run_manifest.json"
        )
    ]
    output_value = manifest.get("output_dir", NOT_RECORDED)
    output_dir = Path(output_value) if output_value not in (
        None, "", NOT_RECORDED, MISSING_EVIDENCE
    ) else None
    if output_dir is not None and output_dir.is_dir():
        log_path = output_dir / "log.txt"
        if log_path.is_file():
            paths.append(log_path)
        paths.extend(sorted(output_dir.glob("*.pt")))
    selected = _selected_checkpoint_from_manifest(run_dir)
    selected_path = selected.get("path") if selected else None
    seen = set()
    rows = []
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.append({
            "artifact_type": _artifact_type_for_path(path, selected_path),
            "path": normalized_path(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    write_tsv(
        run_dir / "artifact_hashes.tsv",
        ("artifact_type", "path", "size_bytes", "sha256"), rows,
    )
    return rows


def _register_run_state(run_dir, status_payload,
                        write_artifact_manifest=False):
    """Register any run state without ever touching formal result tables."""
    run_dir = Path(run_dir)
    if run_dir.parent.name != "runs":
        raise EvidenceError(
            "Failed run evidence directory is outside the run registry"
        )
    records_root = ensure_record_layout(run_dir.parent.parent)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise EvidenceError("Run manifest is missing")
    manifest = read_json(manifest_path)
    console_path = run_dir / "console.log"
    if console_path.is_file():
        manifest.update({
            "console_log_path": normalized_path(console_path.resolve()),
            "console_log_size_bytes": console_path.stat().st_size,
            "console_log_sha256": sha256_file(console_path),
        })
    output_value = manifest.get("output_dir", NOT_RECORDED)
    if output_value not in (None, "", NOT_RECORDED, MISSING_EVIDENCE):
        training_path = Path(output_value) / "log.txt"
        if training_path.is_file():
            manifest.update({
                "training_log_path": normalized_path(training_path.resolve()),
                "training_log_size_bytes": training_path.stat().st_size,
                "training_log_sha256": sha256_file(training_path),
            })
    atomic_write_json(manifest_path, manifest)
    if write_artifact_manifest:
        _write_partial_artifact_manifest(run_dir, manifest)
        artifact_path = run_dir / "artifact_hashes.tsv"
        manifest["artifact_manifest_path"] = normalized_path(
            artifact_path.resolve()
        )
        manifest["artifact_manifest_size_bytes"] = artifact_path.stat().st_size
        manifest["artifact_manifest_sha256"] = sha256_file(artifact_path)
        atomic_write_json(manifest_path, manifest)
    row = _run_row_from_manifest(run_dir, manifest, status_payload)
    upsert_csv(records_root / "runs.csv", RUN_FIELDS, row)
    evidence_paths = [
        path for path in sorted(run_dir.iterdir()) if path.is_file()
    ]
    output_value = manifest.get("output_dir", NOT_RECORDED)
    output_dir = Path(output_value) if output_value not in (
        None, "", NOT_RECORDED, MISSING_EVIDENCE
    ) else None
    if output_dir is not None and output_dir.is_dir():
        log_path = output_dir / "log.txt"
        if log_path.is_file():
            evidence_paths.append(log_path)
        evidence_paths.extend(sorted(output_dir.glob("*.pt")))
    seen = set()
    selected = _selected_checkpoint_from_manifest(run_dir)
    selected_path = selected.get("path") if selected else None
    for evidence_path in evidence_paths:
        resolved = str(evidence_path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        evidence_row = {
            "schema_version": SCHEMA_VERSION,
            "run_id": row["run_id"],
            "run_kind": row["run_kind"],
            "artifact_type": _artifact_type_for_path(
                evidence_path, selected_path
            ),
            "path": normalized_path(evidence_path.resolve()),
            "size_bytes": evidence_path.stat().st_size,
            "sha256": sha256_file(evidence_path),
        }
        upsert_tsv(
            records_root / "evidence_manifest.tsv",
            EVIDENCE_FIELDS,
            evidence_row,
            key_fields=("run_id", "path"),
        )
    return row


def parse_training_log(log_path):
    path = Path(log_path)
    if not path.is_file():
        raise EvidenceError("Training log is missing: {}".format(path))
    timestamps = []
    validations = []
    current = None
    iterations_per_epoch_values = set()
    pcc_epoch_summaries = []
    hard_alignment_epoch_summaries = []
    soft_alignment_epoch_summaries = []
    windowed_soft_alignment_epoch_summaries = []
    epoch_evidence = {}
    required_training_fields = {
        "loss_total": False,
        "loss_id": False,
        "loss_triplet": False,
        "learning_rate": False,
    }
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    for line in raw_text.splitlines():
        timestamp = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if timestamp:
            try:
                timestamps.append(dt.datetime.strptime(
                    timestamp.group(1), "%Y-%m-%d %H:%M:%S"
                ))
            except ValueError:
                pass
        required_training_fields["loss_total"] |= "loss_total:" in line
        required_training_fields["loss_id"] |= "loss_id:" in line
        required_training_fields["loss_triplet"] |= "loss_triplet:" in line
        required_training_fields["learning_rate"] |= "Base Lr:" in line
        pcc_summary = re.search(
            r"PCC Epoch Summary - Epoch:\s*(\d+)\s+"
            r"valid_pcc_pair_count:\s*(\d+)\s+"
            r"mean_fixed_index_part_distance:\s*([0-9.eE+-]+)",
            line,
        )
        if pcc_summary:
            pcc_epoch_summaries.append({
                "epoch": int(pcc_summary.group(1)),
                "valid_pcc_pair_count": int(pcc_summary.group(2)),
                "mean_fixed_index_part_distance": float(pcc_summary.group(3)),
            })
        hard_summary = re.search(
            r"Hard Alignment Epoch Summary - Epoch:\s*(\d+)\s+"
            r"hard_alignment_loss:\s*([0-9.eE+-]+)\s+"
            r"valid_alignment_pair_count:\s*(\d+)\s+"
            r"mean_hard_path_cost:\s*([0-9.eE+-]+)\s+"
            r"mean_path_absolute_offset:\s*([0-9.eE+-]+)",
            line,
        )
        if hard_summary:
            hard_alignment_epoch_summaries.append({
                "epoch": int(hard_summary.group(1)),
                "hard_alignment_loss": float(hard_summary.group(2)),
                "valid_alignment_pair_count": int(hard_summary.group(3)),
                "mean_hard_path_cost": float(hard_summary.group(4)),
                "mean_path_absolute_offset": float(hard_summary.group(5)),
            })
        soft_summary = re.search(
            r"Soft Alignment Epoch Summary - Epoch:\s*(\d+)\s+"
            r"soft_alignment_loss:\s*([0-9.eE+-]+)\s+"
            r"valid_alignment_pair_count:\s*(\d+)\s+"
            r"mean_soft_path_cost:\s*([0-9.eE+-]+)\s+"
            r"alignment_temperature:\s*([0-9.eE+-]+)",
            line,
        )
        if soft_summary:
            values = {
                "epoch": int(soft_summary.group(1)),
                "soft_alignment_loss": float(soft_summary.group(2)),
                "valid_alignment_pair_count": int(soft_summary.group(3)),
                "mean_soft_path_cost": float(soft_summary.group(4)),
                "alignment_temperature": float(soft_summary.group(5)),
            }
            if not all(math.isfinite(values[field]) for field in (
                    "soft_alignment_loss", "mean_soft_path_cost",
                    "alignment_temperature")):
                raise EvidenceError("Soft alignment summary is non-finite")
            soft_alignment_epoch_summaries.append(values)
        windowed_soft_summary = re.search(
            r"Windowed Soft Alignment Epoch Summary - Epoch:\s*(\d+)\s+"
            r"window:\s*(\d+)\s+alignment_temperature:\s*([0-9.eE+-]+)\s+"
            r"windowed_soft_alignment_loss:\s*([0-9.eE+-]+)\s+"
            r"valid_alignment_pair_count:\s*(\d+)\s+"
            r"mean_windowed_soft_path_cost:\s*([0-9.eE+-]+)",
            line,
        )
        if windowed_soft_summary:
            values = {
                "epoch": int(windowed_soft_summary.group(1)),
                "alignment_window": int(windowed_soft_summary.group(2)),
                "alignment_temperature": float(windowed_soft_summary.group(3)),
                "windowed_soft_alignment_loss": float(
                    windowed_soft_summary.group(4)
                ),
                "valid_alignment_pair_count": int(
                    windowed_soft_summary.group(5)
                ),
                "mean_windowed_soft_path_cost": float(
                    windowed_soft_summary.group(6)
                ),
            }
            if values["alignment_window"] <= 0 or not all(
                    math.isfinite(values[field]) for field in (
                        "alignment_temperature",
                        "windowed_soft_alignment_loss",
                        "mean_windowed_soft_path_cost",
                    )):
                raise EvidenceError("Windowed soft alignment summary is invalid")
            windowed_soft_alignment_epoch_summaries.append(values)
        iteration_match = re.search(
            r"Epoch\[(\d+)\]\s+Iteration\[(\d+)/(\d+)\]", line
        )
        if iteration_match:
            iterations_per_epoch_values.add(int(iteration_match.group(3)))
        evidence_match = re.search(
            r"EPOCH_EVIDENCE\s+epoch=(\d+)\s+global_iteration=(\d+)"
            r"\s+epoch_length=(\d+)",
            line,
        )
        if evidence_match:
            epoch = int(evidence_match.group(1))
            global_iteration = int(evidence_match.group(2))
            epoch_length = int(evidence_match.group(3))
            if epoch <= 0 or global_iteration <= 0 or epoch_length <= 0:
                raise EvidenceError(
                    "EPOCH_EVIDENCE values must be positive integers"
                )
            if epoch in epoch_evidence:
                raise EvidenceError(
                    "Duplicate EPOCH_EVIDENCE for epoch {}".format(epoch)
                )
            epoch_evidence[epoch] = {
                "epoch": epoch,
                "global_iteration": global_iteration,
                "epoch_length": epoch_length,
            }
            continue
        epoch_match = re.search(r"Validation Results - Epoch:\s*(\d+)", line)
        if epoch_match:
            if current is not None:
                validations.append(current)
            current = {"epoch": int(epoch_match.group(1))}
            continue
        if current is None:
            continue
        map_match = re.search(r"mAP:\s*([0-9.]+)%", line)
        if map_match:
            current["map_percent"] = float(map_match.group(1))
            continue
        rank_match = re.search(r"Rank-(1|5|10)\s*:\s*([0-9.]+)%", line)
        if rank_match:
            current["rank{}_percent".format(rank_match.group(1))] = float(
                rank_match.group(2)
            )
    if current is not None:
        validations.append(current)
    missing_log_fields = sorted(
        key for key, present in required_training_fields.items() if not present
    )
    if missing_log_fields:
        raise EvidenceError(
            "Training log lacks mandatory fields: {}".format(missing_log_fields)
        )
    if not validations:
        raise EvidenceError("Training log contains no validation history")
    if epoch_evidence:
        evidence_epoch_by_iteration = {}
        for evidence in epoch_evidence.values():
            global_iteration = evidence["global_iteration"]
            if global_iteration in evidence_epoch_by_iteration:
                raise EvidenceError(
                    "EPOCH_EVIDENCE global iteration {} maps to multiple "
                    "epochs: {} and {}".format(
                        global_iteration,
                        evidence_epoch_by_iteration[global_iteration],
                        evidence["epoch"],
                    )
                )
            evidence_epoch_by_iteration[global_iteration] = evidence["epoch"]
        for validation in validations:
            epoch = int(validation["epoch"])
            if epoch not in epoch_evidence:
                raise EvidenceError(
                    "Validation epoch {} lacks EPOCH_EVIDENCE".format(epoch)
                )
            evidence = epoch_evidence[epoch]
            validation["epoch_length"] = evidence["epoch_length"]
            validation["iterations_per_epoch"] = evidence["epoch_length"]
            validation["global_iteration"] = evidence["global_iteration"]
            validation["global_iteration_source"] = (
                "ignite_engine_epoch_evidence"
            )
            validation["timestamp_utc"] = NOT_RECORDED
        global_iteration_source = "ignite_engine_epoch_evidence"
    else:
        if len(iterations_per_epoch_values) != 1:
            raise EvidenceError(
                "Training log cannot determine one iterations_per_epoch value"
            )
        iterations_per_epoch = next(iter(iterations_per_epoch_values))
        for validation in validations:
            validation["epoch_length"] = iterations_per_epoch
            validation["iterations_per_epoch"] = iterations_per_epoch
            validation["global_iteration"] = (
                int(validation["epoch"]) * iterations_per_epoch
            )
            validation["global_iteration_source"] = (
                "legacy_log_denominator_inference"
            )
            validation["timestamp_utc"] = NOT_RECORDED
        global_iteration_source = "legacy_log_denominator_inference"
    runtime = NOT_RECORDED
    if timestamps:
        runtime = (max(timestamps) - min(timestamps)).total_seconds()
    total_pcc_pairs = sum(
        row["valid_pcc_pair_count"] for row in pcc_epoch_summaries
    )
    mean_pcc_distance = NOT_RECORDED
    if total_pcc_pairs:
        mean_pcc_distance = sum(
            row["valid_pcc_pair_count"]
            * row["mean_fixed_index_part_distance"]
            for row in pcc_epoch_summaries
        ) / float(total_pcc_pairs)
    elif pcc_epoch_summaries:
        mean_pcc_distance = 0.0
    total_alignment_pairs = sum(
        row["valid_alignment_pair_count"]
        for row in hard_alignment_epoch_summaries
    )

    def pair_weighted_hard_value(field):
        if total_alignment_pairs:
            return sum(
                row["valid_alignment_pair_count"] * row[field]
                for row in hard_alignment_epoch_summaries
            ) / float(total_alignment_pairs)
        if hard_alignment_epoch_summaries:
            return 0.0
        return NOT_RECORDED

    total_soft_alignment_pairs = sum(
        row["valid_alignment_pair_count"]
        for row in soft_alignment_epoch_summaries
    )

    def pair_weighted_soft_value(field):
        if total_soft_alignment_pairs:
            return sum(
                row["valid_alignment_pair_count"] * row[field]
                for row in soft_alignment_epoch_summaries
            ) / float(total_soft_alignment_pairs)
        if soft_alignment_epoch_summaries:
            return 0.0
        return NOT_RECORDED

    soft_temperatures = {
        row["alignment_temperature"]
        for row in soft_alignment_epoch_summaries
    }
    if len(soft_temperatures) > 1:
        raise EvidenceError(
            "Soft alignment temperature changed across epoch summaries"
        )
    soft_temperature = (
        next(iter(soft_temperatures)) if soft_temperatures else NOT_RECORDED
    )
    total_windowed_soft_alignment_pairs = sum(
        row["valid_alignment_pair_count"]
        for row in windowed_soft_alignment_epoch_summaries
    )

    def pair_weighted_windowed_soft_value(field):
        if total_windowed_soft_alignment_pairs:
            return sum(
                row["valid_alignment_pair_count"] * row[field]
                for row in windowed_soft_alignment_epoch_summaries
            ) / float(total_windowed_soft_alignment_pairs)
        if windowed_soft_alignment_epoch_summaries:
            return 0.0
        return NOT_RECORDED

    windowed_soft_temperatures = {
        row["alignment_temperature"]
        for row in windowed_soft_alignment_epoch_summaries
    }
    windowed_soft_windows = {
        row["alignment_window"]
        for row in windowed_soft_alignment_epoch_summaries
    }
    if len(windowed_soft_temperatures) > 1:
        raise EvidenceError(
            "Windowed soft alignment temperature changed across summaries"
        )
    if len(windowed_soft_windows) > 1:
        raise EvidenceError(
            "Windowed soft alignment window changed across summaries"
        )

    return {
        "path": normalized_path(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "timestamp_runtime_seconds": runtime,
        "raw_text": raw_text,
        "has_camera_aware_loss": "loss_camera_triplet:" in raw_text,
        "has_cross_camera_positive_loss": (
            "loss_cross_camera_positive:" in raw_text
        ),
        "has_pcc_loss": "loss_pcc:" in raw_text,
        "has_hard_alignment_loss": "hard_alignment_loss:" in raw_text,
        "has_soft_alignment_loss": "soft_alignment_loss:" in raw_text,
        "has_windowed_soft_alignment_loss": (
            "windowed_soft_alignment_loss:" in raw_text
        ),
        "pcc_epoch_summaries": pcc_epoch_summaries,
        "hard_alignment_epoch_summaries": hard_alignment_epoch_summaries,
        "soft_alignment_epoch_summaries": soft_alignment_epoch_summaries,
        "windowed_soft_alignment_epoch_summaries": (
            windowed_soft_alignment_epoch_summaries
        ),
        "valid_pcc_pair_count": (
            total_pcc_pairs if pcc_epoch_summaries
            else total_alignment_pairs if hard_alignment_epoch_summaries
            else total_soft_alignment_pairs if soft_alignment_epoch_summaries
            else total_windowed_soft_alignment_pairs
            if windowed_soft_alignment_epoch_summaries
            else NOT_RECORDED
        ),
        "mean_fixed_index_part_distance": mean_pcc_distance,
        "hard_alignment_loss": pair_weighted_hard_value(
            "hard_alignment_loss"
        ),
        "valid_alignment_pair_count": (
            total_alignment_pairs if hard_alignment_epoch_summaries
            else total_soft_alignment_pairs
            if soft_alignment_epoch_summaries
            else total_windowed_soft_alignment_pairs
            if windowed_soft_alignment_epoch_summaries else NOT_RECORDED
        ),
        "mean_hard_path_cost": pair_weighted_hard_value(
            "mean_hard_path_cost"
        ),
        "mean_path_absolute_offset": pair_weighted_hard_value(
            "mean_path_absolute_offset"
        ),
        "soft_alignment_loss": pair_weighted_soft_value(
            "soft_alignment_loss"
        ),
        "mean_soft_path_cost": pair_weighted_soft_value(
            "mean_soft_path_cost"
        ),
        "alignment_temperature": soft_temperature,
        "windowed_soft_alignment_loss": pair_weighted_windowed_soft_value(
            "windowed_soft_alignment_loss"
        ),
        "mean_windowed_soft_path_cost": pair_weighted_windowed_soft_value(
            "mean_windowed_soft_path_cost"
        ),
        "windowed_alignment_temperature": (
            next(iter(windowed_soft_temperatures))
            if windowed_soft_temperatures else NOT_RECORDED
        ),
        "alignment_window": (
            next(iter(windowed_soft_windows))
            if windowed_soft_windows else NOT_RECORDED
        ),
        "epoch_evidence": [
            epoch_evidence[epoch] for epoch in sorted(epoch_evidence)
        ],
        "global_iteration_source": global_iteration_source,
        "validations": validations,
    }


def write_validation_history(path, rows):
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True)
        for row in rows
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def read_validation_history(path):
    target = Path(path)
    if not target.is_file():
        raise EvidenceError("validation_history.jsonl is missing")
    rows = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise EvidenceError(
                    "Invalid validation JSONL line {}: {}".format(line_number, error)
                )
            required = (
                "epoch", "global_iteration", "iterations_per_epoch",
                "rank1_percent", "rank5_percent", "rank10_percent",
                "map_percent",
            )
            missing = [key for key in required if key not in row]
            if missing:
                raise EvidenceError(
                    "Validation row {} lacks {}".format(line_number, missing)
                )
            source = row.get(
                "global_iteration_source", "legacy_validation_history"
            )
            row["global_iteration_source"] = source
            if "epoch_length" not in row:
                row["epoch_length"] = row["iterations_per_epoch"]
            if source == "ignite_engine_epoch_evidence":
                if int(row["epoch_length"]) != int(
                        row["iterations_per_epoch"]):
                    raise EvidenceError(
                        "Authoritative validation row {} has inconsistent "
                        "epoch length".format(line_number)
                    )
            rows.append(row)
    if not rows:
        raise EvidenceError("Validation history is empty")
    epochs = [int(row["epoch"]) for row in rows]
    if len(set(epochs)) != len(epochs):
        raise EvidenceError("Validation history contains duplicate epochs")
    return rows


def select_best_validation(rows):
    return max(
        rows,
        key=lambda row: (
            float(row["rank1_percent"]),
            float(row["map_percent"]),
            int(row["epoch"]),
        ),
    )


def cross_validate_log_metrics(log_info, validation_rows):
    by_epoch = {int(row["epoch"]): row for row in log_info["validations"]}
    keys = ("rank1_percent", "rank5_percent", "rank10_percent", "map_percent")
    for row in validation_rows:
        epoch = int(row["epoch"])
        if epoch not in by_epoch:
            raise EvidenceError("Epoch {} is absent from log validation blocks".format(epoch))
        log_row = by_epoch[epoch]
        if log_row.get("global_iteration_source") == (
                "ignite_engine_epoch_evidence"):
            if row.get("global_iteration_source") != (
                    "ignite_engine_epoch_evidence"):
                raise EvidenceError(
                    "Validation epoch {} does not retain authoritative Ignite "
                    "iteration evidence".format(epoch)
                )
            for key in ("global_iteration", "epoch_length"):
                if int(row[key]) != int(log_row[key]):
                    raise EvidenceError(
                        "Validation/log evidence conflict at epoch {} for {}"
                        .format(epoch, key)
                    )
        for key in keys:
            if key not in log_row:
                raise EvidenceError("Log epoch {} lacks {}".format(epoch, key))
            if abs(float(log_row[key]) - float(row[key])) > 0.051:
                raise EvidenceError(
                    "Validation/log metric conflict at epoch {} for {}".format(epoch, key)
                )
    return True


def _checkpoint_files(output_dir):
    output_dir = Path(output_dir)
    candidates = []
    for pattern in ("*.pt", "*.pth"):
        candidates.extend(output_dir.glob(pattern))
    result = []
    for path in sorted(set(candidates)):
        lower = path.name.lower()
        if "optimizer" in lower or "center_param" in lower:
            continue
        if "checkpoint" in lower or "model" in lower:
            result.append(path)
    return result


def build_checkpoint_manifest(output_dir, validation_rows, selected_epoch,
                              destination):
    sources = {
        row.get("global_iteration_source", "legacy_validation_history")
        for row in validation_rows
    }
    authoritative = sources == {"ignite_engine_epoch_evidence"}
    if "ignite_engine_epoch_evidence" in sources and not authoritative:
        raise EvidenceError(
            "Validation history mixes authoritative and legacy iteration evidence"
        )
    iterations_per_epoch = None
    if not authoritative:
        iteration_per_epoch_values = {
            int(row["iterations_per_epoch"]) for row in validation_rows
        }
        if len(iteration_per_epoch_values) != 1:
            raise EvidenceError("iterations_per_epoch is inconsistent")
        iterations_per_epoch = next(iter(iteration_per_epoch_values))
        if iterations_per_epoch <= 0:
            raise EvidenceError("iterations_per_epoch must be positive")
    validation_by_iteration = {}
    validation_by_epoch = {}
    for row in validation_rows:
        epoch = int(row["epoch"])
        global_iteration = int(row["global_iteration"])
        if epoch in validation_by_epoch:
            raise EvidenceError(
                "Validation history contains duplicate epoch {}".format(epoch)
            )
        if global_iteration in validation_by_iteration:
            raise EvidenceError(
                "Global iteration {} maps to multiple validation epochs"
                .format(global_iteration)
            )
        validation_by_epoch[epoch] = row
        validation_by_iteration[global_iteration] = row
    rows = []
    for path in _checkpoint_files(output_dir):
        global_iteration = NOT_RECORDED
        epoch = NOT_RECORDED
        iteration_match = re.search(r"checkpoint_(\d+)\.(?:pt|pth)$", path.name)
        legacy_epoch_match = re.search(r"model_(\d+)\.pth$", path.name)
        if iteration_match:
            global_iteration = int(iteration_match.group(1))
            if global_iteration in validation_by_iteration:
                validation = validation_by_iteration[global_iteration]
                epoch = int(validation["epoch"])
                mapping_source = validation.get(
                    "global_iteration_source", "legacy_validation_history"
                )
            elif authoritative:
                raise EvidenceError(
                    "Checkpoint global iteration {} does not exactly match "
                    "one Ignite EPOCH_EVIDENCE record".format(
                        global_iteration
                    )
                )
            elif global_iteration % iterations_per_epoch == 0:
                epoch = global_iteration // iterations_per_epoch
                mapping_source = "legacy_log_denominator_inference"
            else:
                raise EvidenceError(
                    "Checkpoint iteration {} cannot map uniquely to epoch".format(
                        global_iteration
                    )
                )
        elif legacy_epoch_match:
            if authoritative:
                raise EvidenceError(
                    "Authoritative Ignite evidence requires checkpoint global "
                    "iteration filenames"
                )
            epoch = int(legacy_epoch_match.group(1))
            global_iteration = epoch * iterations_per_epoch
            mapping_source = "legacy_log_denominator_inference"
        else:
            continue
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "epoch": epoch,
            "global_iteration": global_iteration,
            "global_iteration_source": mapping_source,
            "filename": path.name,
            "path": normalized_path(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "selected": int(epoch) == int(selected_epoch),
        })
    selected = [row for row in rows if row["selected"]]
    if len(selected) != 1:
        raise EvidenceError(
            "Selected epoch {} must bind to exactly one checkpoint; found {}".format(
                selected_epoch, len(selected)
            )
        )
    best_row = next(
        row for row in validation_rows if int(row["epoch"]) == int(selected_epoch)
    )
    if int(selected[0]["global_iteration"]) != int(best_row["global_iteration"]):
        raise EvidenceError("Selected checkpoint global iteration does not match best epoch")
    write_tsv(
        destination,
        (
            "schema_version", "epoch", "global_iteration",
            "global_iteration_source", "filename", "path", "size_bytes",
            "sha256", "selected",
        ),
        rows,
    )
    return rows, selected[0]


def write_tsv(path, fields, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("{}.tmp.{}".format(target.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _table_value(row.get(field)) for field in fields})
    os.replace(str(temporary), str(target))
    return target


def _table_value(value):
    if value is None:
        return NOT_RECORDED
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _read_csv(path):
    target = Path(path)
    if not target.is_file() or target.stat().st_size == 0:
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _legacy_field_value(row, field):
    """Supply explicit, lossless defaults while migrating v1 registries."""
    if field == "schema_version":
        return "1"
    if field == "run_kind":
        return "formal"
    pcc_mode = row.get("pcc_mode") or row.get("alignment_strategy")
    pcc_enabled = str(row.get("pcc_enabled", "")).lower() == "true"
    if field == "method_family":
        return "part_alignment" if pcc_enabled or pcc_mode else NOT_RECORDED
    if field == "method_variant":
        return pcc_mode or NOT_RECORDED
    if field == "alignment_mode":
        return pcc_mode or NOT_APPLICABLE
    if field == "alignment_window":
        return NOT_RECORDED if pcc_mode == "windowed_soft_min" else NOT_APPLICABLE
    if field in (
            "alignment_temperature", "gating_mode", "gating_temperature"):
        return NOT_APPLICABLE if pcc_enabled or pcc_mode else NOT_RECORDED
    if field in (
            "hard_alignment_loss", "valid_alignment_pair_count",
            "mean_hard_path_cost", "mean_path_absolute_offset"):
        return NOT_APPLICABLE if pcc_mode == "fixed_index" else NOT_RECORDED
    if field in ("soft_alignment_loss", "mean_soft_path_cost"):
        return NOT_RECORDED if pcc_mode == "soft_min" else NOT_APPLICABLE
    if field in (
            "parent_branch", "parent_commit",
            "multigranular_feature_signature",
            "multigranular_feature_signature_sha256"):
        return NOT_RECORDED
    return NOT_RECORDED


def migrate_delimited_schema(path, fields, delimiter=","):
    """Expand an older registry header without dropping rows or fields."""
    target = Path(path)
    rows = []
    existing_fields = ()
    if target.is_file() and target.stat().st_size:
        with target.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            existing_fields = tuple(reader.fieldnames or ())
            rows = list(reader)
    unknown_fields = [field for field in existing_fields if field not in fields]
    if unknown_fields:
        raise EvidenceError(
            "Schema migration would discard historical fields {} from {}"
            .format(unknown_fields, target)
        )
    if existing_fields == tuple(fields):
        return rows
    migrated = []
    for row in rows:
        migrated.append({
            field: row.get(field, _legacy_field_value(row, field))
            for field in fields
        })
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("{}.tmp.{}".format(target.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter=delimiter,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(migrated)
    os.replace(str(temporary), str(target))
    return migrated


def upsert_csv(path, fields, row, key_fields=("run_id",)):
    rows = migrate_delimited_schema(path, fields, delimiter=",")
    normalized = {field: _table_value(row.get(field, NOT_RECORDED)) for field in fields}
    key = tuple(normalized[field] for field in key_fields)
    replaced = False
    for index, existing in enumerate(rows):
        if tuple(existing.get(field, "") for field in key_fields) == key:
            rows[index] = normalized
            replaced = True
            break
    if not replaced:
        rows.append(normalized)
    rows.sort(key=lambda item: tuple(item.get(field, "") for field in key_fields))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("{}.tmp.{}".format(target.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(target))
    return rows


def upsert_tsv(path, fields, row, key_fields=("run_id",)):
    target = Path(path)
    rows = migrate_delimited_schema(target, fields, delimiter="\t")
    normalized = {field: _table_value(row.get(field, NOT_RECORDED)) for field in fields}
    key = tuple(normalized[field] for field in key_fields)
    replaced = False
    for index, existing in enumerate(rows):
        if tuple(existing.get(field, "") for field in key_fields) == key:
            rows[index] = normalized
            replaced = True
            break
    if not replaced:
        rows.append(normalized)
    rows.sort(key=lambda item: tuple(item.get(field, "") for field in key_fields))
    write_tsv(target, fields, rows)
    return rows


def csv_to_markdown(csv_path, markdown_path):
    target = Path(csv_path)
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = str(row.get(field, "")).replace("|", "\\|").replace("\n", " ")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    atomic_write_text(markdown_path, "\n".join(lines) + "\n")
    return rows


def ensure_record_layout(records_root):
    root = Path(records_root)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, fields in TABLE_SCHEMAS.items():
        csv_path = tables / "{}.csv".format(name)
        migrate_delimited_schema(csv_path, fields, delimiter=",")
        csv_to_markdown(csv_path, tables / "{}.md".format(name))
    runs_path = root / "runs.csv"
    migrate_delimited_schema(runs_path, RUN_FIELDS, delimiter=",")
    evidence_path = root / "evidence_manifest.tsv"
    migrate_delimited_schema(evidence_path, EVIDENCE_FIELDS, delimiter="\t")
    return root


def _format_percent(value):
    return "{:.4f}".format(float(value)).rstrip("0").rstrip(".")


def _gpu_label(environment):
    gpus = environment.get("gpus", [])
    if not gpus:
        return NOT_RECORDED
    return "; ".join(str(item.get("name", NOT_RECORDED)) for item in gpus)


def validate_anchor_coverage(payload):
    required = (
        "total_anchor_count", "valid_cross_camera_anchor_count",
        "invalid_cross_camera_anchor_count", "coverage_percent",
        "cross_camera_positive_count", "same_camera_positive_count",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise EvidenceError("Anchor coverage lacks {}".format(missing))
    total = int(payload["total_anchor_count"])
    valid = int(payload["valid_cross_camera_anchor_count"])
    invalid = int(payload["invalid_cross_camera_anchor_count"])
    if total <= 0 or valid < 0 or invalid < 0 or valid + invalid != total:
        raise EvidenceError("Anchor coverage counts are inconsistent")
    expected = 100.0 * valid / float(total)
    if abs(float(payload["coverage_percent"]) - expected) > 1e-8:
        raise EvidenceError("Anchor coverage percentage is inconsistent")
    return payload


def validate_distance_distribution(payload):
    required = (
        "same_id_same_camera_mean", "same_id_same_camera_std",
        "same_id_cross_camera_mean", "same_id_cross_camera_std",
        "different_id_mean", "different_id_std", "cross_camera_gap",
    )
    missing = [
        key for key in required
        if key not in payload or payload[key] in (None, NOT_RECORDED, MISSING_EVIDENCE)
    ]
    if missing:
        raise EvidenceError("Distance distribution lacks real values: {}".format(missing))
    return payload


def efficiency_placeholder(error=NOT_RECORDED):
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "incomplete",
        "total_params": NOT_RECORDED,
        "trainable_params": NOT_RECORDED,
        "FLOPs": NOT_RECORDED,
        "MACs": NOT_RECORDED,
        "peak_forward_memory": NOT_RECORDED,
        "peak_train_memory": NOT_RECORDED,
        "inference_latency": NOT_RECORDED,
        "throughput": NOT_RECORDED,
        "error": str(error),
    }


def _run_analysis_tools(repo_root, run_dir, manifest, selected_checkpoint):
    config_path = str(run_dir / "config_source.yml")
    distance_path = run_dir / "distance_distribution.json"
    analysis_seed = manifest["seed"] if manifest["seed"] != NOT_RECORDED else 42
    distance_command = [
        sys.executable,
        str(Path(repo_root) / "tools" / "analyze_distance_distributions.py"),
        "--config-file", config_path,
        "--checkpoint", selected_checkpoint["path"],
        "--output", str(distance_path),
        "--seed", str(analysis_seed),
    ]
    completed = subprocess.run(distance_command, cwd=str(repo_root), check=False)
    if completed.returncode != 0 or not distance_path.is_file():
        raise EvidenceError("Distance distribution analysis failed")
    anchor_path = run_dir / "anchor_coverage.json"
    anchor_command = [
        sys.executable,
        str(Path(repo_root) / "tools" / "analyze_anchor_coverage.py"),
        "--config-file", config_path,
        "--output", str(anchor_path),
        "--analysis-seed", str(analysis_seed),
    ]
    anchor = subprocess.run(anchor_command, cwd=str(repo_root), check=False)
    if anchor.returncode != 0 or not anchor_path.is_file():
        raise EvidenceError("Controlled anchor coverage analysis failed")
    efficiency_path = run_dir / "efficiency_profile.json"
    efficiency_command = [
        sys.executable,
        str(Path(repo_root) / "tools" / "profile_efficiency.py"),
        "--config-file", config_path,
        "--checkpoint", selected_checkpoint["path"],
        "--output", str(efficiency_path),
        "--seed", str(analysis_seed),
    ]
    efficiency = subprocess.run(efficiency_command, cwd=str(repo_root), check=False)
    if efficiency.returncode != 0 or not efficiency_path.is_file():
        atomic_write_json(
            efficiency_path,
            efficiency_placeholder("efficiency profiler failed"),
        )


def _artifact_rows(run_dir, log_info, checkpoint_rows, final_status):
    run_dir = Path(run_dir)
    rows = []
    final_status_bytes = json_text(final_status).encode("utf-8")
    for path in sorted(run_dir.iterdir()):
        if not path.is_file() or path.name in (
                "artifact_hashes.tsv", "run_status.json", "run_manifest.json"):
            continue
        rows.append({
            "artifact_type": _artifact_type_for_path(path),
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    rows.append({
        "artifact_type": "run_status",
        "path": "run_status.json",
        "size_bytes": len(final_status_bytes),
        "sha256": sha256_bytes(final_status_bytes),
    })
    rows.extend([
        {
            "artifact_type": "training_log",
            "path": log_info["path"],
            "size_bytes": log_info["size_bytes"],
            "sha256": log_info["sha256"],
        },
    ])
    for checkpoint in checkpoint_rows:
        rows.append({
            "artifact_type": (
                "selected_checkpoint" if checkpoint["selected"]
                else "checkpoint"
            ),
            "path": checkpoint["path"],
            "size_bytes": checkpoint["size_bytes"],
            "sha256": checkpoint["sha256"],
        })
    return rows


def _remove_run_from_formal_tables(records_root, run_id):
    """Rollback a finalization that failed after formal table staging."""
    tables = Path(records_root) / "tables"
    for name, fields in TABLE_SCHEMAS.items():
        path = tables / "{}.csv".format(name)
        rows = migrate_delimited_schema(path, fields, delimiter=",")
        kept = [row for row in rows if row.get("run_id") != str(run_id)]
        if len(kept) == len(rows):
            continue
        temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(kept)
        os.replace(str(temporary), str(path))
        csv_to_markdown(path, tables / "{}.md".format(name))


def _lambda_table_eligible(manifest):
    family = str(manifest["experiment_family"]).lower()
    identifier = str(manifest["experiment_id"]).lower()
    return (
        manifest["lambda"] != NOT_RECORDED
        and manifest.get("experiment_family") != SOFT_LAMBDA_SWEEP_FAMILY
        and ("lambda" in family or re.search(r"(?:^|-)l\d+", identifier))
    )


def _soft_alignment_lambda_table_eligible(manifest):
    try:
        tau = float(manifest.get("alignment_temperature"))
        alignment_lambda = float(manifest.get("pcc_lambda"))
    except (TypeError, ValueError):
        return False
    return (
        manifest.get("experiment_family") == SOFT_LAMBDA_SWEEP_FAMILY
        and manifest.get("run_kind") == "formal"
        and manifest.get("status", "success") == "success"
        and manifest.get("alignment_mode") == "soft_min"
        and tau == 0.2
        and alignment_lambda in (0.05, 0.1, 0.3)
    )


def _read_tsv_rows(path):
    target = Path(path)
    if not target.is_file() or target.stat().st_size == 0:
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _markdown_value(value):
    if value is None or value == "":
        value = NOT_RECORDED
    return (_normalized_text(value).replace("\\", "\\\\")
            .replace("|", "\\|").replace("\n", "<br>"))


def _markdown_table_section(start, end, title, source_text, fields, rows):
    lines = [
        start, "## {}".format(title), "", source_text, "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            _markdown_value(row.get(field, NOT_RECORDED))
            for field in fields
        ) + " |")
    lines.extend([end, ""])
    return "\n".join(lines)


def _replace_generated_section(content, start, end, generated):
    has_start = start in content
    has_end = end in content
    if has_start != has_end:
        raise EvidenceError(
            "EXPERIMENTS.md has an incomplete generated section {}".format(start)
        )
    if has_start:
        prefix, remainder = content.split(start, 1)
        _, suffix = remainder.split(end, 1)
        return prefix.rstrip() + "\n\n" + generated.rstrip() + "\n" + suffix.lstrip()
    return content.rstrip() + "\n\n" + generated.rstrip() + "\n"


def _authoritative_run_rows(records_root):
    """Read the machine registry, recovering only rows absent from it.

    Persisted registry rows must not be re-derived on another host: doing so
    would rewrite authoritative paths, sizes, and hashes using checkout-local
    path and newline semantics.
    """
    records_root = Path(records_root)
    registry = {
        row.get("run_id"): row for row in _read_csv(records_root / "runs.csv")
        if row.get("run_id")
    }
    for run_dir in sorted((records_root / "runs").glob("*")):
        manifest_path = run_dir / "run_manifest.json"
        status_path = run_dir / "run_status.json"
        if not manifest_path.is_file() or not status_path.is_file():
            continue
        manifest = read_json(manifest_path)
        status = read_json(status_path)
        run_id = manifest.get("run_id", run_dir.name)
        if run_id not in registry:
            registry[run_id] = _run_row_from_manifest(
                run_dir, manifest, status
            )
    return sorted(
        registry.values(),
        key=lambda row: (row.get("start_time", ""), row.get("run_id", "")),
    )


def _authoritative_checkpoint_rows(records_root, run_rows):
    run_index = {row.get("run_id"): row for row in run_rows}
    rows = []
    for run_dir in sorted((Path(records_root) / "runs").glob("*")):
        path = run_dir / "checkpoint_manifest.tsv"
        if not path.is_file():
            continue
        for checkpoint in _read_tsv_rows(path):
            run = run_index.get(run_dir.name, {})
            rows.append({
                "run_id": checkpoint.get("run_id", run_dir.name),
                "experiment_id": run.get("experiment_id", NOT_RECORDED),
                "run_kind": run.get("run_kind", NOT_RECORDED),
                "checkpoint_path": checkpoint.get("path", MISSING_EVIDENCE),
                "size_bytes": checkpoint.get("size_bytes", MISSING_EVIDENCE),
                "ignite_epoch": checkpoint.get("epoch", MISSING_EVIDENCE),
                "global_iteration": checkpoint.get(
                    "global_iteration", MISSING_EVIDENCE
                ),
                "sha256": checkpoint.get("sha256", MISSING_EVIDENCE),
                "selected": checkpoint.get("selected", "False"),
            })
    return sorted(rows, key=lambda row: (
        row.get("run_id", ""), str(row.get("ignite_epoch", "")),
        row.get("checkpoint_path", ""),
    ))


def update_experiments_markdown(experiments_path, records_root):
    """Atomically regenerate all recorder-owned Markdown sections."""
    target = Path(experiments_path)
    records_root = Path(records_root)
    historical = (
        target.read_text(encoding="utf-8")
        if target.is_file() else "# Experiments\n"
    )
    # Reading this source is intentional: it is authoritative artifact evidence,
    # while per-run manifests provide the denormalized display values.
    _read_tsv_rows(records_root / "evidence_manifest.tsv")
    run_rows = _authoritative_run_rows(records_root)
    run_start = {
        row.get("run_id"): row.get("start_time", "") for row in run_rows
    }
    formal_rows = sorted([
        row for row in _read_csv(records_root / "tables" / "main_results.csv")
        if row.get("run_kind") == "formal" and row.get("status") == "success"
    ], key=lambda row: (
        run_start.get(row.get("run_id"), ""), row.get("run_id", "")
    ))
    checkpoint_rows = _authoritative_checkpoint_rows(records_root, run_rows)
    runs_section = _markdown_table_section(
        AUTO_RUNS_START, AUTO_RUNS_END, "Run Registry / All Recorded Runs",
        "Generated from `experiment_records/runs.csv`, per-run manifests, "
        "statuses, and `evidence_manifest.tsv`.", RUN_FIELDS, run_rows,
    )
    formal_section = _markdown_table_section(
        AUTO_RESULTS_START, AUTO_RESULTS_END, "Formal Results",
        "Generated only from successful formal rows in "
        "`experiment_records/tables/main_results.csv`.",
        MAIN_FIELDS, formal_rows,
    )
    checkpoint_fields = (
        "run_id", "experiment_id", "run_kind", "checkpoint_path",
        "size_bytes", "ignite_epoch", "global_iteration", "sha256",
        "selected",
    )
    checkpoint_section = _markdown_table_section(
        AUTO_CHECKPOINTS_START, AUTO_CHECKPOINTS_END, "Checkpoint Evidence",
        "Generated from each run's authoritative `checkpoint_manifest.tsv`; "
        "epochs and iterations are Ignite evidence, never inferred.",
        checkpoint_fields, checkpoint_rows,
    )
    content = _replace_generated_section(
        historical, AUTO_RUNS_START, AUTO_RUNS_END, runs_section
    )
    content = _replace_generated_section(
        content, AUTO_RESULTS_START, AUTO_RESULTS_END, formal_section
    )
    content = _replace_generated_section(
        content, AUTO_CHECKPOINTS_START, AUTO_CHECKPOINTS_END,
        checkpoint_section,
    )
    atomic_write_text(target, content)
    return {
        "all_runs": len(run_rows), "formal_results": len(formal_rows),
        "checkpoints": len(checkpoint_rows),
    }


def _prepare_table_rows(manifest, metrics, environment, efficiency,
                        distance, anchor):
    run_kind = manifest.get("run_kind", "formal")
    common = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "experiment_id": manifest["experiment_id"],
        "experiment_family": manifest["experiment_family"],
        "run_kind": run_kind,
        "method": manifest["method"],
        "method_family": manifest.get("method_family", NOT_RECORDED),
        "method_variant": manifest.get("method_variant", NOT_RECORDED),
        "dataset": manifest["dataset"],
        "branch": manifest["branch"],
        "commit": manifest["commit_id"],
        "parent_branch": manifest.get("parent_branch", NOT_RECORDED),
        "parent_commit": manifest.get("parent_commit", NOT_RECORDED),
        "seed": manifest["seed"],
        "lambda": manifest["lambda"],
        "cross_camera_positive_lambda": manifest.get(
            "cross_camera_positive_lambda", manifest["lambda"]
        ),
        "pcc_lambda": manifest.get("pcc_lambda", NOT_RECORDED),
        "pcc_enabled": manifest.get("pcc_enabled", False),
        "pcc_parts": manifest.get("pcc_parts", NOT_RECORDED),
        "pcc_mode": manifest.get("pcc_mode", NOT_RECORDED),
        "alignment_strategy": manifest.get(
            "alignment_strategy", NOT_RECORDED
        ),
        "alignment_mode": manifest.get("alignment_mode", NOT_RECORDED),
        "alignment_temperature": manifest.get(
            "alignment_temperature", NOT_RECORDED
        ),
        "alignment_window": manifest.get("alignment_window", NOT_RECORDED),
        "gating_mode": manifest.get("gating_mode", NOT_RECORDED),
        "gating_temperature": manifest.get(
            "gating_temperature", NOT_RECORDED
        ),
        "multigranular_feature_signature": manifest.get(
            "multigranular_feature_signature", NOT_RECORDED
        ),
        "multigranular_feature_signature_sha256": manifest.get(
            "multigranular_feature_signature_sha256", NOT_RECORDED
        ),
        "feature_reference_commit": manifest.get(
            "feature_reference_commit", NOT_RECORDED
        ),
        "feature_reference_signature_sha256": manifest.get(
            "feature_reference_signature_sha256", NOT_RECORDED
        ),
        "current_feature_signature_sha256": manifest.get(
            "current_feature_signature_sha256", NOT_RECORDED
        ),
        "feature_compatibility_status": manifest.get(
            "feature_compatibility_status", NOT_RECORDED
        ),
        "feature_compatibility_evidence_path": manifest.get(
            "feature_compatibility_evidence_path", NOT_RECORDED
        ),
        "feature_compatibility_evidence_size_bytes": manifest.get(
            "feature_compatibility_evidence_size_bytes", NOT_RECORDED
        ),
        "feature_compatibility_evidence_sha256": manifest.get(
            "feature_compatibility_evidence_sha256", NOT_RECORDED
        ),
        "baseline": manifest.get("baseline", NOT_RECORDED),
        "margin": manifest["margin"],
        "mode": manifest["mode"],
        "best_epoch": metrics["best_epoch"],
        "selected_epoch": metrics["selected_epoch"],
        "rank1": _format_percent(metrics["rank1_percent"]),
        "rank5": _format_percent(metrics["rank5_percent"]),
        "rank10": _format_percent(metrics["rank10_percent"]),
        "map": _format_percent(metrics["map_percent"]),
        "checkpoint": metrics["checkpoint"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "runtime_seconds": metrics["runtime_seconds"],
        "gpu": _gpu_label(environment),
        "config": manifest["config_file"],
        "source_config_path": normalized_path(
            (Path(metrics["run_dir"]) / "config_source.yml").resolve()
        ),
        "source_config_sha256": manifest.get(
            "config_source_sha256", NOT_RECORDED
        ),
        "resolved_config_path": normalized_path(
            (Path(metrics["run_dir"]) / "config_resolved.yml").resolve()
        ),
        "resolved_config_sha256": manifest.get(
            "config_resolved_sha256", NOT_RECORDED
        ),
        "log_path": metrics["log_path"],
        "training_log_size_bytes": Path(metrics["log_path"]).stat().st_size,
        "log_sha256": metrics["log_sha256"],
        "console_log_path": manifest.get("console_log_path", NOT_RECORDED),
        "console_log_size_bytes": manifest.get(
            "console_log_size_bytes", NOT_RECORDED
        ),
        "console_log_sha256": manifest.get(
            "console_log_sha256", NOT_RECORDED
        ),
        "artifact_manifest_path": manifest.get(
            "artifact_manifest_path", NOT_RECORDED
        ),
        "artifact_manifest_size_bytes": (
            Path(manifest["artifact_manifest_path"]).stat().st_size
            if manifest.get("artifact_manifest_path") not in (
                None, "", NOT_RECORDED, MISSING_EVIDENCE
            ) and Path(manifest["artifact_manifest_path"]).is_file()
            else NOT_RECORDED
        ),
        "artifact_manifest_sha256": manifest.get(
            "artifact_manifest_sha256", NOT_RECORDED
        ),
        "output_dir": manifest["output_dir"],
        "valid_pcc_pair_count": metrics["valid_pcc_pair_count"],
        "mean_fixed_index_part_distance": metrics["mean_fixed_index_part_distance"],
        "hard_alignment_loss": metrics["hard_alignment_loss"],
        "valid_alignment_pair_count": metrics["valid_alignment_pair_count"],
        "mean_hard_path_cost": metrics["mean_hard_path_cost"],
        "mean_path_absolute_offset": metrics["mean_path_absolute_offset"],
        "soft_alignment_loss": metrics["soft_alignment_loss"],
        "mean_soft_path_cost": metrics["mean_soft_path_cost"],
        "status": "success",
        "notes": manifest["notes"],
    }
    lambda_row = {
        "run_id": common["run_id"], "experiment_id": common["experiment_id"],
        "method": common["method"], "dataset": common["dataset"],
        "lambda": common["lambda"], "seed": common["seed"],
        "best_epoch": common["best_epoch"], "rank1": common["rank1"],
        "map": common["map"], "runtime": common["runtime_seconds"],
        "checkpoint": common["checkpoint"], "commit": common["commit"],
    }
    identity = experiment_identity(load_yaml(Path(metrics["run_dir"]) / "config_resolved.yml"))
    same_row = {
        "run_id": common["run_id"], "experiment_id": common["experiment_id"],
        "variant": identity["variant"],
        "positive_relation": identity["positive_relation"],
        "lambda": common["lambda"], "rank1": common["rank1"],
        "map": common["map"], "best_epoch": common["best_epoch"],
        "seed": common["seed"], "checkpoint": common["checkpoint"],
        "commit": common["commit"],
    }
    modules = manifest["modules"]
    caat_row = {
        "run_id": common["run_id"], "experiment_id": common["experiment_id"],
        "baseline": modules["baseline"],
        "camera_aware_triplet": modules["camera_aware_triplet"],
        "cross_camera_positive": modules["cross_camera_positive"],
        "same_camera_positive": modules["same_camera_positive"],
        "hierarchical": modules["hierarchical"], "weighted": modules["weighted"],
        "multi_granularity": modules["multi_granularity"],
        "lambda": common["lambda"], "rank1": common["rank1"],
        "map": common["map"],
        "Params": efficiency.get("total_params", NOT_RECORDED),
        "FLOPs": efficiency.get("FLOPs", NOT_RECORDED),
        "runtime": common["runtime_seconds"], "seed": common["seed"],
        "commit": common["commit"],
    }
    distance_row = {
        "run_id": common["run_id"], "experiment_id": common["experiment_id"],
        "dataset": common["dataset"], "checkpoint": common["checkpoint"],
        "same_id_same_camera_mean": distance["same_id_same_camera_mean"],
        "same_id_same_camera_std": distance["same_id_same_camera_std"],
        "same_id_cross_camera_mean": distance["same_id_cross_camera_mean"],
        "same_id_cross_camera_std": distance["same_id_cross_camera_std"],
        "different_id_mean": distance["different_id_mean"],
        "different_id_std": distance["different_id_std"],
        "cross_camera_gap": distance["cross_camera_gap"],
        "seed": common["seed"], "commit": common["commit"],
    }
    anchor_row = {
        "run_id": common["run_id"], "experiment_id": common["experiment_id"],
        "dataset": common["dataset"], "seed": common["seed"],
        "total_anchors": anchor["total_anchor_count"],
        "valid_cross_camera_anchors": anchor["valid_cross_camera_anchor_count"],
        "invalid_cross_camera_anchors": anchor["invalid_cross_camera_anchor_count"],
        "coverage_percent": anchor["coverage_percent"],
        "cross_camera_positive_count": anchor["cross_camera_positive_count"],
        "same_camera_positive_count": anchor["same_camera_positive_count"],
        "commit": common["commit"],
    }
    pcc_row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": common["run_id"],
        "experiment_id": common["experiment_id"],
        "run_kind": common["run_kind"],
        "baseline": common["baseline"],
        "method_family": common["method_family"],
        "method_variant": common["method_variant"],
        "pcc_enabled": common["pcc_enabled"],
        "alignment_strategy": common["alignment_strategy"],
        "alignment_temperature": common["alignment_temperature"],
        "alignment_window": common["alignment_window"],
        "gating_mode": common["gating_mode"],
        "gating_temperature": common["gating_temperature"],
        "pcc_parts": common["pcc_parts"],
        "cross_camera_positive_lambda": common["cross_camera_positive_lambda"],
        "pcc_lambda": common["pcc_lambda"],
        "valid_pcc_pair_count": common["valid_pcc_pair_count"],
        "mean_fixed_index_part_distance": common["mean_fixed_index_part_distance"],
        "hard_alignment_loss": common["hard_alignment_loss"],
        "valid_alignment_pair_count": common["valid_alignment_pair_count"],
        "mean_hard_path_cost": common["mean_hard_path_cost"],
        "mean_path_absolute_offset": common["mean_path_absolute_offset"],
        "soft_alignment_loss": common["soft_alignment_loss"],
        "mean_soft_path_cost": common["mean_soft_path_cost"],
        "best_epoch": common["best_epoch"],
        "rank1": common["rank1"],
        "map": common["map"],
        "runtime": common["runtime_seconds"],
        "seed": common["seed"],
        "commit": common["commit"],
    }
    alignment_row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": common["run_id"],
        "experiment_id": common["experiment_id"],
        "run_kind": common["run_kind"],
        "baseline": common["baseline"],
        "method_family": common["method_family"],
        "method_variant": common["method_variant"],
        "alignment_mode": common["alignment_mode"],
        "alignment_temperature": common["alignment_temperature"],
        "alignment_window": common["alignment_window"],
        "gating_mode": common["gating_mode"],
        "gating_temperature": common["gating_temperature"],
        "multigranular_feature_signature_sha256": common[
            "multigranular_feature_signature_sha256"
        ],
        "parent_branch": common["parent_branch"],
        "parent_commit": common["parent_commit"],
        "parts": common["pcc_parts"],
        "cross_camera_positive_lambda": common[
            "cross_camera_positive_lambda"
        ],
        "alignment_lambda": common["pcc_lambda"],
        "valid_alignment_pair_count": common[
            "valid_alignment_pair_count"
        ],
        "hard_alignment_loss": common["hard_alignment_loss"],
        "mean_hard_path_cost": common["mean_hard_path_cost"],
        "mean_path_absolute_offset": common[
            "mean_path_absolute_offset"
        ],
        "soft_alignment_loss": common["soft_alignment_loss"],
        "mean_soft_path_cost": common["mean_soft_path_cost"],
        "best_epoch": common["best_epoch"],
        "rank1": common["rank1"],
        "map": common["map"],
        "runtime": common["runtime_seconds"],
        "seed": common["seed"],
        "commit": common["commit"],
    }
    return (
        common, lambda_row, same_row, caat_row, distance_row, anchor_row,
        pcc_row, alignment_row, {
            "schema_version": common["schema_version"],
            "run_id": common["run_id"],
            "experiment_id": common["experiment_id"],
            "run_kind": common["run_kind"],
            "status": common["status"],
            "dataset": common["dataset"],
            "method_variant": common["method_variant"],
            "alignment_mode": common["alignment_mode"],
            "alignment_window": common["alignment_window"],
            "alignment_temperature": common["alignment_temperature"],
            "pcc_lambda": common["pcc_lambda"],
            "parts": common["pcc_parts"],
            "seed": common["seed"],
            "rank1": common["rank1"], "rank5": common["rank5"],
            "rank10": common["rank10"], "map": common["map"],
            "best_epoch": common["best_epoch"],
            "runtime": common["runtime_seconds"],
            "checkpoint": common["checkpoint"],
            "checkpoint_sha256": common["checkpoint_sha256"],
            "commit": common["commit"], "output_dir": common["output_dir"],
        },
    )


def _windowed_soft_alignment_table_eligible(manifest):
    try:
        tau = float(manifest.get("alignment_temperature"))
        alignment_lambda = float(manifest.get("pcc_lambda"))
        window = int(manifest.get("alignment_window"))
    except (TypeError, ValueError):
        return False
    return (
        manifest.get("experiment_family") == WINDOWED_SOFT_ALIGNMENT_FAMILY
        and manifest.get("run_kind") == "formal"
        and manifest.get("status", "success") == "success"
        and manifest.get("alignment_mode") == "windowed_soft_min"
        and tau == 0.2
        and alignment_lambda == 0.05
        and window in (1, 2)
    )


def finalize_run(run_dir, records_root, repo_root, experiments_path,
                 run_analyses=True, verify_git=True):
    """Validate all strong evidence, then atomically upsert result registries."""
    run_dir = Path(run_dir)
    records_root = Path(records_root)
    manifest = read_json(run_dir / "run_manifest.json")
    try:
        status = read_json(run_dir / "run_status.json")
        if verify_git:
            validate_git_runtime_state(
                repo_root,
                run_dir,
                expected_branch=manifest["branch"],
                expected_commit=manifest["commit_id"],
            )
        records_root = ensure_record_layout(records_root)
        if int(status.get("training_exit_code", -1)) != 0:
            raise EvidenceError("Training exit code is not zero")
        source_copy = run_dir / "config_source.yml"
        resolved_copy = run_dir / "config_resolved.yml"
        if sha256_file(source_copy) != manifest["config_source_sha256"]:
            raise EvidenceError("Source config hash changed")
        if sha256_file(resolved_copy) != manifest["config_resolved_sha256"]:
            raise EvidenceError("Resolved config hash changed")
        for path, field, label in (
                (source_copy, "config_source_size_bytes", "Source config"),
                (resolved_copy, "config_resolved_size_bytes", "Resolved config")):
            if field in manifest and int(manifest[field]) != path.stat().st_size:
                raise EvidenceError("{} size changed".format(label))
        output_dir = Path(manifest["output_dir"])
        reproducibility_source = output_dir / "reproducibility.json"
        if reproducibility_source.is_file():
            copy_file_atomic(reproducibility_source, run_dir / "reproducibility.json")
        reproducibility = read_json(run_dir / "reproducibility.json")
        required_reproducibility = (
            "source_seed", "resolved_seed", "applied_seed", "seed",
            "runner_seed",
            "PYTHONHASHSEED", "python_random_seed", "numpy_seed",
            "torch_cpu_seed", "torch_cuda_seed",
            "dataloader_worker_seed_base",
            "dataloader_train_generator_seed",
            "dataloader_validation_generator_seed",
            "dataloader_worker_seed_strategy", "sampler_seed",
            "sampler_seed_strategy", "cudnn_deterministic",
            "cudnn_benchmark", "status",
        )
        missing_reproducibility = [
            key for key in required_reproducibility
            if key not in reproducibility
        ]
        if missing_reproducibility:
            raise EvidenceError(
                "Reproducibility evidence lacks {}".format(
                    missing_reproducibility
                )
            )
        source_cfg = load_yaml(source_copy)
        resolved_cfg = load_yaml(resolved_copy)
        identity = experiment_identity(resolved_cfg)
        manifest.setdefault("run_kind", "formal")
        if manifest["run_kind"] not in ("formal", "smoke"):
            raise EvidenceError("Manifest run_kind is invalid")
        for field in (
                "method_family", "method_variant", "alignment_mode",
                "alignment_temperature", "alignment_window", "gating_mode",
                "gating_temperature"):
            manifest.setdefault(field, identity[field])
            if manifest[field] != identity[field]:
                raise EvidenceError(
                    "Manifest/config identity conflict for {}".format(field)
                )
        for field in (
                "pcc_enabled", "pcc_parts", "pcc_lambda", "pcc_mode"):
            expected = identity[field]
            actual = manifest.get(field, expected)
            if actual != expected:
                raise EvidenceError(
                    "Manifest/config PCC conflict for {}".format(field)
                )
            manifest.setdefault(field, expected)
        signature, signature_sha256 = (
            canonical_multigranular_feature_signature(resolved_cfg)
        )
        recorded_signature = manifest.get(
            "multigranular_feature_signature", NOT_RECORDED
        )
        recorded_signature_sha256 = manifest.get(
            "multigranular_feature_signature_sha256", NOT_RECORDED
        )
        if recorded_signature not in (NOT_RECORDED, MISSING_EVIDENCE, ""):
            if recorded_signature != signature:
                raise EvidenceError(
                    "Multigranular feature signature/config conflict"
                )
            if recorded_signature_sha256 != signature_sha256:
                raise EvidenceError(
                    "Multigranular feature signature SHA256 conflict"
                )
        if identity["alignment_mode"] in ("soft_min", "windowed_soft_min"):
            if recorded_signature in (
                    NOT_RECORDED, MISSING_EVIDENCE, "", None):
                raise EvidenceError(
                    "Soft alignment feature signature is missing"
                )
            for parent_field in ("parent_branch", "parent_commit"):
                if manifest.get(parent_field) in (
                        None, "", NOT_RECORDED, MISSING_EVIDENCE):
                    raise EvidenceError(
                        "Soft alignment {} is missing".format(parent_field)
                    )
            if verify_git:
                lineage = validate_parent_lineage(
                    repo_root,
                    manifest["parent_branch"],
                    manifest["parent_commit"],
                    child_commit=manifest["commit_id"],
                )
                if manifest.get("merge_base", NOT_RECORDED) not in (
                        NOT_RECORDED, lineage["merge_base"]):
                    raise EvidenceError("Recorded merge-base differs")
            if int(manifest.get("schema_version", 1)) >= 4:
                feature_path = run_dir / "feature_compatibility.json"
                if not feature_path.is_file():
                    raise EvidenceError(
                        "Parent-bound feature compatibility evidence is missing"
                    )
                feature_sha = sha256_file(feature_path)
                if feature_sha != manifest.get(
                        "feature_compatibility_evidence_sha256"):
                    raise EvidenceError(
                        "Feature compatibility evidence SHA256 differs"
                    )
                feature = read_json(feature_path)
                required_feature = (
                    "feature_reference_commit",
                    "feature_reference_signature_sha256",
                    "current_feature_signature_sha256",
                    "feature_compatibility_status", "components",
                    "mismatched_components",
                )
                missing_feature = [
                    key for key in required_feature if key not in feature
                ]
                if missing_feature:
                    raise EvidenceError(
                        "Feature compatibility evidence lacks {}".format(
                            missing_feature
                        )
                    )
                if feature["feature_reference_commit"].lower() != str(
                        manifest.get("feature_reference_commit", "")).lower():
                    raise EvidenceError(
                        "Feature reference commit evidence differs"
                    )
                if feature["feature_compatibility_status"] != "compatible":
                    raise EvidenceError("Shared feature compatibility failed")
                if feature["mismatched_components"]:
                    raise EvidenceError(
                        "Shared feature components differ: {}".format(
                            feature["mismatched_components"]
                        )
                    )
                for field in (
                        "feature_reference_commit",
                        "feature_reference_signature_sha256",
                        "current_feature_signature_sha256",
                        "feature_compatibility_status"):
                    if feature[field] != manifest.get(field):
                        raise EvidenceError(
                            "Feature manifest conflict for {}".format(field)
                        )
        seed_values = (
            source_cfg.get("SEED", NOT_RECORDED),
            resolved_cfg.get("SEED", NOT_RECORDED),
            reproducibility.get("applied_seed", NOT_RECORDED),
            reproducibility.get("seed", NOT_RECORDED),
            manifest.get("seed", NOT_RECORDED),
        )
        if NOT_RECORDED in seed_values or MISSING_EVIDENCE in seed_values:
            raise EvidenceError(
                "Applied seed evidence is missing; result remains incomplete"
            )
        source_seed = validate_seed(seed_values[0])
        resolved_seed = validate_seed(seed_values[1])
        validate_seed_evidence_chain(
            source_seed,
            resolved_seed,
            seed_values[2],
            metadata_seed=seed_values[3],
            expected_seed=seed_values[4],
        )
        applied_seed_fields = (
            "runner_seed", "python_random_seed", "numpy_seed",
            "torch_cpu_seed",
            "torch_cuda_seed", "dataloader_worker_seed_base",
            "sampler_seed",
        )
        for key in applied_seed_fields:
            if validate_seed(reproducibility[key]) != resolved_seed:
                raise EvidenceError(
                    "Seed evidence conflict: {} differs from resolved seed"
                    .format(key)
                )
        expected_generator_seeds = {
            "dataloader_train_generator_seed": derive_data_loader_seed(
                resolved_seed, "train"
            ),
            "dataloader_validation_generator_seed": derive_data_loader_seed(
                resolved_seed, "validation"
            ),
        }
        for key, expected in expected_generator_seeds.items():
            if validate_seed(reproducibility[key]) != expected:
                raise EvidenceError(
                    "DataLoader seed evidence conflict for {}".format(key)
                )
        if validate_seed(int(reproducibility["PYTHONHASHSEED"])) != resolved_seed:
            raise EvidenceError("PYTHONHASHSEED differs from resolved seed")
        if reproducibility["status"] != "complete":
            raise EvidenceError("Reproducibility evidence is not complete")
        for key in ("dataloader_worker_seed_strategy", "sampler_seed_strategy"):
            if reproducibility[key] in (NOT_RECORDED, MISSING_EVIDENCE, ""):
                raise EvidenceError("{} is missing".format(key))
        if reproducibility["cudnn_deterministic"] is not True:
            raise EvidenceError("cudnn_deterministic must be true")
        if reproducibility["cudnn_benchmark"] is not False:
            raise EvidenceError("cudnn_benchmark must be false")
        for required in ("environment.json", "environment_packages.txt", "dataset_manifest.json", "model_manifest.json"):
            if not (run_dir / required).is_file():
                raise EvidenceError("Required run evidence is missing: {}".format(required))
        environment = read_json(run_dir / "environment.json")
        if environment.get("git_branch") != manifest["branch"]:
            raise EvidenceError("Environment branch evidence does not match run manifest")
        if environment.get("git_commit") != manifest["commit_id"]:
            raise EvidenceError("Environment commit evidence does not match run manifest")
        training_hash_seed = environment.get(
            "training_subprocess_PYTHONHASHSEED", NOT_RECORDED
        )
        if training_hash_seed != NOT_RECORDED:
            if validate_seed(int(training_hash_seed)) != resolved_seed:
                raise EvidenceError(
                    "Training subprocess PYTHONHASHSEED differs from resolved seed"
                )
        if not environment.get("gpus") or _gpu_label(environment) == NOT_RECORDED:
            raise EvidenceError("GPU evidence is missing")
        dataset_manifest = read_json(run_dir / "dataset_manifest.json")
        if not dataset_manifest.get("dataset_manifest_sha256"):
            raise EvidenceError("Dataset manifest hash is missing")
        if manifest.get("dataset_manifest_sha256", NOT_RECORDED) not in (
                NOT_RECORDED, dataset_manifest["dataset_manifest_sha256"]):
            raise EvidenceError("Dataset manifest signature differs")
        if manifest.get("experiment_family") in STRICT_EVIDENCE_FAMILIES:
            validate_strict_manifest_preflight(manifest)
            if len(str(manifest["protocol_signature_sha256"])) != 64:
                raise EvidenceError("Protocol signature is invalid")
            if len(str(manifest["implementation_signature_sha256"])) != 64:
                raise EvidenceError("Implementation signature is invalid")
            protocol_configuration = json.loads(json.dumps(resolved_cfg))
            if manifest["run_kind"] == "smoke":
                for key in (
                        "MAX_EPOCHS", "CHECKPOINT_PERIOD", "EVAL_PERIOD"):
                    protocol_configuration["SOLVER"][key] = source_cfg[
                        "SOLVER"
                    ][key]
            if config_protocol_signature(protocol_configuration) != manifest[
                    "protocol_signature_sha256"]:
                raise EvidenceError("Protocol signature/config conflict")
            if git_implementation_signature(
                    repo_root, manifest["commit_id"]
            ) != manifest["implementation_signature_sha256"]:
                raise EvidenceError("Implementation signature/Git conflict")
            if _git_output(
                    repo_root, ["rev-parse", "{}^{{tree}}".format(
                        manifest["commit_id"]
                    )]
            ).lower() != manifest["commit_tree"]:
                raise EvidenceError("Commit tree evidence differs")
        if int(manifest.get("schema_version", 1)) >= 4:
            console_path = run_dir / "console.log"
            if not console_path.is_file() or console_path.stat().st_size <= 0:
                raise EvidenceError("Console log is missing or empty")
            console_sha = sha256_file(console_path)
            if console_sha != manifest.get("console_log_sha256"):
                raise EvidenceError("Console log SHA256 differs")
        log_info = parse_training_log(output_dir / "log.txt")
        required_iteration_source = manifest.get(
            "required_global_iteration_source", NOT_RECORDED
        )
        if (required_iteration_source != NOT_RECORDED
                and log_info["global_iteration_source"] !=
                required_iteration_source):
            raise EvidenceError(
                "Run requires global_iteration_source={} but training log "
                "provides {}".format(
                    required_iteration_source,
                    log_info["global_iteration_source"],
                )
            )
        source_config_text = source_copy.read_text(
            encoding="utf-8", errors="replace"
        ).replace("\r\n", "\n").strip()
        normalized_log_text = log_info["raw_text"].replace("\r\n", "\n")
        if source_config_text not in normalized_log_text:
            raise EvidenceError(
                "Training log does not contain the exact source config"
            )
        modules = manifest.get("modules", {})
        if modules.get("camera_aware_triplet") and not log_info["has_camera_aware_loss"]:
            raise EvidenceError("Enabled camera-aware loss is absent from log")
        if modules.get("cross_camera_positive") and not log_info["has_cross_camera_positive_loss"]:
            raise EvidenceError("Enabled cross-camera positive loss is absent from log")
        if modules.get("part_correspondence_consistency"):
            if not log_info["has_pcc_loss"]:
                raise EvidenceError("Enabled PCC loss is absent from log")
            alignment_mode = manifest.get("alignment_mode", manifest.get(
                "pcc_mode", NOT_RECORDED
            ))
            if alignment_mode == "fixed_index":
                if not log_info["pcc_epoch_summaries"]:
                    raise EvidenceError(
                        "Enabled fixed-index PCC has no epoch pair statistics"
                    )
            elif alignment_mode == "hard_shortest_path":
                if not log_info["has_hard_alignment_loss"]:
                    raise EvidenceError(
                        "Hard alignment loss evidence is absent from log"
                    )
                if not log_info["hard_alignment_epoch_summaries"]:
                    raise EvidenceError(
                        "Hard alignment has no epoch pair statistics"
                    )
                required_hard = (
                    "hard_alignment_loss", "valid_alignment_pair_count",
                    "mean_hard_path_cost", "mean_path_absolute_offset",
                )
                missing_hard = [
                    field for field in required_hard
                    if log_info[field] in (
                        NOT_RECORDED, MISSING_EVIDENCE, None, ""
                    )
                ]
                if missing_hard:
                    raise EvidenceError(
                        "Hard alignment evidence lacks {}".format(missing_hard)
                    )
                parts = int(manifest.get("pcc_parts", 0))
                if parts != 6:
                    raise EvidenceError("Hard alignment evidence requires K=6")
                expected_loss = float(
                    log_info["mean_hard_path_cost"]
                ) / float(2 * parts - 1)
                if abs(float(log_info["hard_alignment_loss"])
                       - expected_loss) > 2e-5:
                    raise EvidenceError(
                        "Hard alignment loss is inconsistent with raw path cost"
                    )
                offset = float(log_info["mean_path_absolute_offset"])
                if not 0.0 <= offset <= float(parts - 1):
                    raise EvidenceError(
                        "Hard alignment path offset is outside valid bounds"
                    )
            elif alignment_mode == "soft_min":
                if log_info["hard_alignment_epoch_summaries"]:
                    raise EvidenceError(
                        "Soft alignment must not record a unique Hard path"
                    )
                if not log_info["has_soft_alignment_loss"]:
                    raise EvidenceError(
                        "Soft alignment loss evidence is absent from log"
                    )
                if not log_info["soft_alignment_epoch_summaries"]:
                    raise EvidenceError(
                        "Soft alignment has no epoch pair statistics"
                    )
                required_soft = (
                    "soft_alignment_loss", "valid_alignment_pair_count",
                    "mean_soft_path_cost", "alignment_temperature",
                )
                missing_soft = [
                    field for field in required_soft
                    if log_info[field] in (
                        NOT_RECORDED, MISSING_EVIDENCE, None, ""
                    )
                ]
                if missing_soft:
                    raise EvidenceError(
                        "Soft alignment evidence lacks {}".format(
                            missing_soft
                        )
                    )
                parts = int(manifest.get("pcc_parts", 0))
                if parts != 6:
                    raise EvidenceError("Soft alignment evidence requires K=6")
                if int(log_info["valid_alignment_pair_count"]) <= 0:
                    raise EvidenceError(
                        "Soft alignment cannot succeed with zero valid pairs"
                    )
                soft_loss = float(log_info["soft_alignment_loss"])
                soft_cost = float(log_info["mean_soft_path_cost"])
                logged_tau = float(log_info["alignment_temperature"])
                configured_tau = float(manifest["alignment_temperature"])
                if not all(math.isfinite(value) for value in (
                        soft_loss, soft_cost, logged_tau, configured_tau)):
                    raise EvidenceError("Soft alignment evidence is non-finite")
                if logged_tau <= 0 or configured_tau <= 0:
                    raise EvidenceError(
                        "Soft alignment temperature must be positive"
                    )
                if abs(logged_tau - configured_tau) > 1e-12:
                    raise EvidenceError(
                        "Logged/configured alignment temperatures differ"
                    )
                expected_loss = soft_cost / float(2 * parts - 1)
                if abs(soft_loss - expected_loss) > 2e-5:
                    raise EvidenceError(
                        "Soft alignment loss is inconsistent with raw cost"
                    )
            elif alignment_mode == "windowed_soft_min":
                if log_info["hard_alignment_epoch_summaries"]:
                    raise EvidenceError(
                        "Windowed soft alignment must not record a unique Hard path"
                    )
                if log_info["soft_alignment_epoch_summaries"]:
                    raise EvidenceError(
                        "Windowed soft alignment must not use unrestricted summaries"
                    )
                if not log_info["has_windowed_soft_alignment_loss"]:
                    raise EvidenceError(
                        "Windowed soft alignment loss evidence is absent from log"
                    )
                if not log_info["windowed_soft_alignment_epoch_summaries"]:
                    raise EvidenceError(
                        "Windowed soft alignment has no epoch pair statistics"
                    )
                required_windowed = (
                    "windowed_soft_alignment_loss",
                    "valid_alignment_pair_count",
                    "mean_windowed_soft_path_cost",
                    "windowed_alignment_temperature", "alignment_window",
                )
                missing_windowed = [
                    field for field in required_windowed
                    if log_info[field] in (
                        NOT_RECORDED, MISSING_EVIDENCE, None, ""
                    )
                ]
                if missing_windowed:
                    raise EvidenceError(
                        "Windowed soft alignment evidence lacks {}".format(
                            missing_windowed
                        )
                    )
                parts = int(manifest.get("pcc_parts", 0))
                if parts != 6:
                    raise EvidenceError(
                        "Windowed soft alignment evidence requires K=6"
                    )
                if int(log_info["valid_alignment_pair_count"]) <= 0:
                    raise EvidenceError(
                        "Windowed soft alignment cannot succeed with zero pairs"
                    )
                configured_window = manifest.get("alignment_window")
                if int(log_info["alignment_window"]) != int(configured_window):
                    raise EvidenceError(
                        "Logged/configured alignment windows differ"
                    )
                windowed_loss = float(
                    log_info["windowed_soft_alignment_loss"]
                )
                windowed_cost = float(
                    log_info["mean_windowed_soft_path_cost"]
                )
                logged_tau = float(log_info["windowed_alignment_temperature"])
                configured_tau = float(manifest["alignment_temperature"])
                if not all(math.isfinite(value) for value in (
                        windowed_loss, windowed_cost, logged_tau,
                        configured_tau)):
                    raise EvidenceError(
                        "Windowed soft alignment evidence is non-finite"
                    )
                if logged_tau <= 0 or configured_tau <= 0:
                    raise EvidenceError(
                        "Windowed soft alignment temperature must be positive"
                    )
                if abs(logged_tau - configured_tau) > 1e-12:
                    raise EvidenceError(
                        "Logged/configured alignment temperatures differ"
                    )
                expected_loss = windowed_cost / float(2 * parts - 1)
                if abs(windowed_loss - expected_loss) > 2e-5:
                    raise EvidenceError(
                        "Windowed soft alignment loss is inconsistent with raw cost"
                    )
            else:
                raise EvidenceError(
                    "Unsupported alignment mode in run manifest: {!r}"
                    .format(alignment_mode)
                )
            if int(log_info["valid_pcc_pair_count"]) <= 0:
                raise EvidenceError(
                    "Alignment cannot be registered success with zero pairs"
                )
        validation_source = output_dir / "validation_history.jsonl"
        if validation_source.is_file():
            copy_file_atomic(validation_source, run_dir / "validation_history.jsonl")
        else:
            write_validation_history(
                run_dir / "validation_history.jsonl", log_info["validations"]
            )
        validation_rows = read_validation_history(run_dir / "validation_history.jsonl")
        cross_validate_log_metrics(log_info, validation_rows)
        best = select_best_validation(validation_rows)
        checkpoint_rows, selected = build_checkpoint_manifest(
            output_dir,
            validation_rows,
            best["epoch"],
            run_dir / "checkpoint_manifest.tsv",
        )
        if run_analyses:
            _run_analysis_tools(repo_root, run_dir, manifest, selected)
        anchor = validate_anchor_coverage(read_json(run_dir / "anchor_coverage.json"))
        distance = validate_distance_distribution(
            read_json(run_dir / "distance_distribution.json")
        )
        efficiency_path = run_dir / "efficiency_profile.json"
        efficiency = read_json(efficiency_path) if efficiency_path.is_file() else efficiency_placeholder()
        if not efficiency_path.is_file():
            atomic_write_json(efficiency_path, efficiency)
        model_manifest = read_json(run_dir / "model_manifest.json")
        if identity["alignment_mode"] in ("soft_min", "windowed_soft_min"):
            if model_manifest.get(
                    "multigranular_feature_signature_sha256"
            ) != manifest["multigranular_feature_signature_sha256"]:
                raise EvidenceError(
                    "Model/run feature signature SHA256 evidence differs"
                )
            if model_manifest.get("parent_branch") != manifest.get(
                    "parent_branch"):
                raise EvidenceError("Model/run parent branch evidence differs")
            if model_manifest.get("parent_commit") != manifest.get(
                    "parent_commit"):
                raise EvidenceError("Model/run parent commit evidence differs")
            if int(manifest.get("schema_version", 1)) >= 4:
                for field in (
                        "feature_reference_commit",
                        "feature_reference_signature_sha256",
                        "current_feature_signature_sha256",
                        "feature_compatibility_status",
                        "feature_compatibility_evidence_sha256"):
                    if model_manifest.get(field) != manifest.get(field):
                        raise EvidenceError(
                            "Model/run feature evidence differs for {}".format(
                                field
                            )
                        )
        model_manifest.update({
            "total_params": efficiency.get("total_params", NOT_RECORDED),
            "trainable_params": efficiency.get("trainable_params", NOT_RECORDED),
            "FLOPs": efficiency.get("FLOPs", NOT_RECORDED),
            "MACs": efficiency.get("MACs", NOT_RECORDED),
            "selected_checkpoint_sha256": selected["sha256"],
            "efficiency_profile": "efficiency_profile.json",
            "feature_compatibility_evidence_path": manifest.get(
                "feature_compatibility_evidence_path", NOT_RECORDED
            ),
            "feature_compatibility_evidence_size_bytes": (
                (run_dir / "feature_compatibility.json").stat().st_size
                if (run_dir / "feature_compatibility.json").is_file()
                else NOT_RECORDED
            ),
            "feature_compatibility_evidence_sha256": manifest.get(
                "feature_compatibility_evidence_sha256", NOT_RECORDED
            ),
        })
        atomic_write_json(run_dir / "model_manifest.json", model_manifest)
        runtime_seconds = float(status["training_runtime_seconds"])
        alignment_mode = manifest.get("alignment_mode", NOT_APPLICABLE)
        if alignment_mode == "hard_shortest_path":
            mean_fixed_index_part_distance = NOT_APPLICABLE
            hard_alignment_loss = log_info["hard_alignment_loss"]
            valid_alignment_pair_count = log_info[
                "valid_alignment_pair_count"
            ]
            mean_hard_path_cost = log_info["mean_hard_path_cost"]
            mean_path_absolute_offset = log_info[
                "mean_path_absolute_offset"
            ]
            soft_alignment_loss = NOT_APPLICABLE
            mean_soft_path_cost = NOT_APPLICABLE
        elif alignment_mode == "soft_min":
            mean_fixed_index_part_distance = NOT_APPLICABLE
            hard_alignment_loss = NOT_APPLICABLE
            valid_alignment_pair_count = log_info[
                "valid_alignment_pair_count"
            ]
            mean_hard_path_cost = NOT_APPLICABLE
            mean_path_absolute_offset = NOT_APPLICABLE
            soft_alignment_loss = log_info["soft_alignment_loss"]
            mean_soft_path_cost = log_info["mean_soft_path_cost"]
        elif alignment_mode == "windowed_soft_min":
            mean_fixed_index_part_distance = NOT_APPLICABLE
            hard_alignment_loss = NOT_APPLICABLE
            valid_alignment_pair_count = log_info[
                "valid_alignment_pair_count"
            ]
            mean_hard_path_cost = NOT_APPLICABLE
            mean_path_absolute_offset = NOT_APPLICABLE
            # Existing columns denote unrestricted Soft-Min only.  The
            # windowed values remain in the dedicated sensitivity table.
            soft_alignment_loss = NOT_APPLICABLE
            mean_soft_path_cost = NOT_APPLICABLE
        elif alignment_mode == "fixed_index":
            mean_fixed_index_part_distance = log_info[
                "mean_fixed_index_part_distance"
            ]
            hard_alignment_loss = NOT_APPLICABLE
            valid_alignment_pair_count = log_info["valid_pcc_pair_count"]
            mean_hard_path_cost = NOT_APPLICABLE
            mean_path_absolute_offset = NOT_APPLICABLE
            soft_alignment_loss = NOT_APPLICABLE
            mean_soft_path_cost = NOT_APPLICABLE
        else:
            mean_fixed_index_part_distance = log_info[
                "mean_fixed_index_part_distance"
            ]
            hard_alignment_loss = NOT_APPLICABLE
            valid_alignment_pair_count = NOT_APPLICABLE
            mean_hard_path_cost = NOT_APPLICABLE
            mean_path_absolute_offset = NOT_APPLICABLE
            soft_alignment_loss = NOT_APPLICABLE
            mean_soft_path_cost = NOT_APPLICABLE
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "selection_rule": "highest Rank-1, then highest mAP, then latest epoch",
            "best_epoch": int(best["epoch"]),
            "selected_epoch": int(best["epoch"]),
            "selected_global_iteration": int(best["global_iteration"]),
            "selected_global_iteration_source": best.get(
                "global_iteration_source", "legacy_validation_history"
            ),
            "rank1_percent": float(best["rank1_percent"]),
            "rank5_percent": float(best["rank5_percent"]),
            "rank10_percent": float(best["rank10_percent"]),
            "map_percent": float(best["map_percent"]),
            "checkpoint": selected["path"],
            "checkpoint_sha256": selected["sha256"],
            "log_path": log_info["path"],
            "log_sha256": log_info["sha256"],
            "runtime_seconds": runtime_seconds,
            "valid_pcc_pair_count": log_info["valid_pcc_pair_count"],
            "mean_fixed_index_part_distance": mean_fixed_index_part_distance,
            "hard_alignment_loss": hard_alignment_loss,
            "valid_alignment_pair_count": valid_alignment_pair_count,
            "mean_hard_path_cost": mean_hard_path_cost,
            "mean_path_absolute_offset": mean_path_absolute_offset,
            "soft_alignment_loss": soft_alignment_loss,
            "mean_soft_path_cost": mean_soft_path_cost,
            "run_dir": str(run_dir),
        }
        atomic_write_json(run_dir / "metrics_summary.json", metrics)
        analysis_summary = {
            "schema_version": SCHEMA_VERSION,
            "distance_distribution": "complete",
            "anchor_coverage": "complete",
            "efficiency_profile": efficiency.get("status", NOT_RECORDED),
            "analysis_commit": validate_git_runtime_state(
                repo_root,
                run_dir,
                expected_branch=manifest["branch"],
                expected_commit=manifest["commit_id"],
            )["commit"] if verify_git else manifest["commit_id"],
            "analysis_time": utc_now(),
            "source_checkpoint_sha256": selected["sha256"],
            "training_evidence_separate_from_post_hoc_analysis": True,
        }
        atomic_write_json(run_dir / "analysis_summary.json", analysis_summary)
        rows = _prepare_table_rows(
            manifest, metrics, environment, efficiency, distance, anchor
        )
        (common, lambda_row, same_row, caat_row, distance_row, anchor_row,
         pcc_row, alignment_row, windowed_row) = rows
        final_status = dict(status)
        final_status.update({
            "status": "success",
            "phase": "complete",
            "end_time": status.get("training_end_time", utc_now()),
            "best_epoch": metrics["best_epoch"],
            "selected_epoch": metrics["selected_epoch"],
            "updated_at_utc": utc_now(),
        })
        previous_failure = {
            field: status[field] for field in (
                "error", "error_type", "traceback"
            ) if field in status
        }
        if previous_failure:
            history = list(manifest.get("finalization_provenance", []))
            history.append({
                "recorded_at_utc": utc_now(),
                "event": "successful_finalization_after_prior_failure",
                "prior_failure": previous_failure,
            })
            manifest["finalization_provenance"] = history
        for field in ("error", "error_type", "traceback"):
            final_status.pop(field, None)
        artifact_rows = _artifact_rows(
            run_dir, log_info, checkpoint_rows, final_status
        )
        write_tsv(
            run_dir / "artifact_hashes.tsv",
            ("artifact_type", "path", "size_bytes", "sha256"),
            artifact_rows,
        )
        artifact_manifest_path = run_dir / "artifact_hashes.tsv"
        manifest.update({
            "training_log_path": log_info["path"],
            "training_log_size_bytes": log_info["size_bytes"],
            "training_log_sha256": log_info["sha256"],
            "artifact_manifest_path": normalized_path(
                artifact_manifest_path.resolve()
            ),
            "artifact_manifest_sha256": sha256_file(
                artifact_manifest_path
            ),
            "artifact_manifest_size_bytes": artifact_manifest_path.stat().st_size,
        })
        atomic_write_json(run_dir / "run_manifest.json", manifest)
        artifact_rows.extend([
            {
                "artifact_type": "run_manifest",
                "path": normalized_path(
                    (run_dir / "run_manifest.json").resolve()
                ),
                "size_bytes": (run_dir / "run_manifest.json").stat().st_size,
                "sha256": sha256_file(run_dir / "run_manifest.json"),
            },
            {
                "artifact_type": "artifact_manifest",
                "path": manifest["artifact_manifest_path"],
                "size_bytes": artifact_manifest_path.stat().st_size,
                "sha256": manifest["artifact_manifest_sha256"],
            },
        ])
        rows = _prepare_table_rows(
            manifest, metrics, environment, efficiency, distance, anchor
        )
        (common, lambda_row, same_row, caat_row, distance_row, anchor_row,
         pcc_row, alignment_row, windowed_row) = rows
        tables_dir = records_root / "tables"
        main_rows = _read_csv(tables_dir / "main_results.csv")
        if manifest["run_kind"] == "formal":
            main_rows = upsert_csv(
                tables_dir / "main_results.csv", MAIN_FIELDS, common
            )
            if _lambda_table_eligible(manifest):
                upsert_csv(
                    tables_dir / "lambda_sensitivity.csv", LAMBDA_FIELDS,
                    lambda_row,
                )
            if identity["variant"] in (
                    "baseline", "cross_camera_positive",
                    "same_camera_positive"):
                upsert_csv(
                    tables_dir / "same_camera_positive_ablation.csv",
                    SAME_CAMERA_FIELDS,
                    same_row,
                )
            upsert_csv(tables_dir / "caat_ablation.csv", CAAT_FIELDS, caat_row)
            upsert_csv(
                tables_dir / "distance_distribution.csv", DISTANCE_FIELDS,
                distance_row,
            )
            upsert_csv(
                tables_dir / "anchor_coverage.csv", ANCHOR_FIELDS, anchor_row
            )
            if manifest.get("pcc_enabled", False):
                upsert_csv(
                    tables_dir / "pcc_ablation.csv", PCC_FIELDS, pcc_row
                )
                upsert_csv(
                    tables_dir / "alignment_ablation.csv", ALIGNMENT_FIELDS,
                    alignment_row,
                )
            if _soft_alignment_lambda_table_eligible(manifest):
                upsert_csv(
                    tables_dir / "soft_alignment_lambda_sensitivity.csv",
                    SOFT_ALIGNMENT_LAMBDA_FIELDS,
                    {
                        "schema_version": common["schema_version"],
                        "run_id": common["run_id"],
                        "experiment_id": common["experiment_id"],
                        "run_kind": common["run_kind"],
                        "status": common["status"],
                        "alignment_mode": common["alignment_mode"],
                        "alignment_temperature": common[
                            "alignment_temperature"
                        ],
                        "pcc_lambda": common["pcc_lambda"],
                        "alignment_lambda": common["pcc_lambda"],
                        "parts": common["pcc_parts"],
                        "seed": common["seed"],
                        "rank1": common["rank1"],
                        "rank5": common["rank5"],
                        "rank10": common["rank10"],
                        "map": common["map"],
                        "best_epoch": common["best_epoch"],
                        "runtime": common["runtime_seconds"],
                        "checkpoint": common["checkpoint"],
                        "checkpoint_sha256": common[
                            "checkpoint_sha256"
                        ],
                        "commit": common["commit"],
                        "output_dir": common["output_dir"],
                    },
                )
            if _windowed_soft_alignment_table_eligible(manifest):
                upsert_csv(
                    tables_dir / "windowed_soft_alignment_sensitivity.csv",
                    WINDOWED_SOFT_ALIGNMENT_FIELDS,
                    windowed_row,
                )
            for table_name in TABLE_SCHEMAS:
                csv_to_markdown(
                    tables_dir / "{}.csv".format(table_name),
                    tables_dir / "{}.md".format(table_name),
                )
        run_row = {
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"], "experiment_id": manifest["experiment_id"],
            "experiment_family": manifest["experiment_family"],
            "run_kind": manifest["run_kind"],
            "method": manifest["method"], "dataset": manifest["dataset"],
            "method_family": manifest.get("method_family", NOT_RECORDED),
            "method_variant": manifest.get("method_variant", NOT_RECORDED),
            "branch": manifest["branch"], "commit_id": manifest["commit_id"],
            "parent_branch": manifest.get("parent_branch", NOT_RECORDED),
            "parent_commit": manifest.get("parent_commit", NOT_RECORDED),
            "config_file": manifest["config_file"], "seed": manifest["seed"],
            "lambda": manifest["lambda"],
            "cross_camera_positive_lambda": manifest.get(
                "cross_camera_positive_lambda", manifest["lambda"]
            ),
            "pcc_lambda": manifest.get("pcc_lambda", NOT_RECORDED),
            "pcc_enabled": manifest.get("pcc_enabled", False),
            "pcc_parts": manifest.get("pcc_parts", NOT_RECORDED),
            "pcc_mode": manifest.get("pcc_mode", NOT_RECORDED),
            "alignment_strategy": manifest.get(
                "alignment_strategy", NOT_RECORDED
            ),
            "alignment_mode": manifest.get("alignment_mode", NOT_RECORDED),
            "alignment_temperature": manifest.get(
                "alignment_temperature", NOT_RECORDED
            ),
            "gating_mode": manifest.get("gating_mode", NOT_RECORDED),
            "gating_temperature": manifest.get(
                "gating_temperature", NOT_RECORDED
            ),
            "multigranular_feature_signature": manifest.get(
                "multigranular_feature_signature", NOT_RECORDED
            ),
            "multigranular_feature_signature_sha256": manifest.get(
                "multigranular_feature_signature_sha256", NOT_RECORDED
            ),
            "baseline": manifest.get("baseline", NOT_RECORDED),
            "margin": manifest["margin"],
            "mode": manifest["mode"], "GPU": _gpu_label(environment),
            "start_time": manifest["start_time"], "end_time": final_status["end_time"],
            "runtime": runtime_seconds, "best_epoch": metrics["best_epoch"],
            "selected_epoch": metrics["selected_epoch"],
            "Rank-1": common["rank1"], "Rank-5": common["rank5"],
            "Rank-10": common["rank10"], "mAP": common["map"],
            "checkpoint": selected["path"],
            "checkpoint_sha256": selected["sha256"],
            "log_path": log_info["path"], "log_sha256": log_info["sha256"],
            "output_dir": manifest["output_dir"], "status": "success",
            "valid_pcc_pair_count": metrics["valid_pcc_pair_count"],
            "mean_fixed_index_part_distance": metrics[
                "mean_fixed_index_part_distance"
            ],
            "hard_alignment_loss": metrics["hard_alignment_loss"],
            "valid_alignment_pair_count": metrics[
                "valid_alignment_pair_count"
            ],
            "mean_hard_path_cost": metrics["mean_hard_path_cost"],
            "mean_path_absolute_offset": metrics[
                "mean_path_absolute_offset"
            ],
            "soft_alignment_loss": metrics["soft_alignment_loss"],
            "mean_soft_path_cost": metrics["mean_soft_path_cost"],
            "notes": manifest["notes"],
        }
        # Rebuild from the persisted schema-v4 manifests so the all-runs row
        # cannot omit console/config/feature/artifact evidence fields.
        atomic_write_json(run_dir / "run_status.json", final_status)
        run_row = _run_row_from_manifest(run_dir, manifest, final_status)
        upsert_csv(records_root / "runs.csv", RUN_FIELDS, run_row)
        evidence_rows = []
        for row in artifact_rows:
            evidence_rows.append({
                "schema_version": SCHEMA_VERSION,
                "run_id": manifest["run_id"],
                "run_kind": manifest["run_kind"],
                "artifact_type": row["artifact_type"],
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            })
        for row in evidence_rows:
            upsert_tsv(
                records_root / "evidence_manifest.tsv",
                EVIDENCE_FIELDS,
                row,
                key_fields=("run_id", "path"),
            )
        update_experiments_markdown(experiments_path, records_root)
        return {"manifest": manifest, "metrics": metrics, "status": final_status}
    except BaseException as error:
        failure_status = "incomplete"
        try:
            _remove_run_from_formal_tables(
                records_root, manifest.get("run_id", run_dir.name)
            )
        except Exception:
            pass
        try:
            current_status = read_json(run_dir / "run_status.json")
            exit_code = current_status.get("training_exit_code")
            if exit_code not in (0, "0", NOT_RECORDED):
                failure_status = "failed"
        except Exception:
            pass
        record_run_failure(run_dir, error, status=failure_status)
        raise
