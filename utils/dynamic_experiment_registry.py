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

from utils.dynamic_gating_evidence import (
    DYNAMIC_GATING_SELECTION_RULE,
    read_gating_epoch_records,
    validate_dynamic_gating_evidence,
)
from utils.experiment_schema import (
    AUTO_CHECKPOINTS_END,
    AUTO_CHECKPOINTS_START,
    AUTO_RESULTS_END,
    AUTO_RESULTS_START,
    AUTO_RUNS_END,
    AUTO_RUNS_START,
    ALIGNMENT_PCC_FIELDS,
    CHECKPOINT_FIELDS,
    EVIDENCE_FIELDS,
    FORMAL_FIELDS,
    GATING_STAT_FIELDS,
    LEGACY_DYNAMIC_MARKERS,
    MISSING_EVIDENCE,
    NOT_APPLICABLE,
    NOT_RECORDED,
    RUN_FIELDS,
    SCHEMA_VERSION,
    migrate_row,
)
from utils.multigranularity_signatures import (
    STATIC_BASELINE_BRANCH,
    STATIC_BASELINE_SHA,
)


UNIFIED_SCHEMA_VERSION = SCHEMA_VERSION


class DynamicExperimentEvidenceError(RuntimeError):
    pass


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


_PROTOCOL_IGNORED_PATHS = (
    "SOLVER.MAX_EPOCHS", "SOLVER.CHECKPOINT_PERIOD",
    "SOLVER.EVAL_PERIOD", "OUTPUT_DIR",
)


def _plain_value(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {
            str(key): _plain_value(child)
            for key, child in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_plain_value(child) for child in value]
    return value


def _remove_dotted(mapping, dotted):
    parts = dotted.split(".")
    parent = mapping
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            return
        parent = parent[part]
    if isinstance(parent, dict):
        parent.pop(parts[-1], None)


def candidate_protocol_payload(configuration):
    """Canonical formal/smoke protocol with only declared smoke fields removed."""
    payload = _plain_value(configuration)
    if not isinstance(payload, dict):
        raise DynamicExperimentEvidenceError("Resolved configuration must be a mapping")
    for dotted in _PROTOCOL_IGNORED_PATHS:
        _remove_dotted(payload, dotted)
    return payload


def candidate_protocol_signature(configuration):
    canonical = json.dumps(
        candidate_protocol_payload(configuration), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence_only_git_path(path):
    normalized = str(path).replace("\\", "/")
    if normalized == "EXPERIMENTS.md":
        return True
    if normalized in (
            "experiment_records/runs.csv",
            "experiment_records/evidence_manifest.tsv"):
        return True
    if normalized.startswith("experiment_records/runs/"):
        return True
    if normalized.startswith("experiment_records/tables/") and (
            normalized.endswith(".csv") or normalized.endswith(".md")):
        return True
    return False


def implementation_signature(repo_root, revision):
    """Hash every tracked implementation blob except recorder result products."""
    listing = _git(
        repo_root, ["ls-tree", "-r", "--full-tree", str(revision)]
    )
    rows = []
    for line in listing.splitlines():
        metadata, separator, path = line.partition("\t")
        if not separator or _evidence_only_git_path(path):
            continue
        rows.append("{}\t{}".format(metadata, path.replace("\\", "/")))
    if not rows:
        raise DynamicExperimentEvidenceError("Implementation signature has no tracked inputs")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def validate_smoke_commit_lineage(repo_root, smoke_commit, current_commit):
    if smoke_commit == current_commit:
        return []
    try:
        subprocess.check_call(
            ["git", "-C", str(Path(repo_root).resolve()), "merge-base",
             "--is-ancestor", smoke_commit, current_commit],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DynamicExperimentEvidenceError(
            "Smoke commit is not an ancestor of the formal candidate"
        ) from error
    changed = _git(
        repo_root, ["diff", "--name-only", smoke_commit, current_commit]
    ).splitlines()
    forbidden = sorted(path for path in changed if not _evidence_only_git_path(path))
    if forbidden:
        raise DynamicExperimentEvidenceError(
            "Smoke/formal commits differ outside recorded evidence: {}".format(forbidden)
        )
    return sorted(changed)


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
        "externally_sealed_artifacts": ["run_manifest", "run_status"],
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
                           started_at_utc=None,
                           candidate_protocol_signature_sha256=None,
                           implementation_signature_sha256=None):
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
    resolved_mapping = yaml.safe_load(resolved_config_text)
    protocol_signature = (
        candidate_protocol_signature_sha256
        or candidate_protocol_signature(resolved_mapping)
    )
    implementation_sha = (
        implementation_signature_sha256
        or lineage.get("implementation_signature_sha256", NOT_RECORDED)
    )
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
        "candidate_protocol_signature_sha256": protocol_signature,
        "implementation_signature_sha256": implementation_sha,
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
    if status in ("success", "failed", "incomplete", "interrupted"):
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


def _valid_authority_field(field):
    return (
        isinstance(field, str)
        and bool(field.strip())
        and "\r" not in field
        and "\n" not in field
    )


def _authority_fields(base_fields, declared_fields=(), rows=()):
    """Return fixed v5 fields followed by every authoritative extra field.

    Extra columns are sorted instead of depending on CSV or dictionary order,
    so old and future method-specific evidence cannot disappear from Markdown
    and repeated generation remains byte-for-byte deterministic.
    """
    base = tuple(field for field in base_fields if _valid_authority_field(field))
    base_set = set(base)
    extras = set()
    for field in declared_fields:
        if _valid_authority_field(field) and field not in base_set:
            extras.add(field)
    for row in rows:
        for field in row:
            if _valid_authority_field(field) and field not in base_set:
                extras.add(field)
    return base + tuple(sorted(extras))


def _upsert(rows, row, key="run_id"):
    value = str(row[key])
    result = [existing for existing in rows if str(existing.get(key)) != value]
    result.append(row)
    return sorted(result, key=lambda item: (str(item.get("start_time", "")), str(item.get(key, ""))))


def _artifact_rows(manifest, control_evidence=None):
    rows = []
    indexed = dict(manifest.get("artifacts", {}))
    indexed.update(control_evidence or {})
    for artifact_type, evidence in indexed.items():
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


def _manifest_run_row(manifest, control_evidence=None):
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
    control_evidence = control_evidence or {}
    run_manifest = control_evidence.get("run_manifest", {})
    run_status = control_evidence.get("run_status", {})
    metrics = manifest.get("metrics", {})
    statistics = manifest.get("gating_statistics", {})
    row = {
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "experiment_family": manifest.get("experiment_family", NOT_RECORDED),
        "evidence_id": manifest.get("evidence_id", manifest["run_id"]),
        "run_id": manifest["run_id"], "run_kind": manifest["run_kind"],
        "status": manifest["status"], "method_family": manifest["method_family"],
        "method_variant": manifest["method_variant"],
        "method": manifest.get("method", manifest["method_variant"]),
        "dataset": manifest.get("dataset", NOT_RECORDED),
        "baseline": manifest.get("baseline", NOT_RECORDED),
        "margin": manifest.get("margin", NOT_RECORDED),
        "mode": manifest.get("mode", NOT_RECORDED),
        "lambda": manifest.get("lambda", NOT_APPLICABLE),
        "cross_camera_positive_lambda": manifest.get(
            "cross_camera_positive_lambda", NOT_APPLICABLE
        ),
        "branch": manifest["branch"],
        "commit": manifest["commit"], "parent_branch": manifest["parent_branch"],
        "parent_commit": manifest["parent_commit"], "merge_base": manifest["merge_base"],
        "candidate_protocol_signature_sha256": manifest.get(
            "candidate_protocol_signature_sha256", NOT_RECORDED
        ),
        "implementation_signature_sha256": manifest.get(
            "implementation_signature_sha256", NOT_RECORDED
        ),
        "feature_reference_commit": manifest["feature_reference_commit"],
        "multigranular_feature_signature": manifest.get(
            "multigranular_feature_signature", NOT_RECORDED
        ),
        "multigranular_feature_signature_sha256": manifest.get(
            "multigranular_feature_signature_sha256",
            manifest["current_feature_signature_sha256"],
        ),
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
    for field in ALIGNMENT_PCC_FIELDS:
        row.setdefault(field, NOT_APPLICABLE)
    for prefix, item in (
            ("source_config", source), ("resolved_config", resolved),
            ("training_log", training), ("console_log", console),
            ("selected_checkpoint", checkpoint), ("checkpoint_manifest", checkpoint_manifest),
            ("artifact_manifest", artifact_manifest), ("reproducibility", reproducibility),
            ("run_evidence_manifest", run_evidence_manifest),
            ("run_manifest", run_manifest), ("run_status", run_status),
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


def _markdown_run_ids(section):
    table_lines = [
        line for line in section.splitlines() if line.lstrip().startswith("|")
    ]
    if len(table_lines) <= 2:
        return set()
    headers = [cell.strip() for cell in table_lines[0].strip().strip("|").split("|")]
    if "run_id" not in headers:
        raise DynamicExperimentEvidenceError(
            "Legacy generated section has data but no run_id column"
        )
    index = headers.index("run_id")
    values = set()
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(headers):
            raise DynamicExperimentEvidenceError("Malformed legacy generated row")
        if cells[index] not in ("", NOT_RECORDED):
            values.add(cells[index])
    return values


def _remove_legacy_dynamic_sections(content, authoritative_run_ids):
    migrated = content
    for start, end in LEGACY_DYNAMIC_MARKERS:
        if start not in migrated and end not in migrated:
            continue
        if migrated.count(start) != 1 or migrated.count(end) != 1:
            raise DynamicExperimentEvidenceError(
                "Legacy Dynamic Gating markers are malformed"
            )
        before, rest = migrated.split(start, 1)
        section, after = rest.split(end, 1)
        legacy_ids = _markdown_run_ids(section)
        missing = sorted(legacy_ids - set(authoritative_run_ids))
        if missing:
            raise DynamicExperimentEvidenceError(
                "Legacy Dynamic Gating rows are not migrated to v5: {}".format(missing)
            )
        migrated = before.rstrip() + "\n\n" + after.lstrip("\r\n")
    return migrated


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


def _markdown_content(experiments_path, run_rows, formal_rows, checkpoint_rows,
                      run_fields=(), formal_fields=()):
    path = Path(experiments_path)
    content = path.read_text(encoding="utf-8") if path.is_file() else "# Experiments\n"
    authoritative_ids = {
        str(row.get("run_id")) for row in list(run_rows) + list(checkpoint_rows)
        if row.get("run_id") not in (None, "", NOT_RECORDED)
    }
    content = _remove_legacy_dynamic_sections(content, authoritative_ids)
    rendered_run_fields = _authority_fields(
        RUN_FIELDS, run_fields, run_rows
    )
    rendered_formal_fields = _authority_fields(
        FORMAL_FIELDS, formal_fields, formal_rows
    )
    sections = (
        (AUTO_RUNS_START, AUTO_RUNS_END, "Run Registry / All Recorded Runs", rendered_run_fields, run_rows),
        (AUTO_RESULTS_START, AUTO_RESULTS_END, "Formal Results", rendered_formal_fields, formal_rows),
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
    status_payload = read_json(run_dir / "run_status.json")
    if status_payload.get("run_id") != manifest.get("run_id") \
            or status_payload.get("status") != manifest.get("status"):
        raise DynamicExperimentEvidenceError(
            "run_manifest.json and run_status.json disagree"
        )
    control_evidence = {
        "run_manifest": _file_evidence(run_dir / "run_manifest.json"),
        "run_status": _file_evidence(run_dir / "run_status.json"),
    }
    records_root = Path(manifest["records_root"])
    experiments_path = Path(manifest["experiments_path"])
    runs_path = records_root / "runs.csv"
    formal_path = records_root / "tables" / "main_results.csv"
    evidence_path = records_root / "evidence_manifest.tsv"
    run_rows, old_run_fields = _read_table(runs_path, ",")
    formal_rows, old_formal_fields = _read_table(formal_path, ",")
    evidence_rows, old_evidence_fields = _read_table(evidence_path, "\t")
    persisted_run_fields = _authority_fields(
        RUN_FIELDS, old_run_fields, run_rows
    )
    persisted_formal_fields = _authority_fields(
        FORMAL_FIELDS, old_formal_fields, formal_rows
    )
    persisted_evidence_fields = _authority_fields(
        EVIDENCE_FIELDS, old_evidence_fields, evidence_rows
    )
    run_rows = [migrate_row(item, RUN_FIELDS) for item in run_rows]
    formal_rows = [migrate_row(item, FORMAL_FIELDS) for item in formal_rows]
    evidence_rows = [migrate_row(item, EVIDENCE_FIELDS) for item in evidence_rows]
    row = _manifest_run_row(manifest, control_evidence)
    run_rows = _upsert(run_rows, row)
    formal_rows = [item for item in formal_rows if item.get("run_id") != row["run_id"]]
    if row["run_kind"] == "formal" and row["status"] == "success":
        formal_rows = _upsert(formal_rows, row)
    evidence_rows = [item for item in evidence_rows if item.get("run_id") != row["run_id"]]
    evidence_rows.extend(_artifact_rows(manifest, control_evidence))
    evidence_rows = sorted(evidence_rows, key=lambda item: (str(item.get("run_id", "")), str(item.get("artifact_type", "")), str(item.get("path", ""))))
    checkpoints = _checkpoint_rows(records_root, run_rows)
    markdown = _markdown_content(
        experiments_path, run_rows, formal_rows, checkpoints,
        persisted_run_fields, persisted_formal_fields,
    )
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
    run_rows, run_fields = _read_table(records_root / "runs.csv", ",")
    formal_rows, formal_fields = _read_table(
        records_root / "tables" / "main_results.csv", ","
    )
    run_rows = [migrate_row(item, RUN_FIELDS) for item in run_rows]
    formal_rows = [migrate_row(item, FORMAL_FIELDS) for item in formal_rows]
    checkpoints = _checkpoint_rows(records_root, run_rows)
    content = _markdown_content(
        experiments_path, run_rows, formal_rows, checkpoints,
        run_fields, formal_fields,
    )
    _atomic_write(experiments_path, content)
    return content


def migrate_unified_schema(path, fields, delimiter=","):
    rows, old_fields = _read_table(path, delimiter)
    # Unknown historical columns are retained after the current schema rather
    # than discarded. Missing new evidence remains explicitly not_recorded.
    merged_fields = _authority_fields(fields, old_fields, rows)
    migrated = [migrate_row(row, fields) for row in rows]
    _atomic_write(path, _render_table(migrated, merged_fields, delimiter))
    return merged_fields, migrated


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


def _require_file_evidence(evidence, label):
    if not isinstance(evidence, dict):
        raise DynamicExperimentEvidenceError("{} evidence is missing".format(label))
    path = Path(str(evidence.get("path", "")))
    if not path.is_file() or path.stat().st_size <= 0:
        raise DynamicExperimentEvidenceError("{} artifact is missing/empty".format(label))
    try:
        recorded_size = int(evidence.get("size_bytes"))
    except (TypeError, ValueError) as error:
        raise DynamicExperimentEvidenceError(
            "{} size is not recorded".format(label)
        ) from error
    if recorded_size != path.stat().st_size:
        raise DynamicExperimentEvidenceError("{} size mismatch".format(label))
    if evidence.get("sha256") != sha256_file(path):
        raise DynamicExperimentEvidenceError("{} SHA256 mismatch".format(label))
    return path


def _global_evidence_for_run(records_root, run_id):
    rows, _fields = _read_table(Path(records_root) / "evidence_manifest.tsv", "\t")
    selected = [row for row in rows if row.get("run_id") == run_id]
    if not selected:
        raise DynamicExperimentEvidenceError("Smoke is absent from global evidence manifest")
    return selected


def _validate_smoke_artifacts(records_root, run_dir, manifest, status_payload):
    run_id = manifest["run_id"]
    global_rows = _global_evidence_for_run(records_root, run_id)
    by_type = {}
    for row in global_rows:
        artifact_type = row.get("artifact_type")
        if artifact_type in by_type:
            raise DynamicExperimentEvidenceError(
                "Duplicate global artifact evidence {}".format(artifact_type)
            )
        by_type[artifact_type] = row
        _require_file_evidence(row, "global {}".format(artifact_type))
        if row.get("status") != "success":
            raise DynamicExperimentEvidenceError("Smoke artifact status is not success")

    control = {
        "run_manifest": _file_evidence(run_dir / "run_manifest.json"),
        "run_status": _file_evidence(run_dir / "run_status.json"),
    }
    required = {
        "console_log", "training_log", "source_config", "source_config_origin",
        "resolved_config", "feature_compatibility", "fusion_gating_signature",
        "reproducibility", "environment", "dataset_manifest", "model_manifest",
        "checkpoint_manifest", "selected_checkpoint", "dynamic_gating_summary",
        "gating_samples", "artifact_manifest", "run_evidence_manifest",
        "run_manifest", "run_status",
    }
    missing = sorted(required - set(by_type))
    if missing:
        raise DynamicExperimentEvidenceError(
            "Smoke global evidence is incomplete: {}".format(missing)
        )
    for artifact_type, evidence in control.items():
        row = by_type[artifact_type]
        for field in ("path", "size_bytes", "sha256"):
            if str(row.get(field)) != str(evidence.get(field)):
                raise DynamicExperimentEvidenceError(
                    "{} global sealing mismatch".format(artifact_type)
                )
    for artifact_type, evidence in manifest.get("artifacts", {}).items():
        if artifact_type not in by_type:
            raise DynamicExperimentEvidenceError(
                "Manifest artifact is absent globally: {}".format(artifact_type)
            )
        _require_file_evidence(evidence, artifact_type)
        for field in ("path", "size_bytes", "sha256"):
            if str(by_type[artifact_type].get(field)) != str(evidence.get(field)):
                raise DynamicExperimentEvidenceError(
                    "Artifact/global evidence mismatch: {}".format(artifact_type)
                )
    artifact_path = _require_file_evidence(
        manifest.get("artifact_manifest"), "artifact_manifest"
    )
    artifact_payload = read_json(artifact_path)
    if artifact_payload.get("run_id") != run_id \
            or artifact_payload.get("status") != "success":
        raise DynamicExperimentEvidenceError("Artifact manifest identity/status mismatch")
    if artifact_payload.get("externally_sealed_artifacts") != [
            "run_manifest", "run_status"]:
        raise DynamicExperimentEvidenceError("Control artifact sealing declaration is missing")
    for artifact_type, evidence in artifact_payload.get("artifacts", {}).items():
        _require_file_evidence(evidence, "artifact manifest {}".format(artifact_type))
        if artifact_type not in by_type:
            raise DynamicExperimentEvidenceError(
                "Artifact manifest entry is absent globally"
            )
    run_evidence_path = _require_file_evidence(
        manifest.get("run_evidence_manifest"), "run evidence manifest"
    )
    run_evidence_rows, _run_evidence_fields = _read_table(
        run_evidence_path, "\t"
    )
    run_evidence_types = {
        row.get("artifact_type") for row in run_evidence_rows
    }
    expected_run_evidence_types = set(artifact_payload.get("artifacts", {})) - {
        "run_evidence_manifest"
    }
    if run_evidence_types != expected_run_evidence_types:
        raise DynamicExperimentEvidenceError(
            "Per-run evidence manifest is incomplete"
        )
    for row in run_evidence_rows:
        artifact_type = row["artifact_type"]
        evidence = artifact_payload["artifacts"][artifact_type]
        for field in ("path", "size_bytes", "sha256"):
            if str(row.get(field)) != str(evidence.get(field)):
                raise DynamicExperimentEvidenceError(
                    "Per-run artifact evidence mismatch: {}".format(artifact_type)
                )
    return by_type


def validate_recorded_smoke_for_formal(
        records_root, repo_root, smoke_experiment_id, current_lineage,
        current_source_config, current_resolved_config, current_feature_evidence,
        current_protocol_signature_sha256=None,
        current_implementation_signature_sha256=None,
        selection_resolver=None, dataset_validator=None):
    """Return a fully verified smoke bound to the exact formal candidate."""
    records_root = Path(records_root)
    rows, _fields = _read_table(records_root / "runs.csv", ",")
    candidates = [
        row for row in rows
        if row.get("experiment_id") == smoke_experiment_id
        and row.get("run_kind") == "smoke"
        and row.get("status") == "success"
    ]
    if not candidates:
        raise DynamicExperimentEvidenceError(
            "Formal training requires a fully recorded successful smoke"
        )
    current_protocol_sha = (
        current_protocol_signature_sha256
        or candidate_protocol_signature(current_resolved_config)
    )
    current_implementation_sha = (
        current_implementation_signature_sha256
        or implementation_signature(repo_root, current_lineage["commit"])
    )
    current_source_sha = sha256_file(current_source_config)
    errors = []
    for row in sorted(candidates, key=lambda item: (
            str(item.get("start_time", "")), str(item.get("run_id", ""))), reverse=True):
        try:
            run_id = row.get("run_id", "")
            if not run_id or Path(run_id).name != run_id:
                raise DynamicExperimentEvidenceError("Unsafe/missing smoke run_id")
            run_dir = records_root / "runs" / run_id
            manifest = read_json(run_dir / "run_manifest.json")
            status = read_json(run_dir / "run_status.json")
            if manifest.get("schema_version") != SCHEMA_VERSION \
                    or status.get("schema_version") != SCHEMA_VERSION:
                raise DynamicExperimentEvidenceError("Formal requires a schema-v5 smoke")
            if manifest.get("run_id") != run_id or status.get("run_id") != run_id:
                raise DynamicExperimentEvidenceError("Smoke control identity mismatch")
            if manifest.get("run_kind") != "smoke" \
                    or manifest.get("status") != "success" \
                    or status.get("status") != "success":
                raise DynamicExperimentEvidenceError("Smoke control status mismatch")
            if manifest.get("branch") != current_lineage["branch"]:
                raise DynamicExperimentEvidenceError("Smoke branch mismatch")
            if manifest.get("parent_branch") != current_lineage["parent_branch"] \
                    or manifest.get("parent_commit") != current_lineage["parent_commit"] \
                    or manifest.get("merge_base") != current_lineage["merge_base"]:
                raise DynamicExperimentEvidenceError("Smoke Static parent mismatch")
            if manifest.get("seed") != 42:
                raise DynamicExperimentEvidenceError("Smoke seed mismatch")
            if manifest.get("gating_mode") != "per_sample_dynamic_gating" \
                    or manifest.get("gating_input") != "global" \
                    or float(manifest.get("gating_temperature")) != 1.0 \
                    or manifest.get("gating_normalization") != "scaled_softmax" \
                    or manifest.get("scale_order") != "2,4,6":
                raise DynamicExperimentEvidenceError("Smoke gating protocol mismatch")
            if manifest.get("feature_compatibility_status") != "compatible":
                raise DynamicExperimentEvidenceError("Smoke feature compatibility failed")

            _validate_smoke_artifacts(records_root, run_dir, manifest, status)
            source_path = _require_file_evidence(
                manifest.get("source_config"), "smoke source config"
            )
            if sha256_file(source_path) != current_source_sha:
                raise DynamicExperimentEvidenceError("Smoke source config SHA mismatch")
            resolved_path = _require_file_evidence(
                manifest.get("resolved_config"), "smoke resolved config"
            )
            resolved_mapping = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
            for dotted, expected in (
                    ("SOLVER.MAX_EPOCHS", 1),
                    ("SOLVER.CHECKPOINT_PERIOD", 1),
                    ("SOLVER.EVAL_PERIOD", 1)):
                if _nested(resolved_mapping, dotted) != expected:
                    raise DynamicExperimentEvidenceError("Smoke is not a strict 1-epoch run")
            for dotted, expected in (
                    ("SEED", 42),
                    ("MODEL.MULTI_GRANULARITY_PART_SCALES", [2, 4, 6]),
                    ("MODEL.MULTI_GRANULARITY_GATING_TAU", 1.0),
                    ("MODEL.MULTI_GRANULARITY_GATING_INPUT", "global"),
                    ("MODEL.MULTI_GRANULARITY_GATING_NORMALIZATION", "scaled_softmax")):
                if _nested(resolved_mapping, dotted) != expected:
                    raise DynamicExperimentEvidenceError(
                        "Smoke resolved gating protocol mismatch: {}".format(dotted)
                    )
            smoke_protocol_sha = candidate_protocol_signature(resolved_mapping)
            if manifest.get("candidate_protocol_signature_sha256") != smoke_protocol_sha \
                    or smoke_protocol_sha != current_protocol_sha:
                raise DynamicExperimentEvidenceError("Smoke protocol signature mismatch")

            smoke_commit = manifest.get("commit")
            validate_smoke_commit_lineage(repo_root, smoke_commit, current_lineage["commit"])
            smoke_impl = implementation_signature(repo_root, smoke_commit)
            if manifest.get("implementation_signature_sha256") != smoke_impl \
                    or smoke_impl != current_implementation_sha:
                raise DynamicExperimentEvidenceError("Smoke implementation signature mismatch")
            feature_path = _require_file_evidence(
                manifest.get("feature_compatibility"), "feature compatibility"
            )
            feature = read_json(feature_path)
            if feature.get("current_commit") != smoke_commit \
                    or current_feature_evidence.get("current_commit") != current_lineage["commit"]:
                raise DynamicExperimentEvidenceError(
                    "Feature compatibility evidence is not commit-bound"
                )
            for field in (
                    "feature_reference_commit", "feature_reference_signature_sha256",
                    "current_feature_signature_sha256", "feature_compatibility_status"):
                if feature.get(field) != current_feature_evidence.get(field):
                    raise DynamicExperimentEvidenceError(
                        "Smoke shared feature signature mismatch: {}".format(field)
                    )
            fusion_path = _require_file_evidence(
                manifest.get("fusion_gating_signature"), "gating signature"
            )
            fusion = read_json(fusion_path)
            current_gating_sha = current_feature_evidence[
                "fusion_gating_signature"
            ]["current_sha256"]
            if fusion.get("current_sha256") != current_gating_sha \
                    or manifest.get("gating_signature_sha256") != current_gating_sha:
                raise DynamicExperimentEvidenceError("Smoke gating signature mismatch")
            statistics = manifest.get("gating_statistics", {})
            if int(statistics.get("gating_sample_count", 0)) <= 0:
                raise DynamicExperimentEvidenceError("Smoke gating sample count is zero")
            metrics = manifest.get("metrics", {})
            if int(metrics.get("selected_epoch", 0)) != 1:
                raise DynamicExperimentEvidenceError("Smoke selected epoch is not 1")
            checkpoint_manifest_path = _require_file_evidence(
                manifest.get("checkpoint_manifest"), "checkpoint manifest"
            )
            checkpoint_rows, _checkpoint_fields = _read_table(
                checkpoint_manifest_path, "\t"
            )
            if not checkpoint_rows or any(
                    int(checkpoint.get("epoch", 0)) != 1
                    for checkpoint in checkpoint_rows):
                raise DynamicExperimentEvidenceError(
                    "Smoke checkpoint evidence is not strictly epoch 1"
                )
            dataset_path = _require_file_evidence(
                manifest.get("dataset_manifest"), "dataset manifest"
            )
            validate_dynamic_gating_evidence(
                _require_file_evidence(
                    manifest.get("dynamic_gating_summary"), "gating summary"
                ),
                _require_file_evidence(
                    manifest.get("gating_samples"), "gating samples"
                ),
                manifest["selected_checkpoint"]["sha256"],
                current_resolved_config, statistics, read_json(dataset_path),
                selection_resolver=selection_resolver,
                dataset_validator=dataset_validator,
            )
            return {"row": row, "manifest": manifest, "status": status}
        except (DynamicExperimentEvidenceError, OSError, ValueError, KeyError) as error:
            errors.append("{}: {}".format(row.get("run_id", NOT_RECORDED), error))
    raise DynamicExperimentEvidenceError(
        "No recorded smoke matches the current formal candidate: {}".format(errors)
    )


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
                              gating_statistics, runtime_seconds, return_code=0,
                              resolved_configuration=None,
                              selection_resolver=None, dataset_validator=None):
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
    selection_rule = DYNAMIC_GATING_SELECTION_RULE
    validation_configuration = resolved_configuration
    if validation_configuration is None:
        validation_configuration = yaml.safe_load(
            resolved_output.read_text(encoding="utf-8")
        )
    validate_dynamic_gating_evidence(
        dynamic_summary_path, gating_samples_path, selected["sha256"],
        validation_configuration, gating_statistics, dataset_manifest,
        selection_resolver=selection_resolver,
        dataset_validator=dataset_validator,
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
