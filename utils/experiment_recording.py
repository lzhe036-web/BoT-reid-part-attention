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
    resolved_config_text,
    validate_seed,
    validate_seed_evidence_chain,
)


SCHEMA_VERSION = 1
NOT_RECORDED = "not_recorded"
MISSING_EVIDENCE = "missing_evidence"
AUTO_RESULTS_START = "<!-- AUTO-EXPERIMENT-RESULTS:START -->"
AUTO_RESULTS_END = "<!-- AUTO-EXPERIMENT-RESULTS:END -->"


class EvidenceError(RuntimeError):
    pass


MAIN_FIELDS = (
    "run_id", "experiment_id", "experiment_family", "method", "dataset",
    "branch", "commit", "seed", "lambda", "cross_camera_positive_lambda",
    "pcc_lambda", "pcc_enabled", "pcc_parts", "pcc_mode",
    "alignment_strategy", "baseline", "margin", "mode", "best_epoch",
    "selected_epoch", "rank1", "rank5", "rank10", "map", "checkpoint",
    "checkpoint_sha256", "runtime_seconds", "gpu", "config", "log_path",
    "log_sha256", "output_dir", "valid_pcc_pair_count",
    "mean_fixed_index_part_distance", "multi_granularity_fusion",
    "fusion_mode", "dynamic_granularity_gating", "fusion_dimension",
    "gating_hidden_dimension", "component_count", "static_parameter_count",
    "dynamic_parameter_count", "gate_analysis_path",
    "gate_analysis_sha256", "status", "notes",
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
    "run_id", "experiment_id", "baseline", "pcc_enabled",
    "alignment_strategy", "pcc_parts", "cross_camera_positive_lambda",
    "pcc_lambda", "valid_pcc_pair_count",
    "mean_fixed_index_part_distance", "best_epoch", "rank1", "map",
    "runtime", "seed", "commit",
)
FUSION_FIELDS = (
    "run_id", "experiment_id", "method", "baseline",
    "multi_granularity_fusion", "fusion_mode",
    "dynamic_granularity_gating", "fusion_dimension",
    "gating_hidden_dimension", "component_count", "static_parameter_count",
    "dynamic_parameter_count", "gate_analysis_path",
    "gate_analysis_sha256", "best_epoch", "rank1", "map", "runtime",
    "seed", "commit",
)
RUN_FIELDS = (
    "run_id", "experiment_id", "experiment_family", "method", "dataset",
    "branch", "commit_id", "config_file", "seed", "lambda",
    "cross_camera_positive_lambda", "pcc_lambda", "pcc_enabled",
    "pcc_parts", "pcc_mode", "alignment_strategy", "baseline", "margin",
    "mode", "GPU", "start_time", "end_time", "runtime", "best_epoch",
    "selected_epoch", "Rank-1", "Rank-5", "Rank-10", "mAP", "checkpoint",
    "checkpoint_sha256", "log_path", "log_sha256", "output_dir", "status",
    "valid_pcc_pair_count", "mean_fixed_index_part_distance",
    "multi_granularity_fusion", "fusion_mode",
    "dynamic_granularity_gating", "fusion_dimension",
    "gating_hidden_dimension", "component_count", "static_parameter_count",
    "dynamic_parameter_count", "gate_analysis_path",
    "gate_analysis_sha256", "notes",
)
EVIDENCE_FIELDS = (
    "run_id", "artifact_type", "path", "size_bytes", "sha256",
)

TABLE_SCHEMAS = {
    "main_results": MAIN_FIELDS,
    "lambda_sensitivity": LAMBDA_FIELDS,
    "same_camera_positive_ablation": SAME_CAMERA_FIELDS,
    "caat_ablation": CAAT_FIELDS,
    "distance_distribution": DISTANCE_FIELDS,
    "anchor_coverage": ANCHOR_FIELDS,
    "pcc_ablation": PCC_FIELDS,
    "granularity_fusion_ablation": FUSION_FIELDS,
}


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
        return bool(value) if value != NOT_RECORDED else False

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
            "MODEL.MULTI_GRANULARITY_LOCAL",
            "MODEL.MULTI_GRANULARITY_PART",
            "MODEL.MULTI_GRANULARITY",
        )),
        "part_correspondence_consistency": enabled((
            "MODEL.PART_CORRESPONDENCE_CONSISTENCY",
        )),
        "multi_granularity_fusion": enabled((
            "MODEL.MULTI_GRANULARITY_FUSION",
        )),
        "dynamic_granularity_gating": (
            enabled(("MODEL.MULTI_GRANULARITY_FUSION",))
            and str(nested_value(
                configuration,
                "MODEL.MULTI_GRANULARITY_FUSION_MODE",
                "",
            )).lower() == "dynamic"
        ),
        "camera_conditional_part_attention": enabled((
            "MODEL.CAMERA_CONDITIONAL_PART_ATTENTION",
        )),
    }


def granularity_fusion_metadata(configuration):
    modules = config_modules(configuration)
    enabled = modules["multi_granularity_fusion"]
    if not enabled:
        return {
            "multi_granularity_fusion": False,
            "fusion_mode": NOT_RECORDED,
            "dynamic_granularity_gating": False,
            "fusion_dimension": NOT_RECORDED,
            "gating_hidden_dimension": NOT_RECORDED,
            "component_count": NOT_RECORDED,
            "static_parameter_count": NOT_RECORDED,
            "dynamic_parameter_count": NOT_RECORDED,
        }
    fusion_dim = int(nested_value(
        configuration, "MODEL.MULTI_GRANULARITY_FUSION_DIM"
    ))
    hidden_dim = int(nested_value(
        configuration, "MODEL.DYNAMIC_GATING_HIDDEN_DIM"
    ))
    model_name = str(nested_value(configuration, "MODEL.NAME", "resnet50"))
    global_dim = 512 if model_name in ("resnet18", "resnet34") else 2048
    component_count = 4
    global_projection = global_dim * fusion_dim + fusion_dim
    static_count = global_projection + component_count
    dynamic_count = global_projection + (
        component_count * fusion_dim * hidden_dim + hidden_dim
        + hidden_dim * component_count + component_count
    )
    mode = str(nested_value(
        configuration, "MODEL.MULTI_GRANULARITY_FUSION_MODE"
    )).lower()
    return {
        "multi_granularity_fusion": True,
        "fusion_mode": mode,
        "dynamic_granularity_gating": mode == "dynamic",
        "fusion_dimension": fusion_dim,
        "gating_hidden_dimension": hidden_dim,
        "component_count": component_count,
        "static_parameter_count": static_count,
        "dynamic_parameter_count": dynamic_count,
    }


def experiment_identity(configuration):
    modules = config_modules(configuration)
    fusion = granularity_fusion_metadata(configuration)
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
        method = "C2-L03 + Fixed-Index Part Correspondence Consistency"
        variant = "fixed_index_pcc"
        relation = "same_pid_different_camera_same_index"
    elif fusion["multi_granularity_fusion"]:
        if fusion["fusion_mode"] == "static":
            method = "C2-L03 + Multi-Granularity Static Fusion"
            variant = "multi_granularity_static_fusion"
        elif fusion["fusion_mode"] == "dynamic":
            method = "C2-L03 + Dynamic Granularity Gating"
            variant = "dynamic_granularity_gating"
        else:
            raise EvidenceError(
                "Unsupported granularity fusion mode: {}".format(
                    fusion["fusion_mode"]
                )
            )
    elif modules["multi_granularity"] and modules["cross_camera_positive"]:
        method = (
            "C2-L03 + Multi-Granularity Local Feature "
            "(Global + K2 + K4 + K6, mean aggregation)"
        )
        variant = "multi_granularity_local"
    dataset = nested_value(configuration, "DATASETS.NAMES")
    if isinstance(dataset, (list, tuple)):
        dataset = dataset[0] if dataset else NOT_RECORDED
    return {
        "method": method,
        "variant": variant,
        "positive_relation": relation,
        "dataset": str(dataset),
        "lambda": loss_lambda,
        "margin": nested_value(configuration, "SOLVER.MARGIN"),
        "mode": mode,
        "modules": modules,
        "baseline": "C2-L03" if (
            pcc_enabled or modules["multi_granularity"]
        ) else NOT_RECORDED,
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
        "multi_granularity_fusion": fusion["multi_granularity_fusion"],
        "fusion_mode": fusion["fusion_mode"],
        "dynamic_granularity_gating": fusion[
            "dynamic_granularity_gating"
        ],
        "fusion_dimension": fusion["fusion_dimension"],
        "gating_hidden_dimension": fusion["gating_hidden_dimension"],
        "component_count": fusion["component_count"],
        "static_parameter_count": fusion["static_parameter_count"],
        "dynamic_parameter_count": fusion["dynamic_parameter_count"],
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


def git_metadata(repo_root):
    commit = _git_output(repo_root, ["rev-parse", "HEAD"])
    branch = _git_output(repo_root, ["branch", "--show-current"])
    dirty = bool(_git_status_entries(repo_root))
    if not re.match(r"^[0-9a-fA-F]{40}$", commit):
        raise EvidenceError("Git commit is not a full SHA: {}".format(commit))
    if not branch:
        raise EvidenceError("Detached HEAD is not allowed for formal training")
    return {
        "commit": commit.lower(),
        "branch": branch,
        "dirty": dirty,
    }


def validate_git_preflight(repo_root, expected_branch, expected_commit=None):
    metadata = git_metadata(repo_root)
    if metadata["dirty"]:
        raise EvidenceError("Formal training requires a clean Git worktree")
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


def validate_git_runtime_state(repo_root, run_dir, expected_branch=None,
                               expected_commit=None):
    """Allow only new evidence files under this runner's exact run directory."""
    repo_root = Path(repo_root).resolve()
    run_dir = Path(run_dir).resolve()
    try:
        allowed_relative = normalized_path(run_dir.relative_to(repo_root))
    except ValueError:
        raise EvidenceError("Controlled run_dir must be inside the Git worktree")
    unexpected = []
    for status_code, path in _git_status_entries(repo_root):
        under_current_run = (
            path == allowed_relative or path.startswith(allowed_relative + "/")
        )
        if status_code != "??" or not under_current_run:
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
        when.strftime("%Y%m%dT%H%M%SZ"),
        str(commit)[:12],
        seed if seed != NOT_RECORDED else NOT_RECORDED,
    )


def initialize_run(records_root, experiment_id, experiment_family, run_id,
                   config_file, resolved_cfg, output_dir, git_info, notes,
                   command, expected_branch, method=None,
                   baseline_method=NOT_RECORDED,
                   baseline_commit=NOT_RECORDED):
    records_root = Path(records_root)
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
    seed_value = nested_value(configuration, "SEED")
    seed = validate_seed(seed_value) if seed_value != NOT_RECORDED else NOT_RECORDED
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "experiment_family": experiment_family,
        "method": method or identity["method"],
        "baseline_method": baseline_method,
        "baseline_commit": baseline_commit,
        "dataset": identity["dataset"],
        "branch": git_info["branch"],
        "commit_id": git_info["commit"],
        "expected_branch": expected_branch,
        "config_file": normalized_path(Path(config_file).resolve()),
        "config_source": "config_source.yml",
        "config_source_sha256": sha256_file(source_copy),
        "config_resolved": "config_resolved.yml",
        "config_resolved_sha256": sha256_file(resolved_copy),
        "seed": seed,
        "lambda": identity["lambda"],
        "cross_camera_positive_lambda": identity["cross_camera_positive_lambda"],
        "pcc_enabled": identity["pcc_enabled"],
        "pcc_parts": identity["pcc_parts"],
        "pcc_lambda": identity["pcc_lambda"],
        "pcc_mode": identity["pcc_mode"],
        "alignment_strategy": identity["alignment_strategy"],
        "baseline": identity["baseline"],
        "multi_granularity_fusion": identity[
            "multi_granularity_fusion"
        ],
        "fusion_mode": identity["fusion_mode"],
        "dynamic_granularity_gating": identity[
            "dynamic_granularity_gating"
        ],
        "fusion_dimension": identity["fusion_dimension"],
        "gating_hidden_dimension": identity["gating_hidden_dimension"],
        "component_count": identity["component_count"],
        "static_parameter_count": identity["static_parameter_count"],
        "dynamic_parameter_count": identity["dynamic_parameter_count"],
        "gate_analysis_path": NOT_RECORDED,
        "gate_analysis_sha256": NOT_RECORDED,
        "margin": identity["margin"],
        "mode": identity["mode"],
        "modules": identity["modules"],
        "required_global_iteration_source": (
            "ignite_engine_epoch_evidence"
        ),
        "output_dir": normalized_path(Path(output_dir).resolve()),
        "start_time": utc_now(),
        "command": list(command),
        "notes": notes,
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "run_status.json", {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "phase": "initialized",
        "training_exit_code": NOT_RECORDED,
        "updated_at_utc": utc_now(),
    })
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
    return status


def record_run_failure(run_dir, error, status="incomplete"):
    run_dir = Path(run_dir)
    path = run_dir / "run_status.json"
    payload = read_json(path) if path.is_file() else {"schema_version": SCHEMA_VERSION}
    payload.update({
        "status": status,
        "phase": "failed" if status == "failed" else "finalization",
        "error_type": type(error).__name__,
        "error": str(error),
        "updated_at_utc": utc_now(),
    })
    atomic_write_json(path, payload)
    return payload


def parse_training_log(log_path):
    path = Path(log_path)
    if not path.is_file():
        raise EvidenceError("Training log is missing: {}".format(path))
    timestamps = []
    validations = []
    current = None
    iterations_per_epoch_values = set()
    pcc_epoch_summaries = []
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
        "pcc_epoch_summaries": pcc_epoch_summaries,
        "valid_pcc_pair_count": total_pcc_pairs
        if pcc_epoch_summaries else NOT_RECORDED,
        "mean_fixed_index_part_distance": mean_pcc_distance,
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
            "epoch", "global_iteration", "global_iteration_source",
            "filename", "size_bytes", "sha256", "selected",
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


def upsert_csv(path, fields, row, key_fields=("run_id",)):
    rows = _read_csv(path)
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
    rows = []
    if target.is_file() and target.stat().st_size:
        with target.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
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
        if not csv_path.is_file():
            upsert_csv(csv_path, fields, {field: "" for field in fields}, key_fields=("run_id",))
            rows = _read_csv(csv_path)
            rows = [row for row in rows if any(row.values())]
            temporary = csv_path.with_name("{}.tmp.{}".format(csv_path.name, os.getpid()))
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            os.replace(str(temporary), str(csv_path))
        csv_to_markdown(csv_path, tables / "{}.md".format(name))
    runs_path = root / "runs.csv"
    if not runs_path.is_file():
        with runs_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=RUN_FIELDS, lineterminator="\n").writeheader()
    evidence_path = root / "evidence_manifest.tsv"
    if not evidence_path.is_file():
        write_tsv(evidence_path, EVIDENCE_FIELDS, [])
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
    if manifest.get("dynamic_granularity_gating", False):
        gate_summary_path = run_dir / "granularity_gating_summary.json"
        gate_csv_path = run_dir / "granularity_gating_per_sample.csv"
        gate_command = [
            sys.executable,
            str(Path(repo_root) / "tools" / "analyze_granularity_gating.py"),
            "--config", config_path,
            "--checkpoint", selected_checkpoint["path"],
            "--output-dir", str(run_dir),
            "--split", "all",
        ]
        gate = subprocess.run(gate_command, cwd=str(repo_root), check=False)
        if (gate.returncode != 0 or not gate_summary_path.is_file()
                or not gate_csv_path.is_file()):
            raise EvidenceError("Dynamic granularity gate analysis failed")


def _artifact_rows(run_dir, log_info, selected_checkpoint, final_status):
    run_dir = Path(run_dir)
    rows = []
    final_status_bytes = json_text(final_status).encode("utf-8")
    for path in sorted(run_dir.iterdir()):
        if not path.is_file() or path.name in ("artifact_hashes.tsv", "run_status.json"):
            continue
        rows.append({
            "artifact_type": path.stem,
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
        {
            "artifact_type": "selected_checkpoint",
            "path": selected_checkpoint["path"],
            "size_bytes": selected_checkpoint["size_bytes"],
            "sha256": selected_checkpoint["sha256"],
        },
    ])
    return rows


def _lambda_table_eligible(manifest):
    family = str(manifest["experiment_family"]).lower()
    identifier = str(manifest["experiment_id"]).lower()
    return (
        manifest["lambda"] != NOT_RECORDED
        and ("lambda" in family or re.search(r"(?:^|-)l\d+", identifier))
    )


def update_experiments_markdown(experiments_path, main_rows):
    target = Path(experiments_path)
    historical = target.read_text(encoding="utf-8") if target.is_file() else "# Experiments\n"
    fields = (
        "experiment_id", "run_id", "date", "commit", "branch", "method",
        "dataset", "config", "output_dir", "log_path", "log_sha256", "GPU",
        "seed", "lambda", "cross_camera_positive_lambda", "pcc_lambda",
        "pcc_enabled", "pcc_parts", "pcc_mode", "alignment_strategy",
        "baseline", "multi_granularity_fusion", "fusion_mode",
        "dynamic_granularity_gating", "fusion_dimension",
        "gating_hidden_dimension", "component_count",
        "static_parameter_count", "dynamic_parameter_count",
        "gate_analysis_path", "gate_analysis_sha256",
        "valid_pcc_pair_count",
        "mean_fixed_index_part_distance", "runtime_seconds", "best_epoch", "Rank-1",
        "Rank-5", "Rank-10", "mAP", "checkpoint", "checkpoint_sha256",
        "status", "notes",
    )
    lines = [
        AUTO_RESULTS_START,
        "## Automated Formal Experiment Runs",
        "",
        "This section is generated from `experiment_records/tables/main_results.csv`.",
        "Historical experiment rows outside this section are never rewritten.",
        "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in main_rows:
        date_match = re.search(r"(\d{8})T", row["run_id"])
        run_date = NOT_RECORDED
        if date_match:
            compact = date_match.group(1)
            run_date = "{}-{}-{}".format(compact[:4], compact[4:6], compact[6:8])
        values = {
            "experiment_id": row["experiment_id"],
            "run_id": row["run_id"],
            "date": run_date,
            "commit": row["commit"],
            "branch": row["branch"],
            "method": row["method"],
            "dataset": row["dataset"],
            "config": row["config"],
            "output_dir": row["output_dir"],
            "log_path": row["log_path"],
            "log_sha256": row["log_sha256"],
            "GPU": row["gpu"],
            "seed": row["seed"],
            "lambda": row["lambda"],
            "cross_camera_positive_lambda": row["cross_camera_positive_lambda"],
            "pcc_lambda": row["pcc_lambda"],
            "pcc_enabled": row["pcc_enabled"],
            "pcc_parts": row["pcc_parts"],
            "pcc_mode": row["pcc_mode"],
            "alignment_strategy": row["alignment_strategy"],
            "baseline": row["baseline"],
            "multi_granularity_fusion": row.get(
                "multi_granularity_fusion", NOT_RECORDED
            ),
            "fusion_mode": row.get("fusion_mode", NOT_RECORDED),
            "dynamic_granularity_gating": row.get(
                "dynamic_granularity_gating", NOT_RECORDED
            ),
            "fusion_dimension": row.get(
                "fusion_dimension", NOT_RECORDED
            ),
            "gating_hidden_dimension": row.get(
                "gating_hidden_dimension", NOT_RECORDED
            ),
            "component_count": row.get("component_count", NOT_RECORDED),
            "static_parameter_count": row.get(
                "static_parameter_count", NOT_RECORDED
            ),
            "dynamic_parameter_count": row.get(
                "dynamic_parameter_count", NOT_RECORDED
            ),
            "gate_analysis_path": row.get(
                "gate_analysis_path", NOT_RECORDED
            ),
            "gate_analysis_sha256": row.get(
                "gate_analysis_sha256", NOT_RECORDED
            ),
            "valid_pcc_pair_count": row["valid_pcc_pair_count"],
            "mean_fixed_index_part_distance": row[
                "mean_fixed_index_part_distance"
            ],
            "runtime_seconds": row["runtime_seconds"],
            "best_epoch": row["best_epoch"],
            "Rank-1": row["rank1"],
            "Rank-5": row["rank5"],
            "Rank-10": row["rank10"],
            "mAP": row["map"],
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "status": row["status"],
            "notes": row["notes"],
        }
        lines.append("| " + " | ".join(
            str(values[field]).replace("|", "\\|").replace("\n", " ")
            for field in fields
        ) + " |")
    lines.extend([AUTO_RESULTS_END, ""])
    generated = "\n".join(lines)
    if AUTO_RESULTS_START in historical and AUTO_RESULTS_END in historical:
        prefix = historical.split(AUTO_RESULTS_START, 1)[0].rstrip()
        suffix = historical.split(AUTO_RESULTS_END, 1)[1].lstrip()
        content = prefix + "\n\n" + generated
        if suffix:
            content += "\n" + suffix
    else:
        content = historical.rstrip() + "\n\n" + generated
    atomic_write_text(target, content)


def _prepare_table_rows(manifest, metrics, environment, efficiency,
                        distance, anchor):
    common = {
        "run_id": manifest["run_id"],
        "experiment_id": manifest["experiment_id"],
        "experiment_family": manifest["experiment_family"],
        "method": manifest["method"],
        "dataset": manifest["dataset"],
        "branch": manifest["branch"],
        "commit": manifest["commit_id"],
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
        "log_path": metrics["log_path"],
        "log_sha256": metrics["log_sha256"],
        "output_dir": manifest["output_dir"],
        "valid_pcc_pair_count": metrics["valid_pcc_pair_count"],
        "mean_fixed_index_part_distance": metrics["mean_fixed_index_part_distance"],
        "multi_granularity_fusion": manifest.get(
            "multi_granularity_fusion", False
        ),
        "fusion_mode": manifest.get("fusion_mode", NOT_RECORDED),
        "dynamic_granularity_gating": manifest.get(
            "dynamic_granularity_gating", False
        ),
        "fusion_dimension": manifest.get(
            "fusion_dimension", NOT_RECORDED
        ),
        "gating_hidden_dimension": manifest.get(
            "gating_hidden_dimension", NOT_RECORDED
        ),
        "component_count": manifest.get("component_count", NOT_RECORDED),
        "static_parameter_count": manifest.get(
            "static_parameter_count", NOT_RECORDED
        ),
        "dynamic_parameter_count": manifest.get(
            "dynamic_parameter_count", NOT_RECORDED
        ),
        "gate_analysis_path": manifest.get(
            "gate_analysis_path", NOT_RECORDED
        ),
        "gate_analysis_sha256": manifest.get(
            "gate_analysis_sha256", NOT_RECORDED
        ),
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
        "run_id": common["run_id"],
        "experiment_id": common["experiment_id"],
        "baseline": common["baseline"],
        "pcc_enabled": common["pcc_enabled"],
        "alignment_strategy": common["alignment_strategy"],
        "pcc_parts": common["pcc_parts"],
        "cross_camera_positive_lambda": common["cross_camera_positive_lambda"],
        "pcc_lambda": common["pcc_lambda"],
        "valid_pcc_pair_count": common["valid_pcc_pair_count"],
        "mean_fixed_index_part_distance": common["mean_fixed_index_part_distance"],
        "best_epoch": common["best_epoch"],
        "rank1": common["rank1"],
        "map": common["map"],
        "runtime": common["runtime_seconds"],
        "seed": common["seed"],
        "commit": common["commit"],
    }
    fusion_row = {
        "run_id": common["run_id"],
        "experiment_id": common["experiment_id"],
        "method": common["method"],
        "baseline": common["baseline"],
        "multi_granularity_fusion": common["multi_granularity_fusion"],
        "fusion_mode": common["fusion_mode"],
        "dynamic_granularity_gating": common[
            "dynamic_granularity_gating"
        ],
        "fusion_dimension": common["fusion_dimension"],
        "gating_hidden_dimension": common["gating_hidden_dimension"],
        "component_count": common["component_count"],
        "static_parameter_count": common["static_parameter_count"],
        "dynamic_parameter_count": common["dynamic_parameter_count"],
        "gate_analysis_path": common["gate_analysis_path"],
        "gate_analysis_sha256": common["gate_analysis_sha256"],
        "best_epoch": common["best_epoch"],
        "rank1": common["rank1"],
        "map": common["map"],
        "runtime": common["runtime_seconds"],
        "seed": common["seed"],
        "commit": common["commit"],
    }
    return (
        common, lambda_row, same_row, caat_row, distance_row, anchor_row,
        pcc_row, fusion_row,
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
        output_dir = Path(manifest["output_dir"])
        reproducibility_source = output_dir / "reproducibility.json"
        if reproducibility_source.is_file():
            copy_file_atomic(reproducibility_source, run_dir / "reproducibility.json")
        reproducibility = read_json(run_dir / "reproducibility.json")
        required_reproducibility = (
            "source_seed", "resolved_seed", "applied_seed", "seed",
            "PYTHONHASHSEED", "python_random_seed", "numpy_seed",
            "torch_cpu_seed", "torch_cuda_seed",
            "dataloader_worker_seed_base",
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
            "python_random_seed", "numpy_seed", "torch_cpu_seed",
            "torch_cuda_seed", "dataloader_worker_seed_base",
            "sampler_seed",
        )
        for key in applied_seed_fields:
            if validate_seed(reproducibility[key]) != resolved_seed:
                raise EvidenceError(
                    "Seed evidence conflict: {} differs from resolved seed"
                    .format(key)
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
            if not log_info["pcc_epoch_summaries"]:
                raise EvidenceError("Enabled PCC has no epoch pair statistics")
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
        if manifest.get("dynamic_granularity_gating", False):
            gate_summary_path = run_dir / "granularity_gating_summary.json"
            gate_csv_path = run_dir / "granularity_gating_per_sample.csv"
            if not gate_summary_path.is_file() or not gate_csv_path.is_file():
                raise EvidenceError(
                    "Dynamic gating requires per-sample CSV and summary JSON"
                )
            gate_summary = read_json(gate_summary_path)
            if gate_summary.get("checkpoint_sha256") != selected["sha256"]:
                raise EvidenceError(
                    "Gate analysis checkpoint hash does not match selection"
                )
            if gate_summary.get("per_sample_csv_sha256") != sha256_file(
                    gate_csv_path):
                raise EvidenceError("Gate analysis CSV hash is inconsistent")
            manifest["gate_analysis_path"] = normalized_path(
                gate_summary_path.resolve()
            )
            manifest["gate_analysis_sha256"] = sha256_file(
                gate_summary_path
            )
            atomic_write_json(run_dir / "run_manifest.json", manifest)
        anchor = validate_anchor_coverage(read_json(run_dir / "anchor_coverage.json"))
        distance = validate_distance_distribution(
            read_json(run_dir / "distance_distribution.json")
        )
        efficiency_path = run_dir / "efficiency_profile.json"
        efficiency = read_json(efficiency_path) if efficiency_path.is_file() else efficiency_placeholder()
        if not efficiency_path.is_file():
            atomic_write_json(efficiency_path, efficiency)
        model_manifest = read_json(run_dir / "model_manifest.json")
        model_manifest.update({
            "total_params": efficiency.get("total_params", NOT_RECORDED),
            "trainable_params": efficiency.get("trainable_params", NOT_RECORDED),
            "FLOPs": efficiency.get("FLOPs", NOT_RECORDED),
            "MACs": efficiency.get("MACs", NOT_RECORDED),
            "selected_checkpoint_sha256": selected["sha256"],
            "efficiency_profile": "efficiency_profile.json",
            "gate_analysis_path": manifest.get(
                "gate_analysis_path", NOT_RECORDED
            ),
            "gate_analysis_sha256": manifest.get(
                "gate_analysis_sha256", NOT_RECORDED
            ),
        })
        atomic_write_json(run_dir / "model_manifest.json", model_manifest)
        runtime_seconds = float(status["training_runtime_seconds"])
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
            "mean_fixed_index_part_distance": log_info[
                "mean_fixed_index_part_distance"
            ],
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
         pcc_row, fusion_row) = rows
        final_status = dict(status)
        final_status.update({
            "status": "success",
            "phase": "complete",
            "end_time": status.get("training_end_time", utc_now()),
            "best_epoch": metrics["best_epoch"],
            "selected_epoch": metrics["selected_epoch"],
            "updated_at_utc": utc_now(),
        })
        artifact_rows = _artifact_rows(run_dir, log_info, selected, final_status)
        write_tsv(
            run_dir / "artifact_hashes.tsv",
            ("artifact_type", "path", "size_bytes", "sha256"),
            artifact_rows,
        )
        tables_dir = records_root / "tables"
        main_rows = upsert_csv(tables_dir / "main_results.csv", MAIN_FIELDS, common)
        if _lambda_table_eligible(manifest):
            upsert_csv(tables_dir / "lambda_sensitivity.csv", LAMBDA_FIELDS, lambda_row)
        if experiment_identity(resolved_cfg)["variant"] in (
                "baseline", "cross_camera_positive", "same_camera_positive"):
            upsert_csv(
                tables_dir / "same_camera_positive_ablation.csv",
                SAME_CAMERA_FIELDS,
                same_row,
            )
        upsert_csv(tables_dir / "caat_ablation.csv", CAAT_FIELDS, caat_row)
        upsert_csv(
            tables_dir / "distance_distribution.csv", DISTANCE_FIELDS, distance_row
        )
        upsert_csv(tables_dir / "anchor_coverage.csv", ANCHOR_FIELDS, anchor_row)
        if manifest.get("pcc_enabled", False):
            upsert_csv(tables_dir / "pcc_ablation.csv", PCC_FIELDS, pcc_row)
        if manifest.get("multi_granularity_fusion", False):
            upsert_csv(
                tables_dir / "granularity_fusion_ablation.csv",
                FUSION_FIELDS,
                fusion_row,
            )
        for table_name in TABLE_SCHEMAS:
            csv_to_markdown(
                tables_dir / "{}.csv".format(table_name),
                tables_dir / "{}.md".format(table_name),
            )
        run_row = {
            "run_id": manifest["run_id"], "experiment_id": manifest["experiment_id"],
            "experiment_family": manifest["experiment_family"],
            "method": manifest["method"], "dataset": manifest["dataset"],
            "branch": manifest["branch"], "commit_id": manifest["commit_id"],
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
            "multi_granularity_fusion": manifest.get(
                "multi_granularity_fusion", False
            ),
            "fusion_mode": manifest.get("fusion_mode", NOT_RECORDED),
            "dynamic_granularity_gating": manifest.get(
                "dynamic_granularity_gating", False
            ),
            "fusion_dimension": manifest.get(
                "fusion_dimension", NOT_RECORDED
            ),
            "gating_hidden_dimension": manifest.get(
                "gating_hidden_dimension", NOT_RECORDED
            ),
            "component_count": manifest.get(
                "component_count", NOT_RECORDED
            ),
            "static_parameter_count": manifest.get(
                "static_parameter_count", NOT_RECORDED
            ),
            "dynamic_parameter_count": manifest.get(
                "dynamic_parameter_count", NOT_RECORDED
            ),
            "gate_analysis_path": manifest.get(
                "gate_analysis_path", NOT_RECORDED
            ),
            "gate_analysis_sha256": manifest.get(
                "gate_analysis_sha256", NOT_RECORDED
            ),
            "notes": manifest["notes"],
        }
        upsert_csv(records_root / "runs.csv", RUN_FIELDS, run_row)
        evidence_rows = []
        for row in artifact_rows:
            evidence_rows.append({
                "run_id": manifest["run_id"],
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
        update_experiments_markdown(experiments_path, main_rows)
        atomic_write_json(run_dir / "run_status.json", final_status)
        return {"manifest": manifest, "metrics": metrics, "status": final_status}
    except BaseException as error:
        failure_status = "incomplete"
        try:
            current_status = read_json(run_dir / "run_status.json")
            exit_code = current_status.get("training_exit_code")
            if exit_code not in (0, "0", NOT_RECORDED):
                failure_status = "failed"
        except Exception:
            pass
        record_run_failure(run_dir, error, status=failure_status)
        raise
