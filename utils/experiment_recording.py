# encoding: utf-8
"""Strict evidence recording for the C2-MGP-K246 experiment only."""

from __future__ import absolute_import

import csv
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.backends import cudnn

from utils.config_serialization import (
    cfg_node_to_plain_mapping,
    deserialize_cfg_node_yaml,
    serialize_cfg_node_yaml,
)
from utils.reproducibility import (
    data_loader_generator_metadata,
    read_explicit_config_seed,
    validate_seed_evidence_chain,
)


EXPECTED_BRANCH = "exp/c2-l03-multi-granularity-local-feature"
EXPECTED_EXPERIMENT_FAMILY = "C2-MGP-K246"
EXPECTED_RUN_ID = "C2-MGP-K246-S42"
EXPECTED_EVIDENCE_ID = "EV-TRAIN-MKT-C2-MGP-K246-S42"
EXPECTED_TRAINING_SEED = 42
FORMAL_CONFIG_RELATIVE_PATH = (
    "configs/softmax_triplet_c2_l03_multi_granularity_part_autodl.yml"
)
INDEPENDENT_RUNS = 1
ANALYSIS_SEED = "not_applicable"
SCHEMA_VERSION = 1
EFFICIENCY_SCHEMA_VERSION = 2
CHECKPOINT_SELECTION_RULE = (
    "highest Rank-1; if tied, highest mAP; every metric comes from the same "
    "validation record"
)
SAMPLER_EPOCH_SEED_RULE = (
    "(base_seed + zero_based_epoch_index) modulo 2**32"
)
NOT_RECORDED = "not_recorded"
NOT_ARCHIVED = "not_archived"
NOT_APPLICABLE = "not_applicable"
TRAINING_COMPLETE = "training_complete"
LOCAL_EVIDENCE_PENDING = "local_complete_pending_commit_and_archive"
DATA_LOADER_WORKER_SEED_SCHEME = (
    "torch.initial_seed() modulo 2**32 -> Python random and NumPy"
)
PROFILER_WORKER_ISOLATION = "each variant measured in an independent subprocess"
OPERATION_COUNT_CONVENTION = "Conv2d/Linear MACs; FLOPs=2×MACs"
FINALIZATION_TIMING_BOUNDARY = (
    "after evidence validation, metrics/checkpoint generation, draft artifact "
    "hashing, and registry candidate staging; before final atomic evidence sealing"
)
CONFIG_TYPE_REPAIR_REASON = (
    "replace non-type-safe resolved YAML with strict tagged CfgNode evidence"
)

FORMAL_PROTOCOL = {
    "SEED": 42,
    "MODEL.DEVICE": "cuda",
    "MODEL.NAME": "resnet50",
    "MODEL.LAST_STRIDE": 1,
    "MODEL.NECK": "bnneck",
    "MODEL.METRIC_LOSS_TYPE": "triplet",
    "MODEL.IF_LABELSMOOTH": "on",
    "DATASETS.NAMES": "market1501",
    "DATALOADER.SAMPLER": "softmax_triplet",
    "DATALOADER.NUM_INSTANCE": 4,
    "DATALOADER.NUM_WORKERS": 8,
    "SOLVER.IMS_PER_BATCH": 64,
    "SOLVER.MAX_EPOCHS": 120,
    "SOLVER.OPTIMIZER_NAME": "Adam",
    "SOLVER.BASE_LR": 0.00035,
    "SOLVER.STEPS": [40, 70],
    "SOLVER.CHECKPOINT_PERIOD": 40,
    "SOLVER.EVAL_PERIOD": 40,
    "INPUT.SIZE_TRAIN": [256, 128],
    "INPUT.SIZE_TEST": [256, 128],
    "TEST.IMS_PER_BATCH": 128,
    "TEST.RE_RANKING": "no",
    "TEST.NECK_FEAT": "after",
    "TEST.FEAT_NORM": "yes",
    "MODEL.PRETRAIN_CHOICE": "imagenet",
    "MODEL.IF_WITH_CENTER": "no",
    "MODEL.PART_ATTENTION": False,
    "MODEL.PART_ATTENTION_PARTS": 6,
    "MODEL.MULTI_GRANULARITY_PART": True,
    "MODEL.MULTI_GRANULARITY_PART_SCALES": [2, 4, 6],
    "MODEL.MULTI_GRANULARITY_PART_DIM": 256,
    "MODEL.MULTI_GRANULARITY_PART_AGGREGATION": "mean",
    "MODEL.MULTI_GRANULARITY_PART_FUSION": "concat",
    "MODEL.CROSS_CAMERA_POSITIVE_ONLY": True,
    "MODEL.CROSS_CAMERA_POSITIVE_LAMBDA": 0.3,
    "MODEL.CROSS_CAMERA_POSITIVE_MODE": "mean",
    "MODEL.CAMERA_AWARE_TRIPLET": False,
    "MODEL.CAMERA_AWARE_TRIPLET_LAMBDA": 0.5,
    "MODEL.CAMERA_AWARE_TRIPLET_MARGIN": 0.3,
    "MODEL.CAMERA_AWARE_TRIPLET_MODE": "hard",
    "INPUT.PROB": 0.5,
    "INPUT.RE_PROB": 0.5,
    "INPUT.PIXEL_MEAN": [0.485, 0.456, 0.406],
    "INPUT.PIXEL_STD": [0.229, 0.224, 0.225],
    "INPUT.PADDING": 10,
    "SOLVER.BIAS_LR_FACTOR": 1,
    "SOLVER.MOMENTUM": 0.9,
    "SOLVER.MARGIN": 0.3,
    "SOLVER.CLUSTER_MARGIN": 0.3,
    "SOLVER.CENTER_LR": 0.5,
    "SOLVER.CENTER_LOSS_WEIGHT": 0.0005,
    "SOLVER.RANGE_K": 2,
    "SOLVER.RANGE_MARGIN": 0.3,
    "SOLVER.RANGE_ALPHA": 0,
    "SOLVER.RANGE_BETA": 1,
    "SOLVER.RANGE_LOSS_WEIGHT": 1,
    "SOLVER.WEIGHT_DECAY": 0.0005,
    "SOLVER.WEIGHT_DECAY_BIAS": 0.0005,
    "SOLVER.GAMMA": 0.1,
    "SOLVER.WARMUP_FACTOR": 0.01,
    "SOLVER.WARMUP_ITERS": 10,
    "SOLVER.WARMUP_METHOD": "linear",
    "SOLVER.LOG_PERIOD": 20,
    "TEST.WEIGHT": "path",
}

# These leaves are validated through concrete path identity, existence,
# emptiness and/or SHA256 checks in formal_preflight/finalize_run. They are
# deliberately excluded from value equality because fixture paths differ.
FORMAL_INDEPENDENT_FIELDS = {
    "MODEL.PRETRAIN_PATH",
    "DATASETS.ROOT_DIR",
    "OUTPUT_DIR",
}
FORMAL_RESOLVED_DEFAULTS = {
    # YACS literal-decodes numeric-looking strings, so DEVICE_ID remains in the
    # defaults and is locked when present in the resolved configuration.
    "MODEL.DEVICE_ID": "0",
}

RUN_FIELDS = (
    "experiment_family", "run_id", "evidence_id", "dataset", "method_variant",
    "aux_lambda", "mode", "selected_epoch", "rank1_percent", "rank5_percent",
    "rank10_percent", "map_percent", "re_ranking", "independent_runs",
    "training_seed", "training_commit", "finalization_commit", "branch",
    "training_runtime_seconds",
    "total_run_runtime_seconds", "gpu",
    "descriptor_dim", "total_parameters", "trainable_parameters", "status",
    "evidence_status", "result_commit", "archive_status", "notes",
)

VALIDATION_FIELDS = (
    "epoch", "global_iteration", "timestamp_utc", "rank1_percent",
    "rank5_percent", "rank10_percent", "map_percent", "re_ranking",
    "neck_feat", "feat_norm",
)

REQUIRED_ENVIRONMENT_FIELDS = (
    "hostname", "os", "kernel", "machine_architecture", "python_version",
    "python_executable", "pytorch_version", "torchvision_version",
    "ignite_version", "yacs_version", "numpy_version", "pillow_version",
    "cuda_runtime", "cudnn_version", "nvidia_driver", "gpu_count", "gpus",
    "cuda_visible_devices", "pythonhashseed", "cublas_workspace_config",
    "timezone", "pip_freeze_path", "pip_freeze_sha256",
)

REQUIRED_DATASET_FIELDS = (
    "dataset_name", "data_root", "train_image_count", "query_image_count",
    "gallery_image_count", "train_pid_count", "query_pid_count",
    "gallery_pid_count", "train_camera_count", "query_camera_count",
    "gallery_camera_count", "sampler", "batch_size", "num_instance",
    "num_workers", "train_loader_batches", "sampler_base_seed",
    "sampler_epoch_seed_rule", "split_manifest_sha256", "hash_algorithm",
    "sorting_rule", "data_loader_generators",
)

REQUIRED_MODEL_FIELDS = (
    "backbone", "feature_map_shape", "branches", "scales", "projection_dim",
    "aggregation", "fusion", "descriptor_dim", "num_classes",
    "total_parameters", "trainable_parameters", "pretrained_weight_path",
    "pretrained_weight_sha256", "state_dict_schema",
)

REQUIRED_EFFICIENCY_FIELDS = (
    "schema_version", "status", "mode", "measurement_timestamp_utc",
    "measurement_seed", "seed_source", "python_executable", "argv",
    "display_command", "profiler_script_sha256", "source_config_sha256",
    "resolved_config_sha256", "config_overrides", "measurement", "variants",
    "deltas",
)

REQUIRED_EFFICIENCY_MEASUREMENT_FIELDS = (
    "num_classes", "batch_size", "input_size", "dtype", "device",
    "gpu_name", "gpu_total_memory_mib", "nvidia_driver", "pytorch_version",
    "cuda_runtime", "cudnn_version", "warmup", "measurement_repeats",
    "worker_isolation", "operation_count_convention",
)

REQUIRED_EFFICIENCY_VARIANT_FIELDS = (
    "name", "variant", "measurement_seed", "feature_dim", "total_parameters", "trainable_parameters",
    "flops", "macs", "inference_latency_median_ms",
    "inference_latency_p95_ms", "throughput_images_per_second",
    "forward_peak_memory", "forward_backward_peak_memory",
)

REQUIRED_EFFICIENCY_DELTA_FIELDS = (
    "total_parameters", "trainable_parameters", "descriptor_dim", "flops",
    "macs", "latency_median_ms", "latency_p95_ms",
    "throughput_images_per_second",
    "memory_forward_peak_allocated_mib",
    "memory_forward_peak_reserved_mib",
    "memory_forward_backward_peak_allocated_mib",
    "memory_forward_backward_peak_reserved_mib",
)

FORBIDDEN_PATH_PARTS = {"downloads", "thesis_evidence", "paper_notes"}
FORBIDDEN_BASENAMES = {"experiments.md", "append_experiment_result.py"}


class PreflightError(RuntimeError):
    pass


class EvidenceIncompleteError(RuntimeError):
    pass


class FormalRunLockError(RuntimeError):
    pass


def _nested_value(configuration, dotted_path):
    current = configuration
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(dotted_path)
            current = current[part]
        else:
            if not hasattr(current, part):
                raise KeyError(dotted_path)
            current = getattr(current, part)
    return current


def _protocol_value(value):
    if isinstance(value, tuple):
        return [_protocol_value(item) for item in value]
    if isinstance(value, list):
        return [_protocol_value(item) for item in value]
    return value


def flatten_config_leaves(configuration, prefix=""):
    """Return every YAML/YACS leaf keyed by its dotted configuration path."""
    if isinstance(configuration, dict) or (
            not isinstance(configuration, (str, bytes))
            and hasattr(configuration, "items")):
        flattened = {}
        for key, value in configuration.items():
            dotted = "{}.{}".format(prefix, key) if prefix else str(key)
            flattened.update(flatten_config_leaves(value, dotted))
        return flattened
    return {prefix: _protocol_value(configuration)}


def validate_formal_protocol(configuration, label):
    """Require all training leaves to be locked or independently evidenced."""
    leaves = flatten_config_leaves(configuration)
    required = set(FORMAL_PROTOCOL).union(FORMAL_INDEPENDENT_FIELDS)
    covered = required.union(FORMAL_RESOLVED_DEFAULTS)
    missing = sorted(required.difference(leaves))
    uncovered = sorted(set(leaves).difference(covered))
    if missing:
        raise PreflightError(
            "{} config is missing locked/independent leaves: {}".format(
                label, missing
            )
        )
    if uncovered:
        raise PreflightError(
            "{} config contains training leaves without a protocol or independent "
            "path/hash check: {}".format(label, uncovered)
        )
    for dotted_path, expected in FORMAL_PROTOCOL.items():
        actual = leaves[dotted_path]
        if actual != expected:
            raise PreflightError(
                "{} config protocol mismatch: {} expected {!r}, got {!r}".format(
                    label, dotted_path, expected, actual
                )
            )
    for dotted_path, expected in FORMAL_RESOLVED_DEFAULTS.items():
        if dotted_path in leaves and leaves[dotted_path] != expected:
            raise PreflightError(
                "{} config resolved-default mismatch: {} expected {!r}, got {!r}"
                .format(label, dotted_path, expected, leaves[dotted_path])
            )
    return True


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def local_timezone():
    value = dt.datetime.now().astimezone().tzname()
    return value if value else NOT_RECORDED


def assert_path_allowed(path):
    candidate = Path(path).resolve()
    lower_parts = {part.lower() for part in candidate.parts}
    forbidden_parts = sorted(lower_parts.intersection(FORBIDDEN_PATH_PARTS))
    if forbidden_parts or candidate.name.lower() in FORBIDDEN_BASENAMES:
        raise ValueError("Forbidden experiment-recording path: {}".format(candidate))
    return candidate


def require_temporary_fixture(path):
    candidate = assert_path_allowed(path)
    temporary_root = Path(tempfile.gettempdir()).resolve()
    try:
        common = Path(os.path.commonpath([str(candidate), str(temporary_root)]))
    except ValueError:
        common = None
    if common != temporary_root:
        raise ValueError("Fixture directory must be inside the system temporary directory")
    return candidate


def require_contained_path(path, root, label):
    """Resolve ``path`` and require it to stay beneath the resolved root."""
    fixture_root = Path(root).resolve()
    candidate = Path(path).resolve()
    try:
        common = Path(os.path.commonpath([str(candidate), str(fixture_root)]))
    except ValueError:
        common = None
    if common != fixture_root:
        raise ValueError(
            "Fixture {} escapes fixture root: {}".format(label, candidate)
        )
    return candidate


def _read_contained_json(path, root, label):
    target = require_contained_path(path, root, label)
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_config_contained_paths(configuration, root, label):
    if not isinstance(configuration, dict):
        raise ValueError("Fixture {} config must be a mapping".format(label))
    for dotted_path in (
            "OUTPUT_DIR", "DATASETS.ROOT_DIR", "MODEL.PRETRAIN_PATH"):
        value = _nested_value(configuration, dotted_path)
        require_contained_path(
            value, root, "{} {}".format(label, dotted_path)
        )


def validate_fixture_path_containment(output_dir, record_dir):
    """Validate every dry-run evidence path before finalization reads or writes."""
    root = require_temporary_fixture(output_dir)
    if not root.is_dir():
        raise ValueError("Fixture directory does not exist: {}".format(root))
    require_contained_path(record_dir, root, "registry")

    fixed_paths = (
        "run_manifest.json",
        "run_status.json",
        "reproducibility.json",
        "environment.json",
        "environment_packages.txt",
        "dataset_manifest.json",
        "model_manifest.json",
        "efficiency_profile.json",
        "config_resolved.yml",
        "validation_history.jsonl",
        "log.txt",
    )
    for relative_path in fixed_paths:
        require_contained_path(root / relative_path, root, relative_path)
    for child in root.iterdir():
        require_contained_path(child, root, "output entry {}".format(child.name))

    manifest = _read_contained_json(
        root / "run_manifest.json", root, "run manifest"
    )
    dataset = _read_contained_json(
        root / "dataset_manifest.json", root, "dataset manifest"
    )
    model = _read_contained_json(
        root / "model_manifest.json", root, "model manifest"
    )
    environment = _read_contained_json(
        root / "environment.json", root, "environment manifest"
    )
    source_path = require_contained_path(
        manifest.get("source_config", {}).get("path", ""),
        root,
        "source config",
    )
    resolved_path = require_contained_path(
        root / "config_resolved.yml", root, "resolved config"
    )
    require_contained_path(
        manifest.get("launch_script", {}).get("path", ""),
        root,
        "launch script",
    )
    for label, value in (
            ("manifest output", manifest.get("output_dir", "")),
            ("manifest cwd", manifest.get("cwd", "")),
            ("manifest data root", manifest.get("data_root", "")),
            ("dataset data root", dataset.get("data_root", "")),
            ("pretrained weight", model.get("pretrained_weight_path", ""))):
        require_contained_path(value, root, label)
    packages_path = environment.get("pip_freeze_path", "")
    if not Path(str(packages_path)).is_absolute():
        packages_path = root / str(packages_path)
    require_contained_path(packages_path, root, "environment package list")

    with source_path.open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle)
    with resolved_path.open("r", encoding="utf-8") as handle:
        resolved = deserialize_cfg_node_yaml(handle.read())
    _validate_config_contained_paths(source, root, "source")
    _validate_config_contained_paths(resolved, root, "resolved")
    return root


class FormalRunLock(object):
    """Cross-process fail-closed lock for one formal repo/output/run identity."""

    def __init__(self, repo_root, output_dir, run_id, lock_root=None):
        default_root = Path(tempfile.gettempdir()) / "bot-reid-formal-run-locks"
        self.lock_root = require_temporary_fixture(lock_root or default_root)
        self.lock_root.mkdir(parents=True, exist_ok=True)
        identity = {
            "repo_root": str(Path(repo_root).resolve()),
            "output_dir": str(Path(output_dir).resolve()),
            "run_id": str(run_id),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.path = self.lock_root / "{}.lock".format(digest)
        self.owner = dict(identity)
        self.owner.update({
            "token": uuid.uuid4().hex,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at_utc": utc_now(),
        })
        self.acquired = False

    def _existing_owner_text(self):
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception as error:
            return "unreadable lock owner ({})".format(error)

    def acquire(self):
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(str(self.path), flags, 0o600)
        except FileExistsError as error:
            raise FormalRunLockError(
                "Formal run lock already exists at {}; owner={}".format(
                    self.path, self._existing_owner_text()
                )
            ) from error
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(self.owner, ensure_ascii=False, sort_keys=True) + "\n"
                )
            self.acquired = True
        except BaseException:
            try:
                self.path.unlink()
            except OSError:
                pass
            raise
        return self

    def release(self):
        if not self.acquired:
            return False
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            self.acquired = False
            return False
        if current.get("token") != self.owner["token"]:
            self.acquired = False
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.acquired = False
        return True

    def __enter__(self):
        return self.acquire()

    def __exit__(self, _error_type, _error, _traceback):
        self.release()
        return False


def formal_run_lock(repo_root, output_dir, run_id, lock_root=None):
    return FormalRunLock(repo_root, output_dir, run_id, lock_root=lock_root)


def _normalized_text(value):
    value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return value


def atomic_write_text(path, value):
    target = assert_path_allowed(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    value = _normalized_text(value)
    temporary = target.with_name("{}.tmp.{}".format(target.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
    os.replace(str(temporary), str(target))
    return target


def atomic_write_bytes(path, value):
    target = assert_path_allowed(path)
    if not isinstance(value, bytes):
        raise TypeError("atomic_write_bytes requires bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("{}.tmp.{}".format(target.name, os.getpid()))
    with temporary.open("wb") as handle:
        handle.write(value)
    os.replace(str(temporary), str(target))
    return target


def atomic_write_json(path, value):
    return atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_json(path):
    target = assert_path_allowed(path)
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    target = assert_path_allowed(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(_normalized_text(value).encode("utf-8")).hexdigest()


def append_validation_record(output_dir, record):
    missing = [field for field in VALIDATION_FIELDS if field not in record]
    if missing:
        raise ValueError("Validation record missing fields: {}".format(missing))
    path = assert_path_allowed(Path(output_dir) / "validation_history.jsonl")
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    atomic_write_text(path, existing + line)
    return path


def read_validation_history(path):
    target = assert_path_allowed(path)
    records = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [field for field in VALIDATION_FIELDS if field not in record]
            if missing:
                raise EvidenceIncompleteError(
                    "Validation line {} missing fields: {}".format(line_number, missing)
                )
            records.append(record)
    if not records:
        raise EvidenceIncompleteError("No structured validation records were found")
    return records


def select_best_validation(records):
    validated = []
    for record in records:
        missing = [field for field in VALIDATION_FIELDS if field not in record]
        if missing:
            raise EvidenceIncompleteError("Validation record missing fields: {}".format(missing))
        copied = dict(record)
        for field in ("rank1_percent", "rank5_percent", "rank10_percent", "map_percent"):
            value = copied[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EvidenceIncompleteError("{} must be numeric".format(field))
            copied[field] = float(value)
        copied["epoch"] = int(copied["epoch"])
        copied["global_iteration"] = int(copied["global_iteration"])
        validated.append(copied)
    return max(validated, key=lambda item: (item["rank1_percent"], item["map_percent"]))


def _split_lines(entries, data_root):
    root = Path(data_root).resolve()
    normalized = []
    for image_path, pid, camid in entries:
        path = Path(image_path)
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            relative = Path(os.path.relpath(str(path), str(root)))
        normalized.append((relative.as_posix(), int(pid), int(camid)))
    normalized.sort(key=lambda item: (item[0], item[1], item[2]))
    text = "".join("{}\t{}\t{}\n".format(*item) for item in normalized)
    return normalized, text


def build_dataset_manifest(splits, data_root, dataset_name, sampler, batch_size,
                           num_instance, num_workers, train_loader_batches,
                           sampler_base_seed, data_loader_generators=None):
    result = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": str(dataset_name),
        "data_root": str(Path(data_root)),
        "sampler": str(sampler),
        "batch_size": int(batch_size),
        "num_instance": int(num_instance),
        "num_workers": int(num_workers),
        "train_loader_batches": int(train_loader_batches),
        "sampler_base_seed": int(sampler_base_seed),
        "sampler_epoch_seed_rule": SAMPLER_EPOCH_SEED_RULE,
        "hash_algorithm": "SHA256",
        "sorting_rule": "within each split sort by (relative_path, pid, camid)",
        "split_manifest_sha256": {},
        "data_loader_generators": dict(data_loader_generators or {}),
    }
    combined = []
    for split_name in ("train", "query", "gallery"):
        entries = list(splits.get(split_name, []))
        normalized, text = _split_lines(entries, data_root)
        result["{}_image_count".format(split_name)] = len(normalized)
        result["{}_pid_count".format(split_name)] = len({item[1] for item in normalized})
        result["{}_camera_count".format(split_name)] = len({item[2] for item in normalized})
        result["split_manifest_sha256"][split_name] = sha256_text(text)
        combined.append("[{}]\n{}".format(split_name, text))
    result["combined_manifest_sha256"] = sha256_text("".join(combined))
    return result


def _package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return NOT_RECORDED


def _nvidia_smi_rows():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in output.decode("utf-8", errors="replace").splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 5:
            continue
        try:
            total_memory = int(cells[3])
        except ValueError:
            total_memory = NOT_RECORDED
        rows.append({
            "index": int(cells[0]) if cells[0].isdigit() else cells[0],
            "name": cells[1] or NOT_RECORDED,
            "uuid": cells[2] or NOT_RECORDED,
            "total_memory_mib": total_memory,
            "driver": cells[4] or NOT_RECORDED,
        })
    return rows


def collect_environment(output_dir):
    output = assert_path_allowed(output_dir)
    packages_path = output / "environment_packages.txt"
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
        packages = completed.stdout.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            packages = NOT_RECORDED + "\n"
    except (OSError, subprocess.TimeoutExpired):
        packages = NOT_RECORDED + "\n"
    if packages and not packages.endswith("\n"):
        packages += "\n"
    atomic_write_text(packages_path, packages)

    gpus = _nvidia_smi_rows()
    if not gpus and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            try:
                properties = torch.cuda.get_device_properties(index)
                gpus.append({
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "uuid": NOT_RECORDED,
                    "total_memory_mib": int(properties.total_memory / (1024 ** 2)),
                    "driver": NOT_RECORDED,
                })
            except Exception:
                gpus.append({
                    "index": index,
                    "name": NOT_RECORDED,
                    "uuid": NOT_RECORDED,
                    "total_memory_mib": NOT_RECORDED,
                    "driver": NOT_RECORDED,
                })
    driver = gpus[0]["driver"] if gpus else NOT_RECORDED
    environment = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": utc_now(),
        "hostname": socket.gethostname() or NOT_RECORDED,
        "os": platform.platform() or NOT_RECORDED,
        "kernel": platform.release() or NOT_RECORDED,
        "machine_architecture": platform.machine() or NOT_RECORDED,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "torchvision_version": _package_version("torchvision"),
        "ignite_version": _package_version("pytorch-ignite"),
        "yacs_version": _package_version("yacs"),
        "numpy_version": np.__version__,
        "pillow_version": _package_version("Pillow"),
        "cuda_runtime": torch.version.cuda or NOT_RECORDED,
        "cudnn_version": cudnn.version() if cudnn.version() is not None else NOT_RECORDED,
        "nvidia_driver": driver,
        "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpus": gpus,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", NOT_RECORDED),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", NOT_RECORDED),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", NOT_RECORDED
        ),
        "timezone": local_timezone(),
        "pip_freeze_path": packages_path.name,
        "pip_freeze_sha256": sha256_file(packages_path),
    }
    return environment


def validate_environment_schema(environment):
    missing = [field for field in REQUIRED_ENVIRONMENT_FIELDS if field not in environment]
    if missing:
        raise EvidenceIncompleteError("Environment manifest missing fields: {}".format(missing))
    return True


def _finite_number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceIncompleteError("{} must be numeric".format(label))
    if not math.isfinite(float(value)):
        raise EvidenceIncompleteError("{} must be finite".format(label))
    if positive and float(value) <= 0:
        raise EvidenceIncompleteError("{} must be greater than zero".format(label))
    return float(value)


def _require_exact_integer(value, expected, label):
    if type(value) is not int or value != expected:
        raise EvidenceIncompleteError(
            "{} must be integer {}, got {!r}".format(label, expected, value)
        )


def _require_close(actual, expected, label, rel_tol=1e-12, abs_tol=1e-9):
    actual_value = _finite_number(actual, label)
    expected_value = _finite_number(expected, "expected {}".format(label))
    if not math.isclose(
            actual_value, expected_value, rel_tol=rel_tol, abs_tol=abs_tol):
        raise EvidenceIncompleteError(
            "{} mismatch: expected {!r}, got {!r}".format(
                label, expected, actual
            )
        )


def _parse_formal_profiler_argv(argv):
    """Parse the exact top-level formal profiler command without defaults."""
    required_options = (
        "--config",
        "--resolved-config",
        "--source-config-sha256",
        "--resolved-config-sha256",
        "--mode",
        "--measurement-seed",
        "--device",
        "--batch-size",
        "--input-height",
        "--input-width",
        "--dtype",
        "--warmup",
        "--measurement-repeats",
        "--num-classes",
        "--output-file",
    )
    required = set(required_options)
    expected_length = 1 + 2 * len(required_options)
    if len(argv) != expected_length:
        raise EvidenceIncompleteError(
            "Formal efficiency argv must contain exactly the declared profiler "
            "script and {} option/value pairs".format(len(required_options))
        )
    parsed = {}
    for index in range(1, len(argv), 2):
        option = argv[index]
        value = argv[index + 1]
        if option not in required:
            raise EvidenceIncompleteError(
                "Formal efficiency argv contains an unknown option: {}".format(option)
            )
        if option in parsed:
            raise EvidenceIncompleteError(
                "Formal efficiency argv repeats option: {}".format(option)
            )
        if value.startswith("--"):
            raise EvidenceIncompleteError(
                "Formal efficiency argv {} has no value".format(option)
            )
        parsed[option] = value
    missing = sorted(required.difference(parsed))
    if missing:
        raise EvidenceIncompleteError(
            "Formal efficiency argv is missing options: {}".format(missing)
        )
    return parsed


def _delta_source_values(baseline, experiment):
    values = {
        "total_parameters": (
            baseline["total_parameters"], experiment["total_parameters"]
        ),
        "trainable_parameters": (
            baseline["trainable_parameters"], experiment["trainable_parameters"]
        ),
        "descriptor_dim": (baseline["feature_dim"], experiment["feature_dim"]),
        "flops": (baseline["flops"], experiment["flops"]),
        "macs": (baseline["macs"], experiment["macs"]),
        "latency_median_ms": (
            baseline["inference_latency_median_ms"],
            experiment["inference_latency_median_ms"],
        ),
        "latency_p95_ms": (
            baseline["inference_latency_p95_ms"],
            experiment["inference_latency_p95_ms"],
        ),
        "throughput_images_per_second": (
            baseline["throughput_images_per_second"],
            experiment["throughput_images_per_second"],
        ),
    }
    for prefix, key in (
            ("memory_forward", "forward_peak_memory"),
            ("memory_forward_backward", "forward_backward_peak_memory")):
        for memory_name in ("peak_allocated_mib", "peak_reserved_mib"):
            values["{}_{}".format(prefix, memory_name)] = (
                baseline[key][memory_name], experiment[key][memory_name]
            )
    return values


def validate_efficiency_profile(profile, formal, source_sha256=None,
                                resolved_sha256=None, source_config_path=None,
                                resolved_config_path=None, environment=None,
                                model_manifest=None,
                                expected_profiler_script_sha256=None):
    _require_fields(profile, REQUIRED_EFFICIENCY_FIELDS, "efficiency_profile.json")
    expected_mode = "formal" if formal else "fixture"
    if profile.get("schema_version") != EFFICIENCY_SCHEMA_VERSION:
        raise EvidenceIncompleteError("Efficiency schema_version must be 2")
    if profile.get("status") != "complete":
        raise EvidenceIncompleteError("Efficiency profiling status must be complete")
    if profile.get("mode") != expected_mode:
        raise EvidenceIncompleteError(
            "Efficiency profiling mode must be {}".format(expected_mode)
        )
    if profile.get("measurement_seed") != EXPECTED_TRAINING_SEED:
        raise EvidenceIncompleteError("Efficiency measurement seed must be 42")
    if profile.get("seed_source") != "explicit --measurement-seed":
        raise EvidenceIncompleteError("Efficiency seed_source is invalid")
    if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            str(profile.get("measurement_timestamp_utc", ""))):
        raise EvidenceIncompleteError("Efficiency measurement timestamp is invalid")
    argv = profile.get("argv")
    if not isinstance(argv, list) or not argv or not all(
            isinstance(item, str) for item in profile["argv"]):
        raise EvidenceIncompleteError("Efficiency argv must be an array of strings")
    for field in (
            "python_executable", "display_command", "profiler_script_sha256",
            "source_config_sha256", "resolved_config_sha256"):
        value = profile.get(field)
        if (not isinstance(value, str) or not value
                or value in (NOT_RECORDED, NOT_APPLICABLE)):
            raise EvidenceIncompleteError("Efficiency {} is missing".format(field))
    for field in (
            "profiler_script_sha256", "source_config_sha256",
            "resolved_config_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", profile[field]):
            raise EvidenceIncompleteError("Efficiency {} is not SHA256".format(field))
    profiler_path = (
        Path(__file__).resolve().parents[1]
        / "tools" / "profile_multi_granularity_part.py"
    ).resolve()
    if not profiler_path.is_file():
        raise EvidenceIncompleteError("Fixed profiler script is missing")
    profiler_sha256 = (
        expected_profiler_script_sha256 or sha256_file(profiler_path)
    )
    if formal and profile["profiler_script_sha256"] != profiler_sha256:
        raise EvidenceIncompleteError("Efficiency profiler script SHA256 mismatch")
    if source_sha256 is not None and profile["source_config_sha256"] != source_sha256:
        raise EvidenceIncompleteError("Efficiency source config SHA256 mismatch")
    if resolved_sha256 is not None and profile["resolved_config_sha256"] != resolved_sha256:
        raise EvidenceIncompleteError("Efficiency resolved config SHA256 mismatch")
    if profile.get("config_overrides") != {
            "MODEL.PRETRAIN_CHOICE": "none",
            "MODEL.PRETRAIN_PATH": "",
    }:
        raise EvidenceIncompleteError(
            "Efficiency profiling must explicitly use random initialization"
        )
    expected_display = shlex.join([profile["python_executable"]] + argv)
    if profile.get("display_command") != expected_display:
        raise EvidenceIncompleteError(
            "Efficiency display_command does not match python_executable and argv"
        )
    if formal:
        if Path(argv[0]).resolve() != profiler_path:
            raise EvidenceIncompleteError(
                "Efficiency argv must execute the fixed profiler script"
            )
        if source_config_path is None or resolved_config_path is None:
            raise EvidenceIncompleteError(
                "Formal efficiency validation requires source/resolved config paths"
            )
        source_path = Path(source_config_path).resolve()
        resolved_path = Path(resolved_config_path).resolve()
        fixed_source_parts = Path(FORMAL_CONFIG_RELATIVE_PATH).parts
        if tuple(source_path.parts[-len(fixed_source_parts):]) != fixed_source_parts:
            raise EvidenceIncompleteError(
                "Formal efficiency source config is not the fixed source config"
            )
        if not source_path.is_file() or not resolved_path.is_file():
            raise EvidenceIncompleteError(
                "Formal efficiency source/resolved config file is missing"
            )
        if resolved_path != resolved_path.parent / "config_resolved.yml":
            raise EvidenceIncompleteError(
                "Formal resolved config must be OUTPUT_DIR/config_resolved.yml"
            )
        actual_source_sha256 = sha256_file(source_path)
        actual_resolved_sha256 = sha256_file(resolved_path)
        if profile["source_config_sha256"] != actual_source_sha256:
            raise EvidenceIncompleteError(
                "Efficiency source config SHA256 does not match the fixed file"
            )
        if profile["resolved_config_sha256"] != actual_resolved_sha256:
            raise EvidenceIncompleteError(
                "Efficiency resolved config SHA256 does not match the run file"
            )
        if source_sha256 is not None and source_sha256 != actual_source_sha256:
            raise EvidenceIncompleteError(
                "Expected source config SHA256 does not match the fixed file"
            )
        if resolved_sha256 is not None and resolved_sha256 != actual_resolved_sha256:
            raise EvidenceIncompleteError(
                "Expected resolved config SHA256 does not match the run file"
            )
        parsed_argv = _parse_formal_profiler_argv(argv)
        expected_argv = {
            "--mode": "formal",
            "--measurement-seed": "42",
            "--device": "cuda",
            "--batch-size": "64",
            "--input-height": "256",
            "--input-width": "128",
            "--dtype": "float32",
            "--warmup": "5",
            "--measurement-repeats": "20",
            "--config": str(source_path),
            "--resolved-config": str(resolved_path),
            "--source-config-sha256": actual_source_sha256,
            "--resolved-config-sha256": actual_resolved_sha256,
        }
        for option, expected_value in expected_argv.items():
            actual_value = parsed_argv[option]
            if option in ("--config", "--resolved-config"):
                matches = Path(actual_value).resolve() == Path(expected_value).resolve()
            else:
                matches = actual_value == expected_value
            if not matches:
                raise EvidenceIncompleteError(
                    "Efficiency argv {} expected {!r}, got {!r}".format(
                        option, expected_value, actual_value
                    )
                )
        expected_output = resolved_path.parent / "efficiency_profile.json"
        if Path(parsed_argv["--output-file"]).resolve() != expected_output:
            raise EvidenceIncompleteError(
                "Efficiency argv output file must be in this run's OUTPUT_DIR"
            )
        if parsed_argv["--num-classes"] != str(
                profile.get("measurement", {}).get("num_classes")):
            raise EvidenceIncompleteError(
                "Efficiency argv num_classes does not match measurement"
            )

    measurement = profile.get("measurement")
    if not isinstance(measurement, dict):
        raise EvidenceIncompleteError("Efficiency measurement must be an object")
    _require_fields(
        measurement, REQUIRED_EFFICIENCY_MEASUREMENT_FIELDS,
        "efficiency_profile.measurement",
    )
    if measurement.get("batch_size") != 64:
        raise EvidenceIncompleteError("Efficiency batch_size must be 64")
    if measurement.get("input_size") != [256, 128]:
        raise EvidenceIncompleteError("Efficiency input_size must be [256, 128]")
    if measurement.get("dtype") != "float32":
        raise EvidenceIncompleteError("Efficiency dtype must be float32")
    if not isinstance(measurement.get("num_classes"), int) or measurement[
            "num_classes"] <= 0:
        raise EvidenceIncompleteError("Efficiency num_classes must be positive")
    if formal:
        if measurement.get("warmup") != 5:
            raise EvidenceIncompleteError("Formal efficiency warmup must be 5")
        if measurement.get("measurement_repeats") != 20:
            raise EvidenceIncompleteError(
                "Formal efficiency measurement_repeats must be 20"
            )
        if measurement.get("worker_isolation") != PROFILER_WORKER_ISOLATION:
            raise EvidenceIncompleteError("Efficiency worker isolation is invalid")
        if measurement.get("operation_count_convention") != OPERATION_COUNT_CONVENTION:
            raise EvidenceIncompleteError("Efficiency operation convention is invalid")
    else:
        if (not isinstance(measurement.get("warmup"), int)
                or measurement["warmup"] < 0):
            raise EvidenceIncompleteError(
                "Efficiency warmup must be a non-negative integer"
            )
        if (not isinstance(measurement.get("measurement_repeats"), int)
                or measurement["measurement_repeats"] <= 0):
            raise EvidenceIncompleteError(
                "Efficiency measurement_repeats must be a positive integer"
            )
    if formal:
        if measurement.get("device") != "cuda":
            raise EvidenceIncompleteError("Formal efficiency profiling requires CUDA")
        for field in (
                "gpu_name", "gpu_total_memory_mib", "nvidia_driver",
                "pytorch_version", "cuda_runtime", "cudnn_version"):
            if measurement.get(field) in (None, "", NOT_RECORDED, NOT_APPLICABLE):
                raise EvidenceIncompleteError(
                    "Formal efficiency field {} must be recorded".format(field)
                )
        _finite_number(
            measurement["gpu_total_memory_mib"],
            "efficiency gpu_total_memory_mib", positive=True,
        )
    elif measurement.get("device") == "cpu":
        for field in ("gpu_name", "gpu_total_memory_mib", "nvidia_driver"):
            if measurement.get(field) != NOT_APPLICABLE:
                raise EvidenceIncompleteError(
                    "Fixture CPU efficiency field {} must be not_applicable".format(field)
                )

    variants = profile.get("variants")
    if not isinstance(variants, list) or len(variants) != 2:
        raise EvidenceIncompleteError("Efficiency profile must contain two variants")
    expected_dimensions = (2048, 2816)
    expected_variants = (
        ("C2-L03", "legacy"),
        ("C2-L03 + Multi-Granularity K={2,4,6}", "multi_granularity"),
    )
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise EvidenceIncompleteError("Efficiency variant must be an object")
        _require_fields(
            variant, REQUIRED_EFFICIENCY_VARIANT_FIELDS,
            "efficiency_profile.variants[{}]".format(index),
        )
        if variant.get("feature_dim") != expected_dimensions[index]:
            raise EvidenceIncompleteError("Efficiency descriptor dimension mismatch")
        if (variant.get("name"), variant.get("variant")) != expected_variants[index]:
            raise EvidenceIncompleteError(
                "Efficiency variant name/order mismatch at index {}".format(index)
            )
        if variant.get("measurement_seed") != EXPECTED_TRAINING_SEED:
            raise EvidenceIncompleteError("Efficiency variant seed mismatch")
        for field in (
                "total_parameters", "trainable_parameters", "flops", "macs",
                "inference_latency_median_ms", "inference_latency_p95_ms",
                "throughput_images_per_second"):
            _finite_number(
                variant.get(field),
                "efficiency variant {} {}".format(index, field),
                positive=True,
            )
        if variant["inference_latency_p95_ms"] < variant["inference_latency_median_ms"]:
            raise EvidenceIncompleteError("Efficiency latency p95 is below median")
        _require_close(
            variant["flops"], 2.0 * variant["macs"],
            "efficiency variant {} FLOPs=2*MACs".format(index),
        )
        expected_throughput = (
            float(measurement["batch_size"]) * 1000.0
            / float(variant["inference_latency_median_ms"])
        )
        _require_close(
            variant["throughput_images_per_second"], expected_throughput,
            "efficiency variant {} throughput".format(index),
        )
        for memory_name in ("forward_peak_memory", "forward_backward_peak_memory"):
            memory = variant.get(memory_name)
            if not isinstance(memory, dict):
                raise EvidenceIncompleteError("Efficiency memory block is missing")
            for field in ("peak_allocated_mib", "peak_reserved_mib"):
                value = memory.get(field)
                if formal:
                    _finite_number(
                        value,
                        "efficiency {} {}".format(memory_name, field),
                        positive=True,
                    )
                elif value != NOT_APPLICABLE:
                    raise EvidenceIncompleteError(
                        "Fixture CPU memory values must be not_applicable"
                    )

    if formal:
        if environment is None or model_manifest is None:
            raise EvidenceIncompleteError(
                "Formal efficiency validation requires environment/model manifests"
            )
        if profile["python_executable"] != environment.get("python_executable"):
            raise EvidenceIncompleteError("Efficiency Python executable mismatch")
        environment_gpus = environment.get("gpus")
        if not isinstance(environment_gpus, list) or not environment_gpus:
            raise EvidenceIncompleteError("Environment GPU evidence is missing")
        primary_gpu = environment_gpus[0]
        correspondence = (
            ("gpu_name", primary_gpu.get("name")),
            ("nvidia_driver", environment.get("nvidia_driver")),
            ("pytorch_version", environment.get("pytorch_version")),
            ("cuda_runtime", environment.get("cuda_runtime")),
            ("cudnn_version", environment.get("cudnn_version")),
        )
        for field, expected_value in correspondence:
            if measurement.get(field) != expected_value:
                raise EvidenceIncompleteError(
                    "Efficiency/environment {} mismatch".format(field)
                )
        if primary_gpu.get("driver") != environment.get("nvidia_driver"):
            raise EvidenceIncompleteError(
                "Environment primary GPU driver evidence is internally inconsistent"
            )
        _require_close(
            measurement["gpu_total_memory_mib"],
            primary_gpu.get("total_memory_mib"),
            "efficiency/environment GPU memory",
        )
        experiment = variants[1]
        for profile_field, model_field in (
                ("feature_dim", "descriptor_dim"),
                ("total_parameters", "total_parameters"),
                ("trainable_parameters", "trainable_parameters")):
            if experiment.get(profile_field) != model_manifest.get(model_field):
                raise EvidenceIncompleteError(
                    "Efficiency/model {} mismatch".format(profile_field)
                )

    deltas = profile.get("deltas")
    if not isinstance(deltas, dict) or not deltas:
        raise EvidenceIncompleteError("Efficiency deltas are missing")
    _require_fields(
        deltas, REQUIRED_EFFICIENCY_DELTA_FIELDS,
        "efficiency_profile.deltas",
    )
    delta_sources = _delta_source_values(variants[0], variants[1])
    if set(deltas) != set(REQUIRED_EFFICIENCY_DELTA_FIELDS):
        raise EvidenceIncompleteError("Efficiency delta set is not exact")
    for name, delta in deltas.items():
        if not isinstance(delta, dict) or set(delta) != {"absolute", "percent"}:
            raise EvidenceIncompleteError("Efficiency delta {} is malformed".format(name))
        for field in ("absolute", "percent"):
            value = delta[field]
            if formal or not name.startswith("memory_"):
                _finite_number(value, "efficiency delta {} {}".format(name, field))
            elif value != NOT_APPLICABLE:
                raise EvidenceIncompleteError(
                    "Fixture CPU memory deltas must be not_applicable"
                )
        baseline_value, experiment_value = delta_sources[name]
        if baseline_value == NOT_APPLICABLE or experiment_value == NOT_APPLICABLE:
            if delta != {"absolute": NOT_APPLICABLE, "percent": NOT_APPLICABLE}:
                raise EvidenceIncompleteError(
                    "Efficiency delta {} must be not_applicable".format(name)
                )
            continue
        baseline_value = float(baseline_value)
        if baseline_value == 0.0:
            raise EvidenceIncompleteError(
                "Efficiency delta {} has zero baseline".format(name)
            )
        expected_absolute = float(experiment_value) - baseline_value
        expected_percent = 100.0 * expected_absolute / baseline_value
        _require_close(
            delta["absolute"], expected_absolute,
            "efficiency delta {} absolute".format(name),
        )
        _require_close(
            delta["percent"], expected_percent,
            "efficiency delta {} percent".format(name),
        )
    return True


def _git_output(repo_root, args):
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root)] + list(args),
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError("Git preflight command failed: {}".format(args)) from error
    return output.decode("utf-8", errors="replace").strip()


def formal_preflight(repo_root, config_path, launch_script_path, output_dir,
                     experiment_family, run_id, evidence_id, resolved_config,
                     record_dir=None):
    repo = assert_path_allowed(repo_root)
    source_config = assert_path_allowed(config_path)
    launch_script = assert_path_allowed(launch_script_path)
    output = assert_path_allowed(output_dir)

    expected_config = (repo / FORMAL_CONFIG_RELATIVE_PATH).resolve()
    if source_config != expected_config:
        raise PreflightError(
            "Formal experiment identity is reserved for config {}; got {}".format(
                expected_config, source_config
            )
        )

    branch = _git_output(repo, ["branch", "--show-current"])
    if branch != EXPECTED_BRANCH:
        raise PreflightError(
            "Formal run requires branch {}, got {}".format(EXPECTED_BRANCH, branch)
        )
    commit = _git_output(repo, ["rev-parse", "HEAD"])
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PreflightError("Training commit must be a full 40-character SHA")
    porcelain = _git_output(repo, ["status", "--porcelain"])
    if porcelain:
        raise PreflightError("Formal run requires a completely clean Git worktree and index")

    expected_identity = (
        (experiment_family, EXPECTED_EXPERIMENT_FAMILY),
        (run_id, EXPECTED_RUN_ID),
        (evidence_id, EXPECTED_EVIDENCE_ID),
    )
    for actual, expected in expected_identity:
        if actual != expected:
            raise PreflightError("Experiment identity must be {}, got {}".format(expected, actual))

    if not source_config.is_file():
        raise PreflightError("Source config does not exist: {}".format(source_config))
    if not launch_script.is_file():
        raise PreflightError("Launch script does not exist: {}".format(launch_script))
    with source_config.open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle)
    if not isinstance(source, dict):
        raise PreflightError("Source config is not a YAML mapping")
    validate_formal_protocol(source, "source")
    validate_formal_protocol(resolved_config, "resolved")
    for dotted_path in FORMAL_INDEPENDENT_FIELDS:
        source_value = _protocol_value(_nested_value(source, dotted_path))
        resolved_value = _protocol_value(
            _nested_value(resolved_config, dotted_path)
        )
        if source_value != resolved_value:
            raise PreflightError(
                "Source/resolved independent field mismatch: {}".format(
                    dotted_path
                )
            )
    source_seed = read_explicit_config_seed(str(source_config))
    resolved_seed = _nested_value(resolved_config, "SEED")
    validate_seed_evidence_chain(
        source_seed,
        resolved_seed,
        resolved_seed,
        expected_seed=EXPECTED_TRAINING_SEED,
    )
    resolved_output = Path(source.get("OUTPUT_DIR", ""))
    if str(resolved_output) != str(output):
        raise PreflightError("Preflight OUTPUT_DIR does not match source config")
    if str(Path(str(_nested_value(resolved_config, "OUTPUT_DIR")))) != str(output):
        raise PreflightError("Preflight OUTPUT_DIR does not match resolved config")
    model = source.get("MODEL", {})
    datasets = source.get("DATASETS", {})
    if str(model.get("PRETRAIN_CHOICE", "")).lower() != "imagenet":
        raise PreflightError(
            "Formal evidence runs must start from ImageNet initialization; "
            "checkpoint resume is forbidden"
        )
    pretrained = assert_path_allowed(model.get("PRETRAIN_PATH", ""))
    data_root = assert_path_allowed(datasets.get("ROOT_DIR", ""))
    if not data_root.is_dir():
        raise PreflightError("Dataset root does not exist: {}".format(data_root))
    if not pretrained.is_file():
        raise PreflightError("ImageNet pretrained weight does not exist: {}".format(pretrained))

    if output.exists():
        entries = list(output.iterdir())
        if entries:
            raise PreflightError("OUTPUT_DIR must not exist or must be empty")
    for collision in (
            "log.txt", "run_manifest.json", "run_status.json",
            "validation_history.jsonl", "checkpoint_manifest.tsv"):
        if (output / collision).exists():
            raise PreflightError("Refusing to overwrite existing run artifact: {}".format(collision))
    if record_dir is not None:
        registry = assert_path_allowed(record_dir)
        for row in _read_delimited(registry / "runs.csv", ","):
            if row.get("run_id") == run_id or row.get("evidence_id") == evidence_id:
                raise PreflightError(
                    "run_id/evidence_id already exists in the independent registry"
                )

    return {
        "branch": branch,
        "training_commit": commit,
        "dirty": False,
        "source_config_path": str(source_config),
        "source_config_sha256": sha256_file(source_config),
        "launch_script_path": str(launch_script),
        "launch_script_sha256": sha256_file(launch_script),
        "pretrained_weight_path": str(pretrained),
        "pretrained_weight_sha256": sha256_file(pretrained),
        "data_root": str(data_root),
        "output_dir": str(output),
        "training_seed": source_seed,
    }


def initialize_run(output_dir, preflight, experiment_family, run_id, evidence_id,
                   argv, shell_command, cwd, dataset, execution_mode="formal",
                   started_at_utc=None):
    output = assert_path_allowed(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started_at = started_at_utc or utc_now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_family": experiment_family,
        "run_id": run_id,
        "evidence_id": evidence_id,
        "branch": preflight["branch"],
        "training_commit": preflight["training_commit"],
        "finalization_commit": NOT_RECORDED,
        "finalization_mode": "standard",
        "dirty": preflight["dirty"],
        "source_config": {
            "path": preflight["source_config_path"],
            "sha256": preflight["source_config_sha256"],
        },
        "resolved_config": {"path": "config_resolved.yml", "sha256": NOT_RECORDED},
        "launch_script": {
            "path": preflight["launch_script_path"],
            "sha256": preflight["launch_script_sha256"],
        },
        "argv": list(argv),
        "shell_launch_command": shell_command,
        "cwd": str(cwd),
        "started_at_utc": started_at,
        "ended_at_utc": NOT_RECORDED,
        "timezone": local_timezone(),
        "total_run_runtime_seconds": NOT_RECORDED,
        "environment_collection_runtime_seconds": NOT_RECORDED,
        "profiling_runtime_seconds": NOT_RECORDED,
        "training_runtime_seconds": NOT_RECORDED,
        "finalization_runtime_seconds": NOT_RECORDED,
        "finalization_runtime_complete": False,
        "runtime_source": "time.monotonic",
        "exit_code": NOT_RECORDED,
        "status": "running",
        "training_seed": EXPECTED_TRAINING_SEED,
        "seed_source": "source_config.SEED",
        "dataset": dataset,
        "data_root": preflight["data_root"],
        "output_dir": str(output),
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "execution_mode": execution_mode,
    }
    status = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": started_at,
        "ended_at_utc": NOT_RECORDED,
        "total_run_runtime_seconds": NOT_RECORDED,
        "environment_collection_runtime_seconds": NOT_RECORDED,
        "profiling_runtime_seconds": NOT_RECORDED,
        "training_runtime_seconds": NOT_RECORDED,
        "finalization_runtime_seconds": NOT_RECORDED,
        "finalization_runtime_complete": False,
        "runtime_source": "time.monotonic",
        "exit_code": NOT_RECORDED,
        "errors": [],
    }
    atomic_write_json(output / "run_manifest.json", manifest)
    atomic_write_json(output / "run_status.json", status)
    return manifest


def finish_run_timing(output_dir, runtimes, exit_code):
    output = assert_path_allowed(output_dir)
    manifest = read_json(output / "run_manifest.json")
    status = read_json(output / "run_status.json")
    ended_at = utc_now()
    state = "training_succeeded_pending_evidence" if exit_code == 0 else "failed"
    required = (
        "total_run_runtime_seconds",
        "environment_collection_runtime_seconds",
        "profiling_runtime_seconds",
        "training_runtime_seconds",
    )
    normalized = {}
    for field in required:
        value = runtimes.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError("{} must be a non-negative explicit duration".format(field))
        normalized[field] = float(value)
    normalized["finalization_runtime_seconds"] = float(
        runtimes.get("finalization_runtime_seconds", 0.0)
    )
    for item in (manifest, status):
        item["ended_at_utc"] = ended_at
        item.update(normalized)
        item["runtime_source"] = "time.monotonic"
        item["exit_code"] = int(exit_code)
        item["status"] = state
    atomic_write_json(output / "run_manifest.json", manifest)
    atomic_write_json(output / "run_status.json", status)


CHECKPOINT_FILENAME_RE = re.compile(
    r"^resnet50_checkpoint_(?P<global_iteration>\d+)\.pt$"
)
LOG_ITERATION_RE = re.compile(
    r"Epoch\[(?P<epoch>\d+)\]\s+Iteration\["
    r"(?P<iteration>\d+)/(?P<iterations_per_epoch>\d+)\]"
)
RECORDER_COMPLETION_RE = re.compile(
    r"^\[experiment recorder\] ended_at_utc=(?P<ended_at_utc>[^,]+), "
    r"exit_code=(?P<exit_code>-?\d+), "
    r"pre_finalization_elapsed_seconds=(?P<total_runtime>[0-9.eE+-]+), "
    r"environment_collection_runtime_seconds=(?P<environment_runtime>[0-9.eE+-]+), "
    r"profiling_runtime_seconds=(?P<profiling_runtime>[0-9.eE+-]+), "
    r"training_runtime_seconds=(?P<training_runtime>[0-9.eE+-]+), "
    r"runtime_source=time\.monotonic$"
)
KNOWN_CONFIG_FINALIZATION_ERROR = (
    "resolved config protocol mismatch: MODEL.IF_LABELSMOOTH "
    "expected 'on', got True"
)


def _positive_evidence_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvidenceIncompleteError(
            "{} must be a positive integer".format(label)
        )
    return value


def _validation_iteration_evidence(validation_records):
    epoch_to_iteration = {}
    iteration_to_epoch = {}
    for index, record in enumerate(validation_records, 1):
        epoch = _positive_evidence_integer(
            record.get("epoch"),
            "Validation record {} epoch".format(index),
        )
        iteration = _positive_evidence_integer(
            record.get("global_iteration"),
            "Validation record {} global_iteration".format(index),
        )
        existing_iteration = epoch_to_iteration.get(epoch)
        if existing_iteration is not None and existing_iteration != iteration:
            raise EvidenceIncompleteError(
                "Validation history maps epoch {} to conflicting global iterations: "
                "{} and {}".format(epoch, existing_iteration, iteration)
            )
        existing_epoch = iteration_to_epoch.get(iteration)
        if existing_epoch is not None and existing_epoch != epoch:
            raise EvidenceIncompleteError(
                "Validation history maps global iteration {} to conflicting epochs: "
                "{} and {}".format(iteration, existing_epoch, epoch)
            )
        epoch_to_iteration[epoch] = iteration
        iteration_to_epoch[iteration] = epoch

    ratios = set()
    for epoch, iteration in epoch_to_iteration.items():
        iterations_per_epoch, remainder = divmod(iteration, epoch)
        if remainder or iterations_per_epoch <= 0:
            raise EvidenceIncompleteError(
                "Validation epoch/iteration mapping is not an exact positive ratio: "
                "epoch {} -> global iteration {}".format(epoch, iteration)
            )
        ratios.add(iterations_per_epoch)
    if len(ratios) != 1:
        raise EvidenceIncompleteError(
            "Validation epoch/iteration ratios are inconsistent: {}".format(
                sorted(ratios)
            )
        )
    return epoch_to_iteration, iteration_to_epoch, ratios.pop()


def _log_iteration_evidence(output):
    log_path = require_contained_path(output / "log.txt", output, "training log")
    if not log_path.is_file():
        return None, 0
    epoch_totals = {}
    observed_totals = set()
    internally_consistent = True
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            match = LOG_ITERATION_RE.search(line)
            if match is None:
                continue
            epoch = int(match.group("epoch"))
            iteration = int(match.group("iteration"))
            total = int(match.group("iterations_per_epoch"))
            if epoch <= 0 or iteration <= 0 or total <= 0 or iteration > total:
                raise EvidenceIncompleteError(
                    "Invalid Iteration evidence in log.txt line {}".format(
                        line_number
                    )
                )
            existing = epoch_totals.get(epoch)
            if existing is not None and existing != total:
                internally_consistent = False
            epoch_totals.setdefault(epoch, total)
            observed_totals.add(total)

    # The progress logger calls len(train_loader) independently from Ignite's
    # epoch accounting.  A dynamic identity sampler can refresh that display
    # denominator at an epoch boundary before the logger's epoch label advances.
    # Such progress text is not authoritative iteration evidence.  Discard the
    # whole log source when it is internally inconsistent; validation history
    # and the dataset manifest must still agree in _iterations_per_epoch.
    if not internally_consistent or len(observed_totals) > 1:
        return None, 0
    return (
        observed_totals.pop() if observed_totals else None,
        len(epoch_totals),
    )


def _structured_iteration_evidence(output):
    manifest_path = require_contained_path(
        output / "dataset_manifest.json", output, "dataset manifest"
    )
    if not manifest_path.is_file():
        return None
    dataset = read_json(manifest_path)
    if "train_loader_batches" not in dataset:
        raise EvidenceIncompleteError(
            "dataset_manifest.json is missing train_loader_batches"
        )
    return _positive_evidence_integer(
        dataset["train_loader_batches"],
        "dataset_manifest.json train_loader_batches",
    )


def _reconcile_training_exit_code(output, manifest, status, log_path):
    """Recover a training success overwritten by the known finalization bug."""
    manifest_exit = manifest.get("exit_code")
    status_exit = status.get("exit_code")
    if manifest_exit == 0 and status_exit == 0:
        return None
    if type(manifest_exit) is not int or type(status_exit) is not int:
        raise EvidenceIncompleteError("Recovery exit_code fields must be integers")
    if manifest_exit != status_exit:
        raise EvidenceIncompleteError(
            "Recovery manifest/status exit_code mismatch"
        )
    errors = status.get("errors")
    if not isinstance(errors, list) or KNOWN_CONFIG_FINALIZATION_ERROR not in errors:
        raise EvidenceIncompleteError(
            "Nonzero exit_code lacks the known post-training finalization error"
        )

    completions = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            match = RECORDER_COMPLETION_RE.fullmatch(line.strip())
            if match is not None:
                completion = match.groupdict()
                completion["line_number"] = line_number
                completions.append(completion)
    if len(completions) != 1:
        raise EvidenceIncompleteError(
            "Recovery requires exactly one recorder completion record"
        )
    completion = completions[0]
    if int(completion["exit_code"]) != 0:
        raise EvidenceIncompleteError(
            "Recorder completion does not prove a zero training exit code"
        )
    if completion["ended_at_utc"] != manifest.get("ended_at_utc") or (
            completion["ended_at_utc"] != status.get("ended_at_utc")):
        raise EvidenceIncompleteError(
            "Recorder completion ended_at_utc does not match run state"
        )

    runtime_fields = (
        ("environment_runtime", "environment_collection_runtime_seconds"),
        ("profiling_runtime", "profiling_runtime_seconds"),
        ("training_runtime", "training_runtime_seconds"),
    )
    for log_field, state_field in runtime_fields:
        _require_close(
            float(completion[log_field]),
            manifest.get(state_field),
            "recorder completion {}".format(state_field),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        _require_close(
            status.get(state_field),
            manifest.get(state_field),
            "run_status {}".format(state_field),
        )
    pre_finalization_elapsed = _finite_number(
        float(completion["total_runtime"]),
        "recorder completion pre_finalization_elapsed_seconds",
        positive=True,
    )
    recorded_total = _finite_number(
        manifest.get("total_run_runtime_seconds"),
        "total_run_runtime_seconds",
        positive=True,
    )
    _require_close(
        status.get("total_run_runtime_seconds"),
        recorded_total,
        "run_status total_run_runtime_seconds",
    )
    if recorded_total < pre_finalization_elapsed or (
            recorded_total - pre_finalization_elapsed > 5.0):
        raise EvidenceIncompleteError(
            "Recorder completion elapsed time is incompatible with run state"
        )

    audit = {
        "schema_version": 1,
        "source": "log.txt experiment recorder completion",
        "log_path": os.path.relpath(str(log_path), str(output)).replace("\\", "/"),
        "log_sha256": sha256_file(log_path),
        "log_line_number": int(completion["line_number"]),
        "manifest_exit_code_before_recovery": manifest_exit,
        "status_exit_code_before_recovery": status_exit,
        "reconciled_training_exit_code": 0,
        "known_finalization_error": KNOWN_CONFIG_FINALIZATION_ERROR,
    }
    for item in (manifest, status):
        item["exit_code"] = 0
        item["status"] = "training_succeeded_pending_evidence"
        item["training_exit_code_reconciliation"] = audit
    return audit


def _iterations_per_epoch(output, validation_records):
    epoch_to_iteration, iteration_to_epoch, validation_value = (
        _validation_iteration_evidence(validation_records)
    )
    log_value, log_points = _log_iteration_evidence(output)
    structured_value = _structured_iteration_evidence(output)

    sources = [("validation_history.jsonl", validation_value)]
    if log_value is not None:
        sources.append(("log.txt", log_value))
    if structured_value is not None:
        sources.append(("dataset_manifest.json", structured_value))
    distinct_values = {value for _source, value in sources}
    if len(distinct_values) != 1:
        raise EvidenceIncompleteError(
            "Conflicting iterations_per_epoch evidence: {}".format(
                ", ".join(
                    "{}={}".format(source, value) for source, value in sources
                )
            )
        )

    evidence_points = len(epoch_to_iteration) + log_points
    if structured_value is not None:
        evidence_points += 1
    if evidence_points < 2:
        raise EvidenceIncompleteError(
            "iterations_per_epoch requires at least two consistent evidence points"
        )
    return (
        validation_value,
        epoch_to_iteration,
        iteration_to_epoch,
    )


def _model_checkpoint_candidates(output):
    candidates = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if not path.name.lower().startswith("resnet50_checkpoint_"):
            continue
        if path.is_symlink():
            raise EvidenceIncompleteError(
                "Checkpoint must not be a symbolic link: {}".format(path.name)
            )
        if not path.is_file():
            continue
        match = CHECKPOINT_FILENAME_RE.fullmatch(path.name)
        if match is None:
            raise EvidenceIncompleteError(
                "Checkpoint filename does not match "
                "resnet50_checkpoint_<global_iteration>.pt: {}".format(path.name)
            )
        require_contained_path(path, output, "checkpoint {}".format(path.name))
        iteration = int(match.group("global_iteration"))
        if iteration <= 0:
            raise EvidenceIncompleteError(
                "Checkpoint global iteration must be positive: {}".format(path.name)
            )
        candidates.append((path, iteration))
    return sorted(candidates, key=lambda item: (item[1], item[0].name))


def model_state_dict_schema(state_dict):
    """Return the exact key/shape/dtype schema for a model state dictionary."""
    if not isinstance(state_dict, dict) or not state_dict:
        raise EvidenceIncompleteError("Model state_dict must be a non-empty mapping")
    schema = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not key:
            raise EvidenceIncompleteError("Model state_dict keys must be non-empty strings")
        if not torch.is_tensor(value):
            raise EvidenceIncompleteError(
                "Model state_dict value is not a tensor: {}".format(key)
            )
        schema[key] = {
            "shape": [int(dimension) for dimension in value.shape],
            "dtype": str(value.dtype),
        }
    return schema


def _extract_checkpoint_model_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise EvidenceIncompleteError("Checkpoint root must be a mapping")
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate
    return checkpoint


def validate_selected_checkpoint(checkpoint_path, model_manifest):
    """Safely load and structurally bind the selected model checkpoint."""
    path = assert_path_allowed(checkpoint_path)
    if not path.is_file():
        raise EvidenceIncompleteError("Selected checkpoint is missing")
    if path.stat().st_size <= 0:
        raise EvidenceIncompleteError("Selected checkpoint is empty")
    try:
        checkpoint = torch.load(
            str(path), map_location="cpu", weights_only=True
        )
    except TypeError as error:
        raise EvidenceIncompleteError(
            "Safe checkpoint loading requires torch.load(weights_only=True)"
        ) from error
    except Exception as error:
        raise EvidenceIncompleteError(
            "Selected checkpoint cannot be safely loaded: {}".format(error)
        ) from error
    state_dict = _extract_checkpoint_model_state_dict(checkpoint)
    actual_schema = model_state_dict_schema(state_dict)
    expected_schema = model_manifest.get("state_dict_schema")
    if not isinstance(expected_schema, dict) or not expected_schema:
        raise EvidenceIncompleteError("Model manifest state_dict_schema is missing")
    for key, entry in expected_schema.items():
        if (not isinstance(key, str) or not isinstance(entry, dict)
                or set(entry) != {"shape", "dtype"}
                or not isinstance(entry["shape"], list)
                or not all(type(value) is int for value in entry["shape"])
                or not isinstance(entry["dtype"], str)):
            raise EvidenceIncompleteError(
                "Model manifest state_dict_schema is malformed at {}".format(key)
            )
    if actual_schema != expected_schema:
        missing = sorted(set(expected_schema).difference(actual_schema))
        extra = sorted(set(actual_schema).difference(expected_schema))
        mismatched = sorted(
            key for key in set(actual_schema).intersection(expected_schema)
            if actual_schema[key] != expected_schema[key]
        )
        raise EvidenceIncompleteError(
            "Selected checkpoint state_dict schema mismatch: missing={}, extra={}, "
            "shape_or_dtype={}".format(missing, extra, mismatched)
        )
    return True


def _checkpoint_manifest_rows(output_dir, validation_records, selected):
    output = assert_path_allowed(output_dir)
    iterations_per_epoch, epoch_to_iteration, iteration_to_epoch = (
        _iterations_per_epoch(output, validation_records)
    )
    selected_epoch = _positive_evidence_integer(
        selected.get("epoch"), "Selected epoch"
    )
    selected_record_iteration = _positive_evidence_integer(
        selected.get("global_iteration"), "Selected global iteration"
    )
    expected_global_iteration = selected_epoch * iterations_per_epoch
    if epoch_to_iteration.get(selected_epoch) != selected_record_iteration:
        raise EvidenceIncompleteError(
            "Selected validation record conflicts with validation history at epoch {}"
            .format(selected_epoch)
        )
    if selected_record_iteration != expected_global_iteration:
        raise EvidenceIncompleteError(
            "Selected epoch {} must map to global iteration {}; validation records {}"
            .format(
                selected_epoch,
                expected_global_iteration,
                selected_record_iteration,
            )
        )

    candidates = _model_checkpoint_candidates(output)
    selected_candidates = [
        path for path, iteration in candidates
        if iteration == expected_global_iteration
    ]
    if len(selected_candidates) != 1:
        raise EvidenceIncompleteError(
            "Selected epoch {} maps to global iteration {} and must bind to exactly "
            "one model checkpoint; found {}".format(
                selected_epoch,
                expected_global_iteration,
                len(selected_candidates),
            )
        )

    rows = []
    candidates_by_iteration = {}
    for path, iteration in candidates:
        candidates_by_iteration.setdefault(iteration, []).append(path)
        epoch, remainder = divmod(iteration, iterations_per_epoch)
        if remainder or epoch <= 0:
            raise EvidenceIncompleteError(
                "Checkpoint iteration is not aligned to the verified "
                "iterations_per_epoch={}: {}".format(
                    iterations_per_epoch, path.name
                )
            )
        if iteration_to_epoch.get(iteration) != epoch:
            raise EvidenceIncompleteError(
                "Checkpoint epoch/iteration mapping is absent or inconsistent: "
                "epoch {} -> global iteration {} ({})".format(
                    epoch, iteration, path.name
                )
            )
        is_selected = iteration == expected_global_iteration
        row = {
            "epoch": epoch,
            "global_iteration": iteration,
            "filename": path.name,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "artifact_type": "model_checkpoint",
            "relative_path": path.name,
            "file_size": path.stat().st_size,
            "sha256": sha256_file(path),
            "selected": "true" if is_selected else "false",
        }
        rows.append(row)

    for epoch, iteration in sorted(epoch_to_iteration.items()):
        matching = candidates_by_iteration.get(iteration, [])
        if len(matching) != 1:
            raise EvidenceIncompleteError(
                "Validation epoch {} maps to global iteration {} and must bind to "
                "exactly one model checkpoint; found {}".format(
                    epoch, iteration, len(matching)
                )
            )
    selected_rows = [row for row in rows if row["selected"] == "true"]
    return rows, selected_rows[0]


def build_checkpoint_manifest(output_dir, validation_records, selected):
    output = assert_path_allowed(output_dir)
    rows, selected_model = _checkpoint_manifest_rows(
        output, validation_records, selected
    )
    _write_delimited(
        output / "checkpoint_manifest.tsv",
        rows,
        ("epoch", "global_iteration", "filename", "path", "size_bytes",
         "sha256", "selected", "artifact_type", "relative_path", "file_size"),
        "\t",
    )
    return rows, selected_model


def _write_delimited(path, rows, fields, delimiter):
    atomic_write_text(path, _render_delimited(rows, fields, delimiter))


def _render_delimited(rows, fields, delimiter):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        delimiter=delimiter,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _read_delimited(path, delimiter):
    target = assert_path_allowed(path)
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _require_fields(mapping, fields, label):
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise EvidenceIncompleteError("{} missing fields: {}".format(label, missing))


def _update_failure_status(output, error, failed=False):
    status_path = output / "run_status.json"
    if not status_path.is_file():
        return
    status = read_json(status_path)
    status["status"] = "failed" if failed else "incomplete"
    status["evidence_status"] = "incomplete"
    status.setdefault("errors", []).append(str(error))
    atomic_write_json(status_path, status)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        manifest["status"] = status["status"]
        manifest["evidence_status"] = "incomplete"
        atomic_write_json(manifest_path, manifest)


def record_run_failure(output_dir, error, failed=True):
    """Persist a terminal non-success state without changing exception semantics."""
    output = assert_path_allowed(output_dir)
    _update_failure_status(output, error, failed=failed)


def _artifact_row(output, artifact_type, path):
    target = assert_path_allowed(path)
    return {
        "artifact_type": artifact_type,
        "relative_path": os.path.relpath(str(target), str(output)).replace("\\", "/"),
        "file_size": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _write_artifact_hashes(output, manifest, checkpoints):
    source_config = Path(manifest["source_config"]["path"])
    launch_script = Path(manifest["launch_script"]["path"])
    artifacts = [
        _artifact_row(output, "source_config", source_config),
        _artifact_row(output, "resolved_config", output / "config_resolved.yml"),
        _artifact_row(output, "launch_script", launch_script),
    ]
    fixed = (
        ("reproducibility", "reproducibility.json"),
        ("run_manifest", "run_manifest.json"),
        ("run_status", "run_status.json"),
        ("environment", "environment.json"),
        ("environment_packages", "environment_packages.txt"),
        ("dataset_manifest", "dataset_manifest.json"),
        ("model_manifest", "model_manifest.json"),
        ("validation_history", "validation_history.jsonl"),
        ("metrics_summary", "metrics_summary.json"),
        ("checkpoint_manifest", "checkpoint_manifest.tsv"),
        ("efficiency_profile", "efficiency_profile.json"),
        ("training_log", "log.txt"),
    )
    for artifact_type, relative in fixed:
        path = output / relative
        if not path.is_file():
            raise EvidenceIncompleteError("Required artifact is missing: {}".format(relative))
        artifacts.append(_artifact_row(output, artifact_type, path))
    repair_files = (
        ("config_repair_manifest", "config_repair_manifest.json"),
        ("pre_type_fix_resolved_config", "config_resolved.pre_type_fix.yml"),
    )
    repair_presence = [
        (output / relative).is_file() for _artifact_type, relative in repair_files
    ]
    if any(repair_presence) and not all(repair_presence):
        raise EvidenceIncompleteError(
            "Config repair manifest and preserved pre-fix config must both exist"
        )
    if all(repair_presence):
        for artifact_type, relative in repair_files:
            artifacts.append(_artifact_row(output, artifact_type, output / relative))
    for row in checkpoints:
        artifacts.append(
            _artifact_row(output, row["artifact_type"], output / row["relative_path"])
        )
    _write_delimited(
        output / "artifact_hashes.tsv",
        artifacts,
        ("artifact_type", "relative_path", "file_size", "sha256"),
        "\t",
    )
    return artifacts


EVIDENCE_FIELDS = (
    "run_id", "evidence_id", "artifact_type", "path", "file_size", "sha256"
)


def _normalized_registry_row(row, fields):
    return {field: str(row.get(field, NOT_RECORDED)) for field in fields}


def _new_evidence_rows(run_id, evidence_id, output, artifacts):
    rows = []
    for artifact in artifacts:
        artifact_path = (output / artifact["relative_path"]).resolve()
        rows.append({
            "run_id": run_id,
            "evidence_id": evidence_id,
            "artifact_type": artifact["artifact_type"],
            "path": str(artifact_path).replace("\\", "/"),
            "file_size": artifact["file_size"],
            "sha256": artifact["sha256"],
        })
    artifact_hashes = output / "artifact_hashes.tsv"
    rows.append({
        "run_id": run_id,
        "evidence_id": evidence_id,
        "artifact_type": "artifact_hashes",
        "path": str(artifact_hashes.resolve()).replace("\\", "/"),
        "file_size": artifact_hashes.stat().st_size,
        "sha256": sha256_file(artifact_hashes),
    })
    return rows


def _missing_evidence_text(run_id):
    return """# Missing Evidence Status

## Recorded

- `{run_id}`: seed chain, explicit runtime, environment, dataset/model manifests,
  structured validation metrics, selected checkpoint binding, and artifact hashes.

## not_recorded

- `result_commit`: the completed result has not been committed yet.

## not_archived

- External off-repository archive copy and archive verification are not yet recorded.

## not_applicable

- `analysis_seed`: no stochastic post-training analysis is part of this run.
- mean+/-SD, confidence intervals, significance, stable-improvement and E4 claims:
  only one predeclared seed is present.

## User action still required

- Commit and push the completed result after review.
- Copy the run evidence to an external archive and record its verification.
- Run at least three predeclared distinct seeds before multi-seed statistics.
""".format(run_id=run_id)


def _registry_candidate_contents(record_dir, run_row, evidence_rows):
    existing_runs = _read_delimited(record_dir / "runs.csv", ",")
    normalized_run = _normalized_registry_row(run_row, RUN_FIELDS)
    matching_runs = [
        row for row in existing_runs if row.get("run_id") == run_row["run_id"]
    ]
    if len(matching_runs) > 1:
        raise EvidenceIncompleteError("Registry contains duplicate run_id rows")
    for row in existing_runs:
        if (row.get("evidence_id") == run_row["evidence_id"]
                and row.get("run_id") != run_row["run_id"]):
            raise EvidenceIncompleteError("Registry evidence_id conflicts with another run")
    if matching_runs:
        if _normalized_registry_row(matching_runs[0], RUN_FIELDS) != normalized_run:
            raise EvidenceIncompleteError(
                "Idempotent finalization conflict: runs.csv content changed"
            )
        candidate_runs = list(existing_runs)
    else:
        candidate_runs = list(existing_runs) + [normalized_run]
    candidate_runs.sort(key=lambda item: item.get("run_id", ""))

    existing_evidence = _read_delimited(
        record_dir / "evidence_manifest.tsv", "\t"
    )
    current_evidence = [
        _normalized_registry_row(row, EVIDENCE_FIELDS)
        for row in existing_evidence if row.get("run_id") == run_row["run_id"]
    ]
    normalized_evidence = [
        _normalized_registry_row(row, EVIDENCE_FIELDS) for row in evidence_rows
    ]

    def evidence_key(item):
        return (
            item["run_id"], item["artifact_type"], item["path"], item["sha256"]
        )

    if current_evidence:
        if sorted(current_evidence, key=evidence_key) != sorted(
                normalized_evidence, key=evidence_key):
            raise EvidenceIncompleteError(
                "Idempotent finalization conflict: evidence hashes or paths changed"
            )
        candidate_evidence = list(existing_evidence)
    else:
        candidate_evidence = list(existing_evidence) + normalized_evidence
    candidate_evidence.sort(key=evidence_key)
    return {
        "evidence_manifest.tsv": _render_delimited(
            candidate_evidence, EVIDENCE_FIELDS, "\t"
        ),
        "missing_evidence.md": _missing_evidence_text(run_row["run_id"]),
        "runs.csv": _render_delimited(candidate_runs, RUN_FIELDS, ","),
    }


def _stage_registry_file(target, content):
    target = assert_path_allowed(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="{}.candidate.".format(target.name), dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_normalized_text(content))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def _commit_registry_file(staged, target):
    os.replace(str(staged), str(assert_path_allowed(target)))


def _restore_registry_file(target, previous_bytes):
    target = assert_path_allowed(target)
    if previous_bytes is None:
        if target.exists():
            target.unlink()
        return
    temporary = _stage_registry_file(target, previous_bytes.decode("utf-8"))
    _commit_registry_file(temporary, target)


def _stage_registry_transaction(record_dir, run_row, evidence_rows):
    record_dir = assert_path_allowed(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    contents = _registry_candidate_contents(record_dir, run_row, evidence_rows)
    order = ("evidence_manifest.tsv", "missing_evidence.md", "runs.csv")
    targets = {name: record_dir / name for name in order}
    previous = {
        name: targets[name].read_bytes() if targets[name].is_file() else None
        for name in order
    }
    staged = {}
    try:
        for name in order:
            staged[name] = _stage_registry_file(targets[name], contents[name])
    except Exception:
        for path in staged.values():
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return {
        "order": order,
        "targets": targets,
        "previous": previous,
        "staged": staged,
    }


def _discard_staged_registry_transaction(transaction):
    for path in transaction["staged"].values():
        try:
            path.unlink()
        except OSError:
            pass
    transaction["staged"].clear()


def _commit_staged_registry_transaction(transaction):
    order = transaction["order"]
    targets = transaction["targets"]
    previous = transaction["previous"]
    staged = transaction["staged"]
    try:
        # The success row is the commit marker and is always replaced last.
        for name in order:
            _commit_registry_file(staged[name], targets[name])
            staged.pop(name, None)
    except Exception:
        for path in staged.values():
            try:
                path.unlink()
            except OSError:
                pass
        for name in reversed(order):
            try:
                _restore_registry_file(targets[name], previous[name])
            except Exception:
                pass
        raise


def _commit_registry_transaction(record_dir, run_row, evidence_rows):
    transaction = _stage_registry_transaction(record_dir, run_row, evidence_rows)
    _commit_staged_registry_transaction(transaction)


def _build_metrics_summary(manifest, selected, selected_checkpoint):
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_rule": CHECKPOINT_SELECTION_RULE,
        "selected_epoch": selected["epoch"],
        "selected_global_iteration": selected_checkpoint["global_iteration"],
        "rank1_percent": selected["rank1_percent"],
        "rank5_percent": selected["rank5_percent"],
        "rank10_percent": selected["rank10_percent"],
        "map_percent": selected["map_percent"],
        "training_runtime_seconds": float(manifest["training_runtime_seconds"]),
        "total_run_runtime_seconds": float(manifest["total_run_runtime_seconds"]),
        "training_seed": EXPECTED_TRAINING_SEED,
        "training_commit": manifest["training_commit"],
        "finalization_commit": manifest.get("finalization_commit", NOT_RECORDED),
        "independent_runs": INDEPENDENT_RUNS,
        "status": TRAINING_COMPLETE,
        "evidence_status": LOCAL_EVIDENCE_PENDING,
        "selected_checkpoint": selected_checkpoint["relative_path"],
        "selected_checkpoint_sha256": selected_checkpoint["sha256"],
    }


def _build_run_row(manifest, dataset, model, environment, selected):
    gpu_names = [
        gpu.get("name", NOT_RECORDED) for gpu in environment.get("gpus", [])
    ]
    return {
        "experiment_family": manifest["experiment_family"],
        "run_id": manifest["run_id"],
        "evidence_id": manifest["evidence_id"],
        "dataset": dataset["dataset_name"],
        "method_variant": "C2-L03 + Multi-Granularity Part K={2,4,6}",
        "aux_lambda": "0.3",
        "mode": "mean",
        "selected_epoch": selected["epoch"],
        "rank1_percent": selected["rank1_percent"],
        "rank5_percent": selected["rank5_percent"],
        "rank10_percent": selected["rank10_percent"],
        "map_percent": selected["map_percent"],
        "re_ranking": selected["re_ranking"],
        "independent_runs": INDEPENDENT_RUNS,
        "training_seed": EXPECTED_TRAINING_SEED,
        "training_commit": manifest["training_commit"],
        "finalization_commit": manifest.get("finalization_commit", NOT_RECORDED),
        "branch": manifest["branch"],
        "training_runtime_seconds": float(manifest["training_runtime_seconds"]),
        "total_run_runtime_seconds": float(manifest["total_run_runtime_seconds"]),
        "gpu": "; ".join(gpu_names) if gpu_names else NOT_RECORDED,
        "descriptor_dim": model["descriptor_dim"],
        "total_parameters": model["total_parameters"],
        "trainable_parameters": model["trainable_parameters"],
        "status": TRAINING_COMPLETE,
        "evidence_status": LOCAL_EVIDENCE_PENDING,
        "result_commit": NOT_RECORDED,
        "archive_status": NOT_ARCHIVED,
        "notes": (
            "single predeclared seed; no mean+/-SD, confidence interval, "
            "significance, stable-improvement or E4 claim"
        ),
    }


def _apply_success_state(manifest, status, selected, selected_checkpoint):
    manifest["selected_epoch"] = selected["epoch"]
    manifest["selected_checkpoint"] = {
        "path": selected_checkpoint["relative_path"],
        "sha256": selected_checkpoint["sha256"],
    }
    manifest["status"] = TRAINING_COMPLETE
    manifest["evidence_status"] = LOCAL_EVIDENCE_PENDING
    manifest["result_commit"] = NOT_RECORDED
    manifest["archive_status"] = NOT_ARCHIVED
    status["status"] = TRAINING_COMPLETE
    status["evidence_status"] = LOCAL_EVIDENCE_PENDING
    status["errors"] = []


def finalize_run(output_dir, record_dir, expected=None,
                 total_runtime_started=None, fixture_root=None,
                 finalization_commit=None,
                 expected_profiler_script_sha256=None):
    output = assert_path_allowed(output_dir)
    records = assert_path_allowed(record_dir)
    if fixture_root is not None:
        fixture = validate_fixture_path_containment(output, records)
        if fixture != Path(fixture_root).resolve():
            raise EvidenceIncompleteError(
                "Dry-run fixture root does not match finalization output"
            )
    expected = dict(expected or {})
    finalization_started = time.monotonic()
    try:
        manifest = read_json(output / "run_manifest.json")
        status = read_json(output / "run_status.json")
        if finalization_commit is not None:
            if not re.fullmatch(r"[0-9a-f]{40}", str(finalization_commit)):
                raise EvidenceIncompleteError(
                    "finalization_commit is not a full SHA"
                )
            recorded_finalization = manifest.get(
                "finalization_commit", NOT_RECORDED
            )
            if recorded_finalization not in (
                    NOT_RECORDED, finalization_commit):
                raise EvidenceIncompleteError(
                    "Recorded finalization_commit conflicts with current recovery"
                )
            manifest["finalization_commit"] = finalization_commit
            status["finalization_commit"] = finalization_commit
        reproducibility = read_json(output / "reproducibility.json")
        environment = read_json(output / "environment.json")
        dataset = read_json(output / "dataset_manifest.json")
        model = read_json(output / "model_manifest.json")
        efficiency = read_json(output / "efficiency_profile.json")

        _require_fields(environment, REQUIRED_ENVIRONMENT_FIELDS, "environment.json")
        _require_fields(dataset, REQUIRED_DATASET_FIELDS, "dataset_manifest.json")
        _require_fields(model, REQUIRED_MODEL_FIELDS, "model_manifest.json")
        if int(status.get("exit_code", -1)) != 0 or int(manifest.get("exit_code", -1)) != 0:
            raise EvidenceIncompleteError("Training exit_code is not zero")
        runtime_fields = (
            "total_run_runtime_seconds",
            "environment_collection_runtime_seconds",
            "profiling_runtime_seconds",
            "training_runtime_seconds",
            "finalization_runtime_seconds",
        )
        for runtime_field in runtime_fields:
            _finite_number(
                manifest.get(runtime_field), runtime_field,
                positive=runtime_field == "training_runtime_seconds",
            )
            if status.get(runtime_field) != manifest.get(runtime_field):
                raise EvidenceIncompleteError(
                    "run_status/runtime mismatch: {}".format(runtime_field)
                )
        if status.get("finalization_runtime_complete") != manifest.get(
                "finalization_runtime_complete"):
            raise EvidenceIncompleteError(
                "run_status finalization completion mismatch"
            )
        if manifest.get("runtime_source") != "time.monotonic":
            raise EvidenceIncompleteError("Runtime fields must come from time.monotonic")

        for field, expected_value in (
                ("experiment_family", EXPECTED_EXPERIMENT_FAMILY),
                ("run_id", EXPECTED_RUN_ID),
                ("evidence_id", EXPECTED_EVIDENCE_ID),
                ("branch", EXPECTED_BRANCH)):
            required = expected.get(field, expected_value)
            if manifest.get(field) != required:
                raise EvidenceIncompleteError("{} mismatch".format(field))
        commit = manifest.get("training_commit", "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise EvidenceIncompleteError("Training commit is not a full SHA")
        if manifest.get("dirty") is not False:
            raise EvidenceIncompleteError("Formal training manifest is dirty")

        source_path = assert_path_allowed(manifest["source_config"]["path"])
        resolved_path = output / "config_resolved.yml"
        source_seed = read_explicit_config_seed(str(source_path))
        with source_path.open("r", encoding="utf-8") as handle:
            source = yaml.safe_load(handle)
        with resolved_path.open("r", encoding="utf-8") as handle:
            resolved = deserialize_cfg_node_yaml(handle.read())
        if not isinstance(source, dict) or not isinstance(resolved, dict):
            raise EvidenceIncompleteError("Source/resolved config is not a mapping")
        validate_formal_protocol(source, "source")
        validate_formal_protocol(resolved, "resolved")
        for dotted_path in FORMAL_INDEPENDENT_FIELDS:
            if _protocol_value(_nested_value(source, dotted_path)) != _protocol_value(
                    _nested_value(resolved, dotted_path)):
                raise EvidenceIncompleteError(
                    "Source/resolved independent field mismatch: {}".format(
                        dotted_path
                    )
                )
        execution_mode = manifest.get("execution_mode")
        if execution_mode not in ("formal", "fixture"):
            raise EvidenceIncompleteError("Execution mode must be formal or fixture")
        if execution_mode == "formal":
            _finite_number(
                manifest.get("profiling_runtime_seconds"),
                "profiling_runtime_seconds", positive=True,
            )
        if execution_mode == "formal":
            expected_source = (
                Path(manifest.get("cwd", "")) / FORMAL_CONFIG_RELATIVE_PATH
            ).resolve()
            if source_path != expected_source:
                raise EvidenceIncompleteError(
                    "Formal run did not use the fixed experiment config"
                )
        validate_seed_evidence_chain(
            source_seed,
            resolved["SEED"],
            reproducibility.get("seed"),
            metadata_seed=reproducibility.get("seed"),
            expected_seed=EXPECTED_TRAINING_SEED,
        )
        seed_chain = reproducibility.get("seed_chain", {})
        if not isinstance(seed_chain, dict):
            raise EvidenceIncompleteError("Reproducibility seed_chain must be an object")
        for key in (
                "source_config_seed", "resolved_config_seed", "applied_training_seed",
                "reproducibility_metadata_seed"):
            if seed_chain.get(key) != EXPECTED_TRAINING_SEED:
                raise EvidenceIncompleteError("Reproducibility seed_chain mismatch: {}".format(key))
        if reproducibility.get("seed_applied_before_data_loading") is not True:
            raise EvidenceIncompleteError(
                "seed_applied_before_data_loading must be true"
            )
        worker_seeding = reproducibility.get("data_loader_worker_seeding")
        if not isinstance(worker_seeding, dict):
            raise EvidenceIncompleteError("DataLoader worker seeding evidence is missing")
        expected_worker_seeding = {
            "enabled": True,
            "scheme": DATA_LOADER_WORKER_SEED_SCHEME,
            "num_workers": 8,
        }
        if worker_seeding != expected_worker_seeding:
            raise EvidenceIncompleteError(
                "DataLoader worker seeding evidence mismatch"
            )
        random_state = reproducibility.get("random_state", {})
        if not isinstance(random_state, dict):
            raise EvidenceIncompleteError("Reproducibility random_state must be an object")
        for key in ("python_random_seeded", "numpy_seeded", "torch_cpu_seeded"):
            if random_state.get(key) is not True:
                raise EvidenceIncompleteError("Random state evidence missing: {}".format(key))
        for key in ("python_random_seed", "numpy_seed", "torch_cpu_seed"):
            if random_state.get(key) != EXPECTED_TRAINING_SEED:
                raise EvidenceIncompleteError("Random seed evidence mismatch: {}".format(key))
        if random_state.get("cudnn_deterministic") is not True:
            raise EvidenceIncompleteError("cudnn.deterministic was not recorded as true")
        if random_state.get("cudnn_benchmark") is not False:
            raise EvidenceIncompleteError("cudnn.benchmark was not recorded as false")
        if execution_mode == "formal":
            _require_exact_integer(
                reproducibility.get("seed"), EXPECTED_TRAINING_SEED,
                "reproducibility.seed",
            )
            _require_exact_integer(
                manifest.get("training_seed"), EXPECTED_TRAINING_SEED,
                "run_manifest.training_seed",
            )
            for key in (
                    "source_config_seed", "resolved_config_seed",
                    "applied_training_seed", "reproducibility_metadata_seed"):
                _require_exact_integer(
                    seed_chain.get(key), EXPECTED_TRAINING_SEED,
                    "seed_chain.{}".format(key),
                )
            for key in (
                    "seed", "python_random_seed", "numpy_seed",
                    "torch_cpu_seed", "torch_cuda_manual_seed_all_seed"):
                _require_exact_integer(
                    random_state.get(key), EXPECTED_TRAINING_SEED,
                    "random_state.{}".format(key),
                )
            for key in (
                    "python_random_seeded", "numpy_seeded", "torch_cpu_seeded",
                    "cuda_available", "torch_cuda_manual_seed_all_called",
                    "torch_cuda_all_seeded", "cudnn_deterministic"):
                if random_state.get(key) is not True:
                    raise EvidenceIncompleteError(
                        "Formal random_state.{} must be true".format(key)
                    )
            if random_state.get("cudnn_benchmark") is not False:
                raise EvidenceIncompleteError(
                    "Formal random_state.cudnn_benchmark must be false"
                )
            gpu_count = environment.get("gpu_count")
            if type(gpu_count) is not int or gpu_count <= 0:
                raise EvidenceIncompleteError(
                    "Formal environment.gpu_count must be a positive integer"
                )
            if random_state["cuda_available"] is not (gpu_count > 0):
                raise EvidenceIncompleteError(
                    "CUDA seed evidence conflicts with environment.gpu_count"
                )
            if efficiency.get("mode") != "formal" or efficiency.get(
                    "measurement", {}).get("device") != "cuda":
                raise EvidenceIncompleteError(
                    "CUDA seed evidence requires formal CUDA profiling"
                )
            _require_exact_integer(
                efficiency.get("measurement_seed"), EXPECTED_TRAINING_SEED,
                "efficiency_profile.measurement_seed",
            )
            if environment.get("pythonhashseed") != str(EXPECTED_TRAINING_SEED):
                raise EvidenceIncompleteError(
                    "Formal environment PYTHONHASHSEED must be string '42'"
                )
            if random_state.get("pythonhashseed") != str(EXPECTED_TRAINING_SEED):
                raise EvidenceIncompleteError(
                    "Formal random_state PYTHONHASHSEED must be string '42'"
                )
            if environment.get("cublas_workspace_config") != ":4096:8":
                raise EvidenceIncompleteError(
                    "Formal environment CUBLAS_WORKSPACE_CONFIG must be :4096:8"
                )
            if random_state.get("cublas_workspace_config") != ":4096:8":
                raise EvidenceIncompleteError(
                    "Formal random_state CUBLAS_WORKSPACE_CONFIG must be :4096:8"
                )
        if environment.get("gpu_count", 0) and random_state.get(
                "torch_cuda_all_seeded") is not True:
            raise EvidenceIncompleteError("torch.cuda.manual_seed_all evidence is missing")
        if str(environment.get("pythonhashseed")) != str(EXPECTED_TRAINING_SEED):
            raise EvidenceIncompleteError("PYTHONHASHSEED must be 42")
        if environment.get("cublas_workspace_config") != ":4096:8":
            raise EvidenceIncompleteError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
        if str(random_state.get("pythonhashseed")) != str(EXPECTED_TRAINING_SEED):
            raise EvidenceIncompleteError("Reproducibility PYTHONHASHSEED must be 42")
        if random_state.get("cublas_workspace_config") != ":4096:8":
            raise EvidenceIncompleteError(
                "Reproducibility CUBLAS_WORKSPACE_CONFIG must be :4096:8"
            )
        if reproducibility.get("random_identity_sampler", {}).get("base_seed") != 42:
            raise EvidenceIncompleteError("Sampler base seed mismatch")
        if reproducibility.get("random_identity_sampler", {}).get(
                "epoch_seed_rule") != SAMPLER_EPOCH_SEED_RULE:
            raise EvidenceIncompleteError("Sampler epoch seed rule mismatch")
        if dataset.get("sampler_base_seed") != EXPECTED_TRAINING_SEED:
            raise EvidenceIncompleteError("Dataset sampler base seed mismatch")
        if dataset.get("sampler_epoch_seed_rule") != SAMPLER_EPOCH_SEED_RULE:
            raise EvidenceIncompleteError("Dataset sampler epoch seed rule mismatch")
        for field, expected_value in (
                ("dataset_name", "market1501"),
                ("sampler", "softmax_triplet"),
                ("batch_size", 64),
                ("num_instance", 4),
                ("num_workers", 8)):
            if dataset.get(field) != expected_value:
                raise EvidenceIncompleteError(
                    "Dataset manifest protocol mismatch: {}".format(field)
                )
        expected_loader_metadata = data_loader_generator_metadata(
            EXPECTED_TRAINING_SEED
        )
        loader_metadata = reproducibility.get("data_loader_generators")
        if loader_metadata != expected_loader_metadata:
            raise EvidenceIncompleteError(
                "Reproducibility DataLoader generator evidence mismatch"
            )
        if dataset.get("data_loader_generators") != expected_loader_metadata:
            raise EvidenceIncompleteError(
                "Dataset DataLoader generator evidence mismatch"
            )
        if manifest.get("training_seed") != EXPECTED_TRAINING_SEED:
            raise EvidenceIncompleteError("run_manifest training_seed mismatch")

        source_hash = sha256_file(source_path)
        resolved_hash = sha256_file(resolved_path)
        if source_hash != manifest["source_config"].get("sha256"):
            raise EvidenceIncompleteError("Source config SHA256 mismatch")
        manifest["resolved_config"] = {
            "path": "config_resolved.yml",
            "sha256": resolved_hash,
        }
        configuration = reproducibility.get("configuration", {})
        if configuration.get("source_file_sha256") != source_hash:
            raise EvidenceIncompleteError("Reproducibility source config SHA256 mismatch")
        if configuration.get("resolved_file_sha256") != resolved_hash:
            raise EvidenceIncompleteError("Reproducibility resolved config SHA256 mismatch")
        validate_efficiency_profile(
            efficiency,
            formal=execution_mode == "formal",
            source_sha256=source_hash,
            resolved_sha256=resolved_hash,
            source_config_path=source_path,
            resolved_config_path=resolved_path,
            environment=environment,
            model_manifest=model,
            expected_profiler_script_sha256=expected_profiler_script_sha256,
        )
        launch_path = assert_path_allowed(manifest["launch_script"]["path"])
        if sha256_file(launch_path) != manifest["launch_script"].get("sha256"):
            raise EvidenceIncompleteError("Launch script SHA256 mismatch")
        if model.get("scales") != [2, 4, 6]:
            raise EvidenceIncompleteError("Model scales must be [2, 4, 6]")
        if model.get("projection_dim") != 256 or model.get("descriptor_dim") != 2816:
            raise EvidenceIncompleteError("Model projection/descriptor dimensions mismatch")
        if model.get("aggregation") != "mean" or model.get("fusion") != "concat":
            raise EvidenceIncompleteError("Model aggregation/fusion mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", str(model.get("pretrained_weight_sha256", ""))):
            raise EvidenceIncompleteError("Pretrained weight SHA256 is missing")
        expected_output_path = Path(str(source["OUTPUT_DIR"])).resolve()
        if expected_output_path != output.resolve():
            raise EvidenceIncompleteError("Config/output manifest path mismatch")
        expected_data_root = Path(str(source["DATASETS"]["ROOT_DIR"])).resolve()
        if (Path(str(manifest.get("data_root", ""))).resolve() != expected_data_root
                or Path(str(dataset.get("data_root", ""))).resolve()
                != expected_data_root):
            raise EvidenceIncompleteError("Config/dataset data root mismatch")
        expected_pretrained = assert_path_allowed(
            source["MODEL"]["PRETRAIN_PATH"]
        )
        if Path(str(model.get("pretrained_weight_path", ""))).resolve() != expected_pretrained:
            raise EvidenceIncompleteError("Config/model pretrained path mismatch")
        if (not expected_pretrained.is_file()
                or sha256_file(expected_pretrained)
                != model.get("pretrained_weight_sha256")):
            raise EvidenceIncompleteError("Pretrained weight SHA256 mismatch")
        if efficiency["measurement"].get("num_classes") != model.get("num_classes"):
            raise EvidenceIncompleteError("Efficiency/model num_classes mismatch")
        validation_path = require_contained_path(
            output / "validation_history.jsonl", output, "validation history"
        )
        validation_records = read_validation_history(validation_path)
        selected = select_best_validation(validation_records)
        checkpoint_rows, selected_checkpoint = build_checkpoint_manifest(
            output, validation_records, selected
        )
        validate_selected_checkpoint(
            output / selected_checkpoint["relative_path"], model
        )
        if selected_checkpoint["sha256"] != sha256_file(
                output / selected_checkpoint["relative_path"]):
            raise EvidenceIncompleteError("Selected checkpoint SHA256 mismatch")
        log_path = output / "log.txt"
        if not log_path.is_file():
            raise EvidenceIncompleteError("Training log is missing")

        if (manifest.get("finalization_runtime_complete") is True
                and (
                    manifest.get("finalization_timing_boundary")
                    != FINALIZATION_TIMING_BOUNDARY
                    or status.get("finalization_timing_boundary")
                    != FINALIZATION_TIMING_BOUNDARY
                )):
            raise EvidenceIncompleteError(
                "Recorded finalization timing boundary is missing or incorrect"
            )

        # Phase 1: produce and validate a complete draft, including artifact
        # hashing and registry candidate staging. Nothing is committed to the
        # success registry in this phase.
        _apply_success_state(manifest, status, selected, selected_checkpoint)
        draft_metrics = _build_metrics_summary(
            manifest, selected, selected_checkpoint
        )
        atomic_write_json(output / "metrics_summary.json", draft_metrics)
        atomic_write_json(output / "run_manifest.json", manifest)
        atomic_write_json(output / "run_status.json", status)
        draft_artifacts = _write_artifact_hashes(
            output, manifest, checkpoint_rows
        )
        draft_run_row = _build_run_row(
            manifest, dataset, model, environment, selected
        )
        draft_evidence_rows = _new_evidence_rows(
            manifest["run_id"], manifest["evidence_id"], output,
            draft_artifacts,
        )
        draft_transaction = _stage_registry_transaction(
            records, draft_run_row, draft_evidence_rows
        )
        _discard_staged_registry_transaction(draft_transaction)

        # This is the explicit timing cutoff. The final atomic seal below is
        # intentionally excluded so updating this value cannot invalidate its
        # own manifest/artifact hashes. Phase 1 has already exercised the same
        # hash and registry-candidate construction path.
        if manifest.get("finalization_runtime_complete") is not True:
            timing_boundary = time.monotonic()
            finalization_runtime = timing_boundary - finalization_started
            if finalization_runtime < 0:
                raise EvidenceIncompleteError(
                    "Finalization monotonic runtime cannot be negative"
                )
            manifest["finalization_runtime_seconds"] = float(finalization_runtime)
            status["finalization_runtime_seconds"] = float(finalization_runtime)
            if total_runtime_started is None:
                total_runtime = float(
                    manifest["total_run_runtime_seconds"] + finalization_runtime
                )
            else:
                total_runtime = float(timing_boundary - total_runtime_started)
                if total_runtime < 0:
                    raise EvidenceIncompleteError(
                        "Total monotonic runtime cannot be negative"
                    )
            manifest["total_run_runtime_seconds"] = total_runtime
            status["total_run_runtime_seconds"] = total_runtime
            ended_at = utc_now()
            manifest["ended_at_utc"] = ended_at
            status["ended_at_utc"] = ended_at
            manifest["finalization_runtime_complete"] = True
            status["finalization_runtime_complete"] = True
            manifest["finalization_timing_boundary"] = FINALIZATION_TIMING_BOUNDARY
            status["finalization_timing_boundary"] = FINALIZATION_TIMING_BOUNDARY
        if manifest["total_run_runtime_seconds"] < (
                manifest["profiling_runtime_seconds"]
                + manifest["training_runtime_seconds"]):
            raise EvidenceIncompleteError(
                "Total runtime cannot be smaller than profiling plus training runtime"
            )

        # Phase 2: regenerate every runtime-dependent artifact and atomically
        # seal evidence. This phase is outside the declared timing boundary.
        metrics = _build_metrics_summary(manifest, selected, selected_checkpoint)
        atomic_write_json(output / "metrics_summary.json", metrics)
        atomic_write_json(output / "run_manifest.json", manifest)
        atomic_write_json(output / "run_status.json", status)
        artifacts = _write_artifact_hashes(output, manifest, checkpoint_rows)
        run_row = _build_run_row(
            manifest, dataset, model, environment, selected
        )
        evidence_rows = _new_evidence_rows(
            manifest["run_id"], manifest["evidence_id"], output, artifacts
        )
        _commit_registry_transaction(records, run_row, evidence_rows)
        return {"manifest": manifest, "metrics": metrics, "run_row": run_row}
    except Exception as error:
        failed = False
        try:
            current_status = read_json(output / "run_status.json")
            failed = int(current_status.get("exit_code", 0)) != 0
        except Exception:
            pass
        _update_failure_status(output, error, failed=failed)
        if isinstance(error, EvidenceIncompleteError):
            raise
        raise EvidenceIncompleteError(str(error)) from error


def _git_commit_object_exists(repo_root, commit):
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", "{}^{{commit}}".format(commit)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _git_commit_is_ancestor(repo_root, ancestor, descendant):
    completed = subprocess.run(
        [
            "git", "-C", str(repo_root), "merge-base", "--is-ancestor",
            str(ancestor), str(descendant),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _git_blob_sha256(repo_root, commit, relative_path):
    try:
        content = subprocess.check_output(
            [
                "git", "-C", str(repo_root), "show",
                "{}:{}".format(commit, relative_path),
            ],
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceIncompleteError(
            "Training commit does not contain {}".format(relative_path)
        ) from error
    return hashlib.sha256(content).hexdigest()


def _replace_profile_resolved_hash(profile, resolved_sha256):
    argv = profile.get("argv")
    if not isinstance(argv, list):
        raise EvidenceIncompleteError("Efficiency argv is missing during recovery")
    positions = [
        index for index, value in enumerate(argv)
        if value == "--resolved-config-sha256"
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise EvidenceIncompleteError(
            "Efficiency argv must contain one resolved config SHA option"
        )
    profile["resolved_config_sha256"] = resolved_sha256
    argv[positions[0] + 1] = resolved_sha256
    python_executable = profile.get("python_executable")
    if not isinstance(python_executable, str) or not python_executable:
        raise EvidenceIncompleteError(
            "Efficiency python_executable is missing during recovery"
        )
    profile["display_command"] = shlex.join([python_executable] + argv)


def _validate_repaired_config_state(output, manifest, reproducibility,
                                    efficiency, repair, finalization_commit):
    resolved_path = output / "config_resolved.yml"
    backup_path = output / "config_resolved.pre_type_fix.yml"
    if not backup_path.is_file() or not resolved_path.is_file():
        raise EvidenceIncompleteError(
            "Recovered resolved config or preserved pre-fix config is missing"
        )
    old_hash = sha256_file(backup_path)
    new_hash = sha256_file(resolved_path)
    if repair.get("repair_reason") != CONFIG_TYPE_REPAIR_REASON:
        raise EvidenceIncompleteError("Config repair reason mismatch")
    if repair.get("training_commit") != manifest.get("training_commit"):
        raise EvidenceIncompleteError("Config repair training_commit mismatch")
    if repair.get("finalization_commit") != finalization_commit:
        raise EvidenceIncompleteError("Config repair finalization_commit mismatch")
    if repair.get("repair_code_commit") != finalization_commit:
        raise EvidenceIncompleteError("Config repair code commit mismatch")
    if repair.get("old_resolved_config", {}).get("sha256") != old_hash:
        raise EvidenceIncompleteError("Preserved pre-fix config SHA256 mismatch")
    if repair.get("new_resolved_config", {}).get("sha256") != new_hash:
        raise EvidenceIncompleteError("Repaired resolved config SHA256 mismatch")
    if manifest.get("resolved_config", {}).get("sha256") != new_hash:
        raise EvidenceIncompleteError("Manifest repaired config SHA256 mismatch")
    if reproducibility.get("configuration", {}).get(
            "resolved_file_sha256") != new_hash:
        raise EvidenceIncompleteError(
            "Reproducibility repaired config SHA256 mismatch"
        )
    if efficiency.get("resolved_config_sha256") != new_hash:
        raise EvidenceIncompleteError("Efficiency repaired config SHA256 mismatch")
    with resolved_path.open("r", encoding="utf-8") as handle:
        restored = deserialize_cfg_node_yaml(handle.read())
    validate_formal_protocol(restored, "repaired resolved")
    return new_hash


def recover_existing_run(output_dir, record_dir, repo_root,
                         fixture_root=None):
    """Repair resolved evidence and finalize one already-completed run only."""
    output = assert_path_allowed(output_dir)
    records = assert_path_allowed(record_dir)
    repo = assert_path_allowed(repo_root)
    if not output.is_dir():
        raise EvidenceIncompleteError("Recovery OUTPUT_DIR does not exist")
    if fixture_root is not None:
        fixture = require_temporary_fixture(fixture_root)
        require_contained_path(output, fixture, "recovery output")
        require_contained_path(records, fixture, "recovery registry")

    manifest = read_json(output / "run_manifest.json")
    status = read_json(output / "run_status.json")
    reproducibility = read_json(output / "reproducibility.json")
    efficiency = read_json(output / "efficiency_profile.json")
    model = read_json(output / "model_manifest.json")
    for field, expected_value in (
            ("run_id", EXPECTED_RUN_ID),
            ("evidence_id", EXPECTED_EVIDENCE_ID),
            ("branch", EXPECTED_BRANCH)):
        if manifest.get(field) != expected_value:
            raise EvidenceIncompleteError(
                "Recovery manifest {} mismatch".format(field)
            )
    if Path(str(manifest.get("output_dir", ""))).resolve() != output.resolve():
        raise EvidenceIncompleteError("Recovery OUTPUT_DIR manifest mismatch")

    training_commit = manifest.get("training_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", str(training_commit)):
        raise EvidenceIncompleteError("Recovery training_commit is not a full SHA")
    if not _git_commit_object_exists(repo, training_commit):
        raise EvidenceIncompleteError("Recovery training_commit does not exist")
    current_branch = _git_output(repo, ["branch", "--show-current"])
    if current_branch != manifest["branch"]:
        raise EvidenceIncompleteError("Recovery branch does not match training branch")
    finalization_commit = _git_output(repo, ["rev-parse", "HEAD"])
    if not re.fullmatch(r"[0-9a-f]{40}", finalization_commit):
        raise EvidenceIncompleteError("Recovery finalization_commit is not a full SHA")
    training_profiler_sha256 = _git_blob_sha256(
        repo, training_commit, "tools/profile_multi_granularity_part.py"
    )
    if efficiency.get("profiler_script_sha256") != training_profiler_sha256:
        raise EvidenceIncompleteError(
            "Recorded profiler does not match the training commit"
        )

    source_path = assert_path_allowed(manifest.get("source_config", {}).get("path", ""))
    if fixture_root is not None:
        require_contained_path(source_path, fixture, "recovery source config")
        require_contained_path(
            manifest.get("launch_script", {}).get("path", ""),
            fixture,
            "recovery launch script",
        )
    if not source_path.is_file():
        raise EvidenceIncompleteError("Recovery source config is missing")
    source_hash = sha256_file(source_path)
    if source_hash != manifest.get("source_config", {}).get("sha256"):
        raise EvidenceIncompleteError("Recovery source config SHA256 mismatch")
    training_source_sha256 = _git_blob_sha256(
        repo, training_commit, FORMAL_CONFIG_RELATIVE_PATH
    )
    if training_source_sha256 != source_hash:
        raise EvidenceIncompleteError(
            "Recovery source config does not match the training commit"
        )
    with source_path.open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle)
    validate_formal_protocol(source, "recovery source")

    log_path = output / "log.txt"
    if not log_path.is_file() or log_path.stat().st_size <= 0:
        raise EvidenceIncompleteError("Recovery training log is missing or empty")
    validation_path = require_contained_path(
        output / "validation_history.jsonl", output, "validation history"
    )
    validation_records = read_validation_history(validation_path)
    selected = select_best_validation(validation_records)
    _checkpoint_rows, selected_checkpoint = _checkpoint_manifest_rows(
        output, validation_records, selected
    )
    selected_path = output / selected_checkpoint["relative_path"]
    validate_selected_checkpoint(selected_path, model)
    selected_hash = sha256_file(selected_path)
    if selected_hash != selected_checkpoint["sha256"]:
        raise EvidenceIncompleteError("Recovery selected checkpoint SHA256 mismatch")
    exit_code_reconciliation = _reconcile_training_exit_code(
        output, manifest, status, log_path
    )

    resolved_path = output / "config_resolved.yml"
    backup_path = output / "config_resolved.pre_type_fix.yml"
    repair_path = output / "config_repair_manifest.json"
    if repair_path.is_file():
        repair = read_json(repair_path)
        repair_commit = repair.get("repair_code_commit")
        if not re.fullmatch(r"[0-9a-f]{40}", str(repair_commit)) or (
                not _git_commit_object_exists(repo, repair_commit)):
            raise EvidenceIncompleteError(
                "Config repair commit is missing or invalid"
            )
        if not _git_commit_is_ancestor(
                repo, repair_commit, finalization_commit):
            raise EvidenceIncompleteError(
                "Current recovery commit does not descend from config repair commit"
            )
        _validate_repaired_config_state(
            output, manifest, reproducibility, efficiency, repair, repair_commit,
        )
        attempts = repair.get("recovery_attempt_commits")
        if attempts is None:
            attempts = [repair_commit]
        if not isinstance(attempts, list) or not all(
                re.fullmatch(r"[0-9a-f]{40}", str(item)) for item in attempts):
            raise EvidenceIncompleteError(
                "Config repair recovery_attempt_commits is invalid"
            )
        if finalization_commit not in attempts:
            attempts.append(finalization_commit)
        repair["recovery_attempt_commits"] = attempts
        if exit_code_reconciliation is not None:
            repair["training_exit_code_reconciliation"] = (
                exit_code_reconciliation
            )
        manifest["finalization_commit"] = finalization_commit
        manifest["finalization_mode"] = "recover_existing_run"
        status["finalization_commit"] = finalization_commit
        status["finalization_mode"] = "recover_existing_run"
        atomic_write_json(output / "run_manifest.json", manifest)
        atomic_write_json(output / "run_status.json", status)
        atomic_write_json(repair_path, repair)
    else:
        if backup_path.exists():
            raise EvidenceIncompleteError(
                "Pre-fix config exists without a repair manifest"
            )
        if not resolved_path.is_file() or resolved_path.stat().st_size <= 0:
            raise EvidenceIncompleteError("Existing resolved config is missing or empty")
        old_bytes = resolved_path.read_bytes()
        old_hash = hashlib.sha256(old_bytes).hexdigest()

        from config import cfg
        local_cfg = cfg.clone()
        local_cfg.merge_from_file(str(source_path))
        local_cfg.freeze()
        repaired_text = serialize_cfg_node_yaml(local_cfg)
        repaired = deserialize_cfg_node_yaml(repaired_text)
        validate_formal_protocol(repaired, "repaired resolved")
        for dotted_path in FORMAL_INDEPENDENT_FIELDS:
            if _protocol_value(_nested_value(source, dotted_path)) != _protocol_value(
                    _nested_value(repaired, dotted_path)):
                raise EvidenceIncompleteError(
                    "Recovery source/resolved independent field mismatch: {}"
                    .format(dotted_path)
                )
        if Path(str(repaired["OUTPUT_DIR"])).resolve() != output.resolve():
            raise EvidenceIncompleteError("Repaired config OUTPUT_DIR mismatch")
        new_hash = sha256_text(repaired_text)
        if new_hash == old_hash:
            raise EvidenceIncompleteError(
                "Existing resolved config already matches the repaired content"
            )

        old_declared_hashes = {
            "run_manifest": manifest.get("resolved_config", {}).get("sha256"),
            "reproducibility": reproducibility.get("configuration", {}).get(
                "resolved_file_sha256"
            ),
            "efficiency_profile": efficiency.get("resolved_config_sha256"),
        }
        atomic_write_bytes(backup_path, old_bytes)
        atomic_write_text(resolved_path, repaired_text)

        manifest["resolved_config"] = {
            "path": "config_resolved.yml",
            "sha256": new_hash,
        }
        manifest["finalization_commit"] = finalization_commit
        manifest["finalization_mode"] = "recover_existing_run"
        status["finalization_commit"] = finalization_commit
        status["finalization_mode"] = "recover_existing_run"
        reproducibility.setdefault("configuration", {})[
            "resolved_file_sha256"
        ] = new_hash
        _replace_profile_resolved_hash(efficiency, new_hash)

        repair = {
            "schema_version": 1,
            "mode": "finalization_only",
            "repair_reason": CONFIG_TYPE_REPAIR_REASON,
            "repaired_at_utc": utc_now(),
            "run_id": manifest["run_id"],
            "evidence_id": manifest["evidence_id"],
            "branch": manifest["branch"],
            "training_commit": training_commit,
            "finalization_commit": finalization_commit,
            "repair_code_commit": finalization_commit,
            "recovery_attempt_commits": [finalization_commit],
            "source_config": {
                "path": str(source_path),
                "sha256": source_hash,
            },
            "old_resolved_config": {
                "path": "config_resolved.pre_type_fix.yml",
                "sha256": old_hash,
                "file_size": len(old_bytes),
                "declared_hashes_before_repair": old_declared_hashes,
            },
            "new_resolved_config": {
                "path": "config_resolved.yml",
                "sha256": new_hash,
                "file_size": len(repaired_text.encode("utf-8")),
            },
            "validated_training_log": {
                "path": "log.txt",
                "sha256": sha256_file(log_path),
            },
            "validated_selected_checkpoint": {
                "path": selected_checkpoint["relative_path"],
                "sha256": selected_hash,
            },
            "training_artifacts_modified": False,
        }
        if exit_code_reconciliation is not None:
            repair["training_exit_code_reconciliation"] = (
                exit_code_reconciliation
            )
        atomic_write_json(output / "run_manifest.json", manifest)
        atomic_write_json(output / "run_status.json", status)
        atomic_write_json(output / "reproducibility.json", reproducibility)
        atomic_write_json(output / "efficiency_profile.json", efficiency)
        atomic_write_json(repair_path, repair)
        _validate_repaired_config_state(
            output, manifest, reproducibility, efficiency, repair,
            finalization_commit,
        )

    return finalize_run(
        output,
        records,
        fixture_root=fixture_root,
        finalization_commit=finalization_commit,
        expected_profiler_script_sha256=training_profiler_sha256,
    )
