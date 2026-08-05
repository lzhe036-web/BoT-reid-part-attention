# encoding: utf-8
"""Reproducibility helpers shared by training and data loading."""

from __future__ import absolute_import

import datetime
import hashlib
import json
import numbers
import os
import platform
import random
import subprocess
import sys

import numpy as np
import torch
import yaml
from torch.backends import cudnn

from utils.config_serialization import serialize_cfg_node_yaml


UINT32_LIMIT = 2 ** 32
DATA_LOADER_STREAM_OFFSETS = {
    "train": 0,
    "query": 1,
    "gallery": 2,
}
DATA_LOADER_GENERATOR_SEED_RULE = (
    "(base_seed + stream_offset[train=0,query=1,gallery=2]) modulo 2**32"
)


def read_explicit_config_seed(config_file):
    """Read a top-level YAML seed without falling back to config defaults."""
    if not config_file or not os.path.isfile(config_file):
        raise ValueError("Source config file is required for explicit seed validation")
    with open(config_file, "r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle)
    if not isinstance(source, dict) or "SEED" not in source:
        raise ValueError("Source config must explicitly define top-level SEED")
    return validate_seed(source["SEED"])


def validate_seed_evidence_chain(source_seed, resolved_seed, applied_seed,
                                 metadata_seed=None, expected_seed=None):
    """Require every available seed source to identify the same run."""
    values = {
        "source config": validate_seed(source_seed),
        "resolved config": validate_seed(resolved_seed),
        "applied training": validate_seed(applied_seed),
    }
    if metadata_seed is not None:
        values["reproducibility metadata"] = validate_seed(metadata_seed)
    if expected_seed is not None:
        values["expected experiment"] = validate_seed(expected_seed)
    if len(set(values.values())) != 1:
        raise ValueError(
            "Training seed evidence conflict: {}".format(
                ", ".join("{}={}".format(key, value) for key, value in values.items())
            )
        )
    return next(iter(values.values()))


def validate_seed(seed):
    """Return ``seed`` as an int or fail before any training work starts."""
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("SEED must be an integer in [0, 2**32), got {!r}".format(seed))
    seed = int(seed)
    if seed < 0 or seed >= UINT32_LIMIT:
        raise ValueError("SEED must be an integer in [0, 2**32), got {}".format(seed))
    return seed


def derive_data_loader_seed(base_seed, stream):
    """Derive a stable per-stream DataLoader seed without global RNG state."""
    base_seed = validate_seed(base_seed)
    if stream not in DATA_LOADER_STREAM_OFFSETS:
        raise ValueError(
            "Unknown DataLoader seed stream {!r}; expected one of {}".format(
                stream, sorted(DATA_LOADER_STREAM_OFFSETS)
            )
        )
    return (base_seed + DATA_LOADER_STREAM_OFFSETS[stream]) % UINT32_LIMIT


def make_data_loader_generator(base_seed, stream):
    """Return an independently seeded generator for one DataLoader stream."""
    generator = torch.Generator()
    generator.manual_seed(derive_data_loader_seed(base_seed, stream))
    return generator


def data_loader_generator_metadata(base_seed):
    base_seed = validate_seed(base_seed)
    return {
        "seed_derivation_rule": DATA_LOADER_GENERATOR_SEED_RULE,
        "stream_offsets": dict(DATA_LOADER_STREAM_OFFSETS),
        "stream_seeds": {
            stream: derive_data_loader_seed(base_seed, stream)
            for stream in sorted(DATA_LOADER_STREAM_OFFSETS)
        },
        "loader_bindings": {
            "train_loader": "train",
            "validation_query_gallery_loader": "query",
            "gallery_stream": (
                "reserved deterministic stream; query and gallery are combined "
                "by the legacy validation loader"
            ),
        },
        "independent_from_global_torch_rng": True,
    }


def ensure_python_hash_seed(seed, argv=None):
    """Restart once when needed so PYTHONHASHSEED applies to this interpreter."""
    seed = validate_seed(seed)
    expected = str(seed)
    if os.environ.get("PYTHONHASHSEED") == expected:
        return False

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = expected
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    arguments = [sys.executable] + list(sys.argv if argv is None else argv)
    os.execvpe(sys.executable, arguments, environment)
    raise RuntimeError("os.execvpe returned unexpectedly")


def seed_everything(seed):
    """Seed all RNGs used by this project and select deterministic cuDNN paths."""
    seed = validate_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Required by deterministic CUDA matrix multiplication on supported CUDA
    # versions.  Older runtimes ignore it; the exact value is still recorded.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    # benchmark=True can select a different convolution implementation between
    # runs.  These settings are supported by the old PyTorch releases used by
    # the original BoT project, unlike newer strict-determinism APIs.
    cudnn.benchmark = False
    cudnn.deterministic = True

    return {
        "seed": seed,
        "python_random_seed": seed,
        "python_random_seeded": True,
        "numpy_seed": seed,
        "numpy_seeded": True,
        "torch_cpu_seed": seed,
        "torch_cpu_seeded": True,
        "torch_cuda_manual_seed_all_seed": seed if cuda_available else "not_recorded",
        "torch_cuda_manual_seed_all_called": cuda_available,
        "torch_cuda_all_seeded": cuda_available,
        "cuda_available": cuda_available,
        "cudnn_benchmark": bool(cudnn.benchmark),
        "cudnn_deterministic": bool(cudnn.deterministic),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def seed_worker(_worker_id):
    """Seed Python and NumPy inside a PyTorch DataLoader worker process."""
    worker_seed = int(torch.initial_seed() % UINT32_LIMIT)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write(path, value):
    temporary_path = "{}.tmp.{}".format(path, os.getpid())
    with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
    os.replace(temporary_path, path)


def _git_output(repo_dir, args):
    try:
        output = subprocess.check_output(
            ["git", "-C", repo_dir] + list(args),
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.decode("utf-8", errors="replace").strip()


def _git_metadata(repo_dir):
    commit = _git_output(repo_dir, ["rev-parse", "HEAD"])
    branch = _git_output(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    status = _git_output(repo_dir, ["status", "--porcelain"])
    return {
        "repository": os.path.abspath(repo_dir),
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def _gpu_names():
    if not torch.cuda.is_available():
        return []
    try:
        return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception:
        # Metadata collection must not hide the seed record merely because an
        # old CUDA runtime cannot query a device name.
        return []


def write_reproducibility_record(
        output_dir,
        cfg,
        seed_state,
        config_file="",
        source_config_seed=None,
        cli_overrides=None,
        command=None,
        repo_dir=None):
    """Persist the resolved config and a machine-readable run record.

    The record is written before the data loader, model, or optimizer is built.
    A write failure is deliberately allowed to stop training: a run that cannot
    record its seed must not silently proceed.
    """
    if not output_dir:
        raise ValueError("OUTPUT_DIR must be set so the training seed can be recorded")

    config_seed = validate_seed(cfg.SEED)
    recorded_seed = validate_seed(seed_state["seed"])
    if source_config_seed is None:
        raise ValueError(
            "source_config_seed is required and must come from an explicit YAML SEED"
        )
    validated_seed = validate_seed_evidence_chain(
        source_config_seed,
        config_seed,
        recorded_seed,
    )

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    resolved_config = serialize_cfg_node_yaml(cfg)
    resolved_config_path = os.path.join(output_dir, "config_resolved.yml")
    _atomic_write(resolved_config_path, resolved_config)

    config_path = None
    config_sha256 = None
    if config_file:
        config_path = os.path.abspath(config_file)
        if os.path.isfile(config_path):
            config_sha256 = _sha256_file(config_path)

    if repo_dir is None:
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "seed": int(seed_state["seed"]),
        "seed_source": "resolved_config.SEED",
        "seed_applied_before_data_loading": True,
        "seed_chain": {
            "source_config_seed": validated_seed,
            "resolved_config_seed": config_seed,
            "applied_training_seed": recorded_seed,
            "reproducibility_metadata_seed": recorded_seed,
        },
        "random_state": dict(seed_state),
        "data_loader_worker_seeding": {
            "enabled": True,
            "scheme": "torch.initial_seed() modulo 2**32 -> Python random and NumPy",
            "num_workers": int(cfg.DATALOADER.NUM_WORKERS)
            if hasattr(cfg, "DATALOADER") else None,
        },
        "random_identity_sampler": {
            "base_seed": config_seed,
            "epoch_seed_rule": "(base_seed + zero_based_epoch_index) modulo 2**32",
        },
        "data_loader_generators": data_loader_generator_metadata(config_seed),
        "configuration": {
            "source_file": config_path,
            "source_file_sha256": config_sha256,
            "cli_overrides": list(cli_overrides or []),
            "resolved_file": os.path.basename(resolved_config_path),
            "resolved_file_sha256": _sha256_text(resolved_config),
        },
        "command": list(command or []),
        "code": _git_metadata(repo_dir),
        "environment": {
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "torchvision_version": getattr(sys.modules.get("torchvision"), "__version__", None),
            "ignite_version": getattr(sys.modules.get("ignite"), "__version__", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "gpu_names": _gpu_names(),
        },
        "scope_note": (
            "A fixed seed and deterministic cuDNN selection improve repeatability, "
            "but exact equality is not guaranteed across different hardware, drivers, "
            "PyTorch versions, or unsupported nondeterministic operations."
        ),
    }

    metadata_path = os.path.join(output_dir, "reproducibility.json")
    _atomic_write(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return metadata_path, metadata
