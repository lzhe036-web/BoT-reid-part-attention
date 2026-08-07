# encoding: utf-8
"""One reproducibility protocol shared by training, loaders, and recording."""

from __future__ import absolute_import

import numbers
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


UINT32_LIMIT = 2 ** 32
DATA_LOADER_STREAM_OFFSETS = {"train": 0, "validation": 1}
WORKER_SEED_STRATEGY = (
    "torch_initial_seed_mod_2**32_from_explicit_train_or_validation_generator;"
    "seeds_python_numpy_torch"
)
SAMPLER_SEED_STRATEGY = "global_python_numpy_rng_seeded_before_sampler_use"
RUNNER_SEED_ENV = "BOT_REID_RUNNER_SEED"


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("SEED must be an integer in [0, 2**32), got {!r}".format(seed))
    seed = int(seed)
    if seed < 0 or seed >= UINT32_LIMIT:
        raise ValueError("SEED must be an integer in [0, 2**32), got {}".format(seed))
    return seed


def validate_seed_evidence_chain(source_seed, resolved_seed, applied_seed,
                                 metadata_seed=None, expected_seed=None):
    """Validate recorded values without applying or changing any RNG state."""
    values = {
        "source_seed": validate_seed(source_seed),
        "resolved_seed": validate_seed(resolved_seed),
        "applied_seed": validate_seed(applied_seed),
    }
    if metadata_seed is not None:
        values["metadata_seed"] = validate_seed(metadata_seed)
    if expected_seed is not None:
        values["expected_seed"] = validate_seed(expected_seed)
    if len(set(values.values())) != 1:
        raise ValueError(
            "Seed evidence conflict: {}".format(
                ", ".join("{}={}".format(key, value) for key, value in values.items())
            )
        )
    return next(iter(values.values()))


def derive_data_loader_seed(base_seed, stream):
    base_seed = validate_seed(base_seed)
    if stream not in DATA_LOADER_STREAM_OFFSETS:
        raise ValueError("Unknown DataLoader seed stream: {}".format(stream))
    return (base_seed + DATA_LOADER_STREAM_OFFSETS[stream]) % UINT32_LIMIT


def make_data_loader_generator(base_seed, stream):
    """Create an explicitly seeded generator for one DataLoader stream."""
    generator = torch.Generator()
    generator.manual_seed(derive_data_loader_seed(base_seed, stream))
    return generator


def seed_worker(_worker_id):
    """Deterministically seed every RNG used inside a DataLoader worker."""
    worker_seed = int(torch.initial_seed() % UINT32_LIMIT)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloader_generator(base_seed, stream):
    """Training-facing alias kept next to the shared generator implementation."""
    return make_data_loader_generator(base_seed, stream)


def apply_reproducibility(seed):
    """Apply the formal RNG/cuDNN protocol before any training objects exist.

    ``PYTHONHASHSEED`` must already be present in the process environment.  It
    cannot be made effective for the current interpreter by assigning it here,
    so a missing or conflicting inherited value fails closed.
    """
    seed = validate_seed(seed)
    inherited_hash_seed = os.environ.get("PYTHONHASHSEED")
    if inherited_hash_seed != str(seed):
        raise RuntimeError(
            "PYTHONHASHSEED must be set to {} before Python starts; got {!r}"
            .format(seed, inherited_hash_seed)
        )
    runner_seed = os.environ.get(RUNNER_SEED_ENV)
    if runner_seed is not None and validate_seed(int(runner_seed)) != seed:
        raise RuntimeError(
            "Runner/config seed conflict: {}={} but resolved SEED={}"
            .format(RUNNER_SEED_ENV, runner_seed, seed)
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Invoke both CUDA APIs explicitly. They are safe lazy calls before CUDA is
    # initialized and make the applied protocol unambiguous in evidence/tests.
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return {
        "applied_seed": seed,
        "PYTHONHASHSEED": inherited_hash_seed,
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_cpu_seed": seed,
        "torch_cuda_seed": seed,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "application_order": [
            "python_random", "numpy", "torch_cpu", "torch_cuda",
            "torch_cuda_all", "cudnn_deterministic", "cudnn_benchmark",
        ],
    }


def collect_reproducibility_evidence(source_seed, resolved_seed, applied):
    """Build evidence only from the receipt returned by the applied protocol."""
    source_seed = validate_seed(source_seed)
    resolved_seed = validate_seed(resolved_seed)
    applied_seed = validate_seed(applied["applied_seed"])
    validate_seed_evidence_chain(source_seed, resolved_seed, applied_seed)
    current_deterministic = bool(torch.backends.cudnn.deterministic)
    current_benchmark = bool(torch.backends.cudnn.benchmark)
    if current_deterministic is not True or current_benchmark is not False:
        raise RuntimeError("cuDNN reproducibility settings changed after application")
    if os.environ.get("PYTHONHASHSEED") != str(resolved_seed):
        raise RuntimeError("PYTHONHASHSEED changed after reproducibility application")
    runner_seed = os.environ.get(RUNNER_SEED_ENV)
    if runner_seed is not None and validate_seed(int(runner_seed)) != resolved_seed:
        raise RuntimeError("Runner seed changed after reproducibility application")

    evidence = dict(applied)
    evidence.update({
        "schema_version": 1,
        "source_seed": source_seed,
        "resolved_seed": resolved_seed,
        "seed": applied_seed,
        "runner_seed": (
            validate_seed(int(runner_seed)) if runner_seed is not None else None
        ),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "dataloader_worker_seed_base": resolved_seed,
        "dataloader_train_generator_seed": derive_data_loader_seed(
            resolved_seed, "train"
        ),
        "dataloader_validation_generator_seed": derive_data_loader_seed(
            resolved_seed, "validation"
        ),
        "dataloader_worker_seed_strategy": WORKER_SEED_STRATEGY,
        "sampler_seed": resolved_seed,
        "sampler_seed_strategy": SAMPLER_SEED_STRATEGY,
        "cudnn_deterministic": current_deterministic,
        "cudnn_benchmark": current_benchmark,
        "status": "complete",
    })
    return evidence


def write_reproducibility_evidence(output_dir, evidence):
    """Atomically write the training-produced reproducibility receipt."""
    if not output_dir:
        raise ValueError("OUTPUT_DIR is required for reproducibility evidence")
    target = Path(output_dir) / "reproducibility.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("{}.tmp.{}".format(target.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(target))
    return target


def seed_analysis_process(seed):
    """Seed only the independent analysis process that calls this function."""
    seed = validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _plain(value):
    if hasattr(value, "items") and not isinstance(value, dict):
        value = dict(value.items())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def resolved_config_text(cfg):
    return yaml.safe_dump(
        _plain(cfg), allow_unicode=True, default_flow_style=False, sort_keys=True
    )
