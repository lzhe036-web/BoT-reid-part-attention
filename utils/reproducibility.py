# encoding: utf-8
"""Read-only seed validation and deterministic post-hoc analysis helpers.

Nothing in the formal training wrapper imports a function that seeds or
reconfigures the training process.  ``seed_analysis_process`` is used only by
standalone analysis subprocesses after training has exited.
"""

from __future__ import absolute_import

import numbers
import random

import numpy as np
import torch
import yaml


UINT32_LIMIT = 2 ** 32
DATA_LOADER_STREAM_OFFSETS = {"train": 0, "validation": 1}


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
        raise ValueError("Unknown analysis DataLoader seed stream: {}".format(stream))
    return (base_seed + DATA_LOADER_STREAM_OFFSETS[stream]) % UINT32_LIMIT


def make_data_loader_generator(base_seed, stream):
    """Create a generator for a standalone post-hoc analysis DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(derive_data_loader_seed(base_seed, stream))
    return generator


def seed_worker(_worker_id):
    """Seed workers belonging to a standalone post-hoc analysis DataLoader."""
    worker_seed = int(torch.initial_seed() % UINT32_LIMIT)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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
