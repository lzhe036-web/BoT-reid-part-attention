#!/usr/bin/env python
# encoding: utf-8
"""Controlled post-hoc simulation of legacy PK-sampler anchor coverage.

This tool runs in a separate process after training.  Its output is explicitly
labelled as analysis evidence and is never presented as an observed training
trace.
"""

from __future__ import absolute_import

import argparse
import copy
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.datasets import init_dataset
from utils.experiment_recording import NOT_RECORDED, atomic_write_json, utc_now


def sample_epoch_indices(samples, batch_size, num_instances, python_rng, numpy_rng):
    if batch_size % num_instances:
        raise ValueError("batch size must be divisible by NUM_INSTANCE")
    pids_per_batch = batch_size // num_instances
    index_by_pid = defaultdict(list)
    for index, (_path, pid, _camid) in enumerate(samples):
        index_by_pid[int(pid)].append(index)
    chunks_by_pid = defaultdict(list)
    for pid, source_indices in index_by_pid.items():
        indices = copy.deepcopy(source_indices)
        if len(indices) < num_instances:
            indices = numpy_rng.choice(
                indices, size=num_instances, replace=True
            ).tolist()
        python_rng.shuffle(indices)
        for start in range(0, len(indices) - num_instances + 1, num_instances):
            chunks_by_pid[pid].append(indices[start:start + num_instances])
    available = list(index_by_pid.keys())
    result = []
    while len(available) >= pids_per_batch:
        selected = python_rng.sample(available, pids_per_batch)
        for pid in selected:
            result.extend(chunks_by_pid[pid].pop(0))
            if not chunks_by_pid[pid]:
                available.remove(pid)
    return result


def summarize_batches(samples, indices, batch_size):
    totals = {
        "total_anchor_count": 0,
        "valid_cross_camera_anchor_count": 0,
        "cross_camera_positive_count": 0,
        "same_camera_positive_count": 0,
        "batch_count": 0,
    }
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        if len(batch) != batch_size:
            continue
        pids = np.asarray([samples[index][1] for index in batch])
        camids = np.asarray([samples[index][2] for index in batch])
        same_identity = pids[:, None] == pids[None, :]
        same_camera = camids[:, None] == camids[None, :]
        not_self = ~np.eye(batch_size, dtype=bool)
        cross = same_identity & ~same_camera & not_self
        same = same_identity & same_camera & not_self
        totals["batch_count"] += 1
        totals["total_anchor_count"] += batch_size
        totals["valid_cross_camera_anchor_count"] += int(cross.any(axis=1).sum())
        totals["cross_camera_positive_count"] += int(cross.sum())
        totals["same_camera_positive_count"] += int(same.sum())
    return totals


def analyze(config_file, output, analysis_seed, epochs=10):
    with Path(config_file).open("r", encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle) or {}
    dataset_cfg = configuration.get("DATASETS", {})
    loader_cfg = configuration.get("DATALOADER", {})
    solver_cfg = configuration.get("SOLVER", {})
    dataset_name = dataset_cfg.get("NAMES", "market1501")
    data_root = dataset_cfg.get("ROOT_DIR", "./data")
    dataset = init_dataset(dataset_name, root=data_root)
    samples = list(dataset.train)
    batch_size = int(solver_cfg.get("IMS_PER_BATCH", 64))
    num_instances = int(loader_cfg.get("NUM_INSTANCE", 16))
    python_rng = random.Random(int(analysis_seed))
    numpy_rng = np.random.RandomState(int(analysis_seed))
    aggregate = {
        "total_anchor_count": 0,
        "valid_cross_camera_anchor_count": 0,
        "cross_camera_positive_count": 0,
        "same_camera_positive_count": 0,
        "batch_count": 0,
    }
    for _epoch in range(int(epochs)):
        indices = sample_epoch_indices(
            samples, batch_size, num_instances, python_rng, numpy_rng
        )
        summary = summarize_batches(samples, indices, batch_size)
        for key in aggregate:
            aggregate[key] += summary[key]
    total = aggregate["total_anchor_count"]
    if total <= 0:
        raise ValueError("Controlled sampler analysis observed no anchors")
    valid = aggregate["valid_cross_camera_anchor_count"]
    payload = {
        "schema_version": 1,
        "analysis_time": utc_now(),
        "source": "post_hoc_controlled_sampler_analysis",
        "training_trace_observed": False,
        "analysis_seed": int(analysis_seed),
        "training_seed": configuration.get("SEED", NOT_RECORDED),
        "epochs_simulated": int(epochs),
        "sampler": loader_cfg.get("SAMPLER", NOT_RECORDED),
        "batch_size": batch_size,
        "num_instance": num_instances,
        "batch_count": aggregate["batch_count"],
        "total_anchor_count": total,
        "valid_cross_camera_anchor_count": valid,
        "invalid_cross_camera_anchor_count": total - valid,
        "cross_camera_anchor_coverage": valid / float(total),
        "coverage_percent": 100.0 * valid / float(total),
        "cross_camera_positive_count": aggregate["cross_camera_positive_count"],
        "same_camera_positive_count": aggregate["same_camera_positive_count"],
        "average_cross_camera_positive_per_anchor": (
            aggregate["cross_camera_positive_count"] / float(total)
        ),
        "average_same_camera_positive_per_anchor": (
            aggregate["same_camera_positive_count"] / float(total)
        ),
        "scope_note": (
            "Controlled replay of sampler rules; not claimed to be the exact "
            "historical batch sequence when an applied training seed is unavailable."
        ),
    }
    atomic_write_json(output, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze PK-sampler anchor coverage")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis-seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        analyze(
            args.config_file, args.output, args.analysis_seed, epochs=args.epochs
        )
    except BaseException as error:
        print("Anchor coverage analysis failed: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
