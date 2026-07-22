#!/usr/bin/env python
# encoding: utf-8
"""Offline simulation of cross-camera-positive supervision coverage.

This reproduces the RandomIdentitySampler grouping rule over training metadata.
It measures sampling coverage, not model quality, and never reads image pixels.
"""

from __future__ import print_function

import argparse
import csv
import glob
import json
import os
import random
import re
import statistics
from collections import defaultdict

import numpy as np
import yaml


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Simulate full PK-sampling epochs and report the proportion of "
            "anchors that have at least one same-id, different-camera positive."
        )
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument(
        "--compare-config-file",
        default="",
        help="Optional second config; relevant sampler settings must match.",
    )
    parser.add_argument("--dataset", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    return parser


def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _nested(config, *keys, **kwargs):
    default = kwargs.get("default")
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _dataset_name(config, override):
    value = override or _nested(config, "DATASETS", "NAMES", default="market1501")
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value).strip("()[]'\" ").lower()


def _sampler_signature(config, dataset_override="", data_root_override=""):
    return {
        "dataset": _dataset_name(config, dataset_override),
        "data_root": data_root_override
        or str(_nested(config, "DATASETS", "ROOT_DIR", default="./data")),
        "sampler": str(
            _nested(config, "DATALOADER", "SAMPLER", default="softmax")
        ),
        "num_instances": int(
            _nested(config, "DATALOADER", "NUM_INSTANCE", default=16)
        ),
        "batch_size": int(
            _nested(config, "SOLVER", "IMS_PER_BATCH", default=64)
        ),
    }


def _validate_signatures(first, second):
    relevant_keys = ("dataset", "sampler", "num_instances", "batch_size")
    differences = {
        key: (first[key], second[key])
        for key in relevant_keys
        if first[key] != second[key]
    }
    if differences:
        raise ValueError(
            "Configs are not sampling-aligned: {}".format(
                json.dumps(differences, ensure_ascii=False)
            )
        )


def _prepare_new_output_dir(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise FileExistsError(
                "Output path exists and is not a directory: {}".format(path)
            )
        with os.scandir(path) as entries:
            is_nonempty = next(entries, None) is not None
        if is_nonempty:
            raise FileExistsError(
                "Output directory is not empty; choose a new directory so existing "
                "analysis results are not overwritten: {}".format(path)
            )
    else:
        os.makedirs(path)


def _first_existing(candidates):
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise RuntimeError(
        "Training directory not found. Checked:\n{}".format(
            "\n".join(candidates)
        )
    )


def resolve_train_dir(dataset, data_root):
    root = os.path.abspath(os.path.expanduser(data_root))
    if dataset == "market1501":
        return _first_existing(
            [
                os.path.join(root, "market1501", "bounding_box_train"),
                os.path.join(root, "Market1501", "bounding_box_train"),
                os.path.join(root, "bounding_box_train"),
            ]
        )
    if dataset in ("dukemtmc", "dukemtmc-reid"):
        return _first_existing(
            [
                os.path.join(
                    root,
                    "dukemtmc-reid",
                    "DukeMTMC-reID",
                    "bounding_box_train",
                ),
                os.path.join(
                    root, "DukeMTMC-reID", "DukeMTMC-reID", "bounding_box_train"
                ),
                os.path.join(root, "DukeMTMC-reID", "bounding_box_train"),
            ]
        )
    raise ValueError("Unsupported dataset for metadata parser: {}".format(dataset))


def load_train_metadata(dataset, train_dir):
    pattern = re.compile(r"([-\d]+)_c(\d)")
    samples = []
    for image_path in sorted(glob.glob(os.path.join(train_dir, "*.jpg"))):
        match = pattern.search(os.path.basename(image_path))
        if match is None:
            continue
        pid, camid = map(int, match.groups())
        if dataset == "market1501" and pid == -1:
            continue
        samples.append(
            {"image_path": image_path, "pid": pid, "camid": camid - 1}
        )
    if not samples:
        raise RuntimeError("No training JPG metadata found in {}".format(train_dir))
    return samples


def sample_epoch_indices(samples, batch_size, num_instances, py_rng, np_rng):
    if batch_size % num_instances != 0:
        raise ValueError("batch_size must be divisible by num_instances.")
    pids_per_batch = batch_size // num_instances
    index_by_pid = defaultdict(list)
    for index, sample in enumerate(samples):
        index_by_pid[sample["pid"]].append(index)

    chunks_by_pid = {}
    for pid, source_indices in index_by_pid.items():
        indices = list(source_indices)
        if len(indices) < num_instances:
            indices = np_rng.choice(
                indices, size=num_instances, replace=True
            ).tolist()
        py_rng.shuffle(indices)
        chunks_by_pid[pid] = [
            indices[start : start + num_instances]
            for start in range(0, len(indices) - num_instances + 1, num_instances)
        ]

    available_pids = list(index_by_pid.keys())
    final_indices = []
    while len(available_pids) >= pids_per_batch:
        selected_pids = py_rng.sample(available_pids, pids_per_batch)
        for pid in selected_pids:
            final_indices.extend(chunks_by_pid[pid].pop(0))
            if not chunks_by_pid[pid]:
                available_pids.remove(pid)
    return final_indices


def analyze_batch(samples, batch_indices, epoch, batch_index):
    pids = np.asarray([samples[index]["pid"] for index in batch_indices])
    camids = np.asarray([samples[index]["camid"] for index in batch_indices])
    same_pid = pids[:, None] == pids[None, :]
    different_camera = camids[:, None] != camids[None, :]
    not_self = ~np.eye(len(batch_indices), dtype=bool)
    cross_camera_mask = same_pid & different_camera & not_self
    same_id_positive_mask = same_pid & not_self
    valid_anchor = cross_camera_mask.any(axis=1)
    valid_count = int(valid_anchor.sum())

    pid_camera_counts = {}
    for pid in sorted(set(pids.tolist())):
        cameras = camids[pids == pid]
        pid_camera_counts[str(pid)] = {
            str(camera): int((cameras == camera).sum())
            for camera in sorted(set(cameras.tolist()))
        }
    return {
        "epoch": epoch,
        "batch_index": batch_index,
        "batch_size": len(batch_indices),
        "valid_cross_camera_anchor_count": valid_count,
        "valid_cross_camera_anchor_ratio": valid_count / float(len(batch_indices)),
        "has_valid_cross_camera_anchor": valid_count > 0,
        "unique_pid_count": int(len(set(pids.tolist()))),
        "unique_camid_count": int(len(set(camids.tolist()))),
        "cross_camera_positive_ordered_pair_count": int(cross_camera_mask.sum()),
        "all_same_id_positive_ordered_pair_count": int(
            same_id_positive_mask.sum()
        ),
        "pid_camera_counts": json.dumps(
            pid_camera_counts, ensure_ascii=False, sort_keys=True
        ),
    }


def summarize_rows(rows):
    ratios = [row["valid_cross_camera_anchor_ratio"] for row in rows]
    total_anchors = sum(row["batch_size"] for row in rows)
    total_valid = sum(row["valid_cross_camera_anchor_count"] for row in rows)
    zero_batches = sum(
        1 for row in rows if not row["has_valid_cross_camera_anchor"]
    )
    cross_pairs = sum(
        row["cross_camera_positive_ordered_pair_count"] for row in rows
    )
    all_positive_pairs = sum(
        row["all_same_id_positive_ordered_pair_count"] for row in rows
    )
    return {
        "batch_count": len(rows),
        "total_anchor_count": total_anchors,
        "valid_cross_camera_anchor_count": total_valid,
        "weighted_valid_anchor_ratio": total_valid / float(total_anchors),
        "batch_ratio_mean": statistics.mean(ratios),
        "batch_ratio_std_population": statistics.pstdev(ratios),
        "batch_ratio_min": min(ratios),
        "batch_ratio_median": statistics.median(ratios),
        "batch_ratio_max": max(ratios),
        "zero_valid_batch_count": zero_batches,
        "zero_valid_batch_ratio": zero_batches / float(len(rows)),
        "cross_camera_positive_ordered_pair_count": cross_pairs,
        "all_same_id_positive_ordered_pair_count": all_positive_pairs,
        "cross_camera_positive_ordered_pair_ratio": (
            cross_pairs / float(all_positive_pairs)
            if all_positive_pairs
            else 0.0
        ),
    }


def _write_outputs(output_dir, rows, epoch_summaries, overall, metadata):
    os.makedirs(output_dir, exist_ok=True)
    batch_fields = (
        "epoch",
        "batch_index",
        "batch_size",
        "valid_cross_camera_anchor_count",
        "valid_cross_camera_anchor_ratio",
        "has_valid_cross_camera_anchor",
        "unique_pid_count",
        "unique_camid_count",
        "cross_camera_positive_ordered_pair_count",
        "all_same_id_positive_ordered_pair_count",
        "pid_camera_counts",
    )
    with open(
        os.path.join(output_dir, "batch_coverage.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=batch_fields)
        writer.writeheader()
        writer.writerows(rows)

    epoch_fields = ("epoch",) + tuple(
        key for key in epoch_summaries[0].keys() if key != "epoch"
    )
    with open(
        os.path.join(output_dir, "epoch_summary.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=epoch_fields)
        writer.writeheader()
        writer.writerows(epoch_summaries)

    payload = {"metadata": metadata, "overall": overall, "epochs": epoch_summaries}
    with open(
        os.path.join(output_dir, "summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = [
        "# Cross-camera-positive batch coverage",
        "",
        "> This is an offline RandomIdentitySampler coverage simulation. It "
        "measures how often C2 receives valid supervision, not model accuracy.",
        "",
        "## Metadata",
        "",
        "- Dataset: `{}`".format(metadata["dataset"]),
        "- Training images: `{}`".format(metadata["training_image_count"]),
        "- Epochs: `{}`".format(metadata["epochs"]),
        "- Seed: `{}`".format(metadata["seed"]),
        "- Batch size / instances per identity: `{}/{}`".format(
            metadata["batch_size"], metadata["num_instances"]
        ),
        "",
        "## Overall result",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Total anchors | {} |".format(overall["total_anchor_count"]),
        "| Valid cross-camera anchors | {} |".format(
            overall["valid_cross_camera_anchor_count"]
        ),
        "| Weighted valid-anchor ratio | {:.4%} |".format(
            overall["weighted_valid_anchor_ratio"]
        ),
        "| Per-batch ratio mean ± population std | {:.4%} ± {:.4%} |".format(
            overall["batch_ratio_mean"], overall["batch_ratio_std_population"]
        ),
        "| Per-batch ratio min / median / max | {:.4%} / {:.4%} / {:.4%} |".format(
            overall["batch_ratio_min"],
            overall["batch_ratio_median"],
            overall["batch_ratio_max"],
        ),
        "| Zero-valid batches | {} / {} ({:.4%}) |".format(
            overall["zero_valid_batch_count"],
            overall["batch_count"],
            overall["zero_valid_batch_ratio"],
        ),
        "| Ordered cross-camera positive pair ratio | {:.4%} |".format(
            overall["cross_camera_positive_ordered_pair_ratio"]
        ),
        "",
        "`cross_camera_positive_count` in the training code is the valid-anchor "
        "count, not a pair count. The ordered-pair statistic above is reported "
        "separately.",
        "",
    ]
    with open(
        os.path.join(output_dir, "summary.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    config = _read_yaml(args.config_file)
    signature = _sampler_signature(config, args.dataset, args.data_root)
    if signature["sampler"] != "softmax_triplet":
        raise ValueError(
            "Coverage simulation requires DATALOADER.SAMPLER='softmax_triplet'."
        )
    if args.compare_config_file:
        comparison = _sampler_signature(
            _read_yaml(args.compare_config_file), args.dataset, args.data_root
        )
        _validate_signatures(signature, comparison)

    _prepare_new_output_dir(args.output_dir)
    train_dir = resolve_train_dir(signature["dataset"], signature["data_root"])
    samples = load_train_metadata(signature["dataset"], train_dir)
    py_rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)

    all_rows = []
    epoch_summaries = []
    for epoch in range(1, args.epochs + 1):
        indices = sample_epoch_indices(
            samples,
            signature["batch_size"],
            signature["num_instances"],
            py_rng,
            np_rng,
        )
        epoch_rows = []
        for start in range(0, len(indices), signature["batch_size"]):
            batch_indices = indices[start : start + signature["batch_size"]]
            if len(batch_indices) != signature["batch_size"]:
                raise RuntimeError("Sampler produced a partial final batch.")
            epoch_rows.append(
                analyze_batch(
                    samples,
                    batch_indices,
                    epoch,
                    start // signature["batch_size"],
                )
            )
        epoch_summary = summarize_rows(epoch_rows)
        epoch_summary["epoch"] = epoch
        epoch_summaries.append(epoch_summary)
        all_rows.extend(epoch_rows)

    overall = summarize_rows(all_rows)
    metadata = {
        "config_file": args.config_file,
        "compare_config_file": args.compare_config_file or None,
        "dataset": signature["dataset"],
        "data_root": signature["data_root"],
        "resolved_train_dir": train_dir,
        "training_image_count": len(samples),
        "sampler": signature["sampler"],
        "batch_size": signature["batch_size"],
        "num_instances": signature["num_instances"],
        "epochs": args.epochs,
        "seed": args.seed,
        "simulation_scope": "offline sampling coverage; no model/checkpoint used",
        "anchor_definition": (
            "anchor has at least one other sample with same pid and different camid"
        ),
        "pair_counting": "ordered pairs, self excluded",
    }
    _write_outputs(
        args.output_dir, all_rows, epoch_summaries, overall, metadata
    )
    if not 0.0 <= overall["weighted_valid_anchor_ratio"] <= 1.0:
        raise RuntimeError("Weighted valid-anchor ratio is outside [0, 1].")
    print(
        "Coverage analysis written to {} (weighted valid-anchor ratio {:.4%})".format(
            args.output_dir, overall["weighted_valid_anchor_ratio"]
        )
    )


if __name__ == "__main__":
    main()
