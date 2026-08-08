#!/usr/bin/env python
# encoding: utf-8
"""Extract per-sample Global/K2/K4/K6 weights from a checkpoint."""

from __future__ import absolute_import

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from data.datasets import ImageDataset, init_dataset
from data.transforms import build_transforms
from modeling import build_model
from modeling.granularity_fusion import GRANULARITY_LABELS
from utils.experiment_recording import (
    atomic_write_json,
    normalized_path,
    sha256_file,
)
from utils.reproducibility import build_dataloader_generator, seed_worker


CSV_FIELDS = (
    "image_path", "pid", "camid", "weight_global", "weight_k2",
    "weight_k4", "weight_k6", "gate_entropy", "max_granularity",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze a dynamic granularity gating checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split", choices=("train", "query", "gallery", "all"),
        default="all",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    return parser.parse_args(argv)


def _checkpoint_state_dict(payload):
    if isinstance(payload, dict) and payload and all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in payload.items()):
        state_dict = payload
    elif isinstance(payload, dict):
        state_dict = None
        for key in ("model", "state_dict", "model_state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate and all(
                    isinstance(name, str) and torch.is_tensor(value)
                    for name, value in candidate.items()):
                state_dict = candidate
                break
        if state_dict is None:
            raise RuntimeError("checkpoint does not contain a model state_dict")
    else:
        raise RuntimeError("checkpoint payload must be a state_dict mapping")
    keys = tuple(state_dict.keys())
    if keys and all(key.startswith("module.") for key in keys):
        state_dict = {
            key[len("module."):]: value for key, value in state_dict.items()
        }
    return state_dict


def load_checkpoint_strict(model, checkpoint_path):
    """Load every model tensor; missing or unexpected keys fail closed."""
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = _checkpoint_state_dict(payload)
    model.load_state_dict(state_dict, strict=True)
    return model


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _component_statistics(rows):
    statistics = {}
    variances = []
    for label in GRANULARITY_LABELS:
        values = [float(row["weight_{}".format(label)]) for row in rows]
        mean = sum(values) / float(len(values))
        variance = sum((value - mean) ** 2 for value in values) / float(
            len(values)
        )
        variances.append(variance)
        statistics[label] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
            "sample_weight_variance": variance,
        }
    return statistics, variances


def write_gate_analysis_outputs(rows, checkpoint_path, config_path,
                                output_dir, metadata=None):
    """Write auditable CSV/JSON outputs from already extracted real weights."""
    if not rows:
        raise RuntimeError("gate analysis produced no samples")
    normalized_rows = []
    for row in rows:
        weights = [float(row["weight_{}".format(label)])
                   for label in GRANULARITY_LABELS]
        if abs(sum(weights) - 1.0) > 1e-5:
            raise RuntimeError("gate weights must sum to one per sample")
        normalized = {field: row[field] for field in CSV_FIELDS}
        for label in GRANULARITY_LABELS:
            normalized["weight_{}".format(label)] = "{:.10f}".format(
                float(row["weight_{}".format(label)])
            )
        normalized["gate_entropy"] = "{:.10f}".format(
            float(row["gate_entropy"])
        )
        normalized_rows.append(normalized)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _write_csv(
        output_dir / "granularity_gating_per_sample.csv", normalized_rows
    )
    statistics, variances = _component_statistics(normalized_rows)
    metadata = dict(metadata or {})
    summary = {
        "schema_version": 1,
        "analysis": "checkpoint_per_sample_granularity_gating",
        "adaptive_evidence_requires_trained_checkpoint": True,
        "random_initialization_is_not_adaptive_evidence": True,
        "sample_count": len(normalized_rows),
        "component_labels": list(GRANULARITY_LABELS),
        "component_statistics": statistics,
        "sample_weight_variance": {
            label: variance
            for label, variance in zip(GRANULARITY_LABELS, variances)
        },
        "mean_sample_weight_variance": sum(variances) / len(variances),
        "checkpoint": normalized_path(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config": normalized_path(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "per_sample_csv": normalized_path(csv_path.resolve()),
        "per_sample_csv_sha256": sha256_file(csv_path),
    }
    summary.update(metadata)
    summary_path = output_dir / "granularity_gating_summary.json"
    atomic_write_json(summary_path, summary)
    return csv_path, summary_path


def _git_value(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT)] + list(args),
        stderr=subprocess.DEVNULL,
    ).decode("utf-8", errors="replace").strip()


def _analysis_collate(batch):
    images, pids, camids, paths = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.tensor(pids, dtype=torch.int64),
        torch.tensor(camids, dtype=torch.int64),
        tuple(paths),
    )


def _split_samples(dataset, split):
    if split == "train":
        return dataset.train
    if split == "query":
        return dataset.query
    if split == "gallery":
        return dataset.gallery
    return dataset.query + dataset.gallery


def _resolved_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analyze(args):
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(config_path))
    if not local_cfg.MODEL.MULTI_GRANULARITY_FUSION:
        raise RuntimeError("granularity fusion must be enabled")
    if local_cfg.MODEL.MULTI_GRANULARITY_FUSION_MODE != "dynamic":
        raise RuntimeError("per-sample gate analysis requires dynamic mode")

    dataset = init_dataset(
        local_cfg.DATASETS.NAMES, root=local_cfg.DATASETS.ROOT_DIR
    )
    samples = _split_samples(dataset, args.split)
    transforms = build_transforms(local_cfg, is_train=False)
    loader = DataLoader(
        ImageDataset(samples, transforms),
        batch_size=local_cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=local_cfg.DATALOADER.NUM_WORKERS,
        collate_fn=_analysis_collate,
        worker_init_fn=seed_worker,
        generator=build_dataloader_generator(local_cfg.SEED, "validation"),
    )

    local_cfg.defrost()
    local_cfg.MODEL.PRETRAIN_CHOICE = "none"
    local_cfg.MODEL.PRETRAIN_PATH = ""
    local_cfg.freeze()
    model = build_model(local_cfg, dataset.num_train_pids)
    load_checkpoint_strict(model, checkpoint_path)
    device = _resolved_device(args.device)
    model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for images, pids, camids, paths in loader:
            _descriptor, details = model.forward_with_granularity_details(
                images.to(device)
            )
            weights = details["weights"].detach().cpu()
            entropy = -torch.sum(
                weights * torch.log(weights.clamp_min(1e-12)), dim=1
            )
            maxima = torch.argmax(weights, dim=1)
            for index, image_path in enumerate(paths):
                row = {
                    "image_path": normalized_path(Path(image_path).resolve()),
                    "pid": int(pids[index]),
                    "camid": int(camids[index]),
                    "gate_entropy": float(entropy[index]),
                    "max_granularity": GRANULARITY_LABELS[int(maxima[index])],
                }
                for column, label in enumerate(GRANULARITY_LABELS):
                    row["weight_{}".format(label)] = float(
                        weights[index, column]
                    )
                rows.append(row)

    metadata = {
        "seed": int(local_cfg.SEED),
        "branch": _git_value("branch", "--show-current"),
        "commit": _git_value("rev-parse", "HEAD"),
        "fusion_mode": "dynamic",
        "split": args.split,
    }
    return write_gate_analysis_outputs(
        rows, checkpoint_path, config_path, args.output_dir, metadata
    )


def main(argv=None):
    csv_path, summary_path = analyze(parse_args(argv))
    print("Per-sample gate weights: {}".format(csv_path))
    print("Gate summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
