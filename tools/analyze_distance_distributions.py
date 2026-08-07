#!/usr/bin/env python
# encoding: utf-8
"""Compute auditable Re-ID feature-distance distributions from a checkpoint."""

from __future__ import absolute_import

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from data.collate_batch import val_collate_fn
from data.datasets import ImageDataset, init_dataset
from data.transforms import build_transforms
from modeling.baseline import Baseline
from utils.experiment_recording import atomic_write_json, sha256_file, utc_now
from utils.reproducibility import make_data_loader_generator, seed_worker, validate_seed


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Distance category has no observed pairs")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def compute_distance_statistics(features, pids, camids):
    features = np.asarray(features, dtype=np.float64)
    pids = np.asarray(pids).reshape(-1)
    camids = np.asarray(camids).reshape(-1)
    if features.ndim != 2 or len(features) != len(pids) or len(pids) != len(camids):
        raise ValueError("features, pids, and camids have inconsistent shapes")
    squared = np.sum(features * features, axis=1, keepdims=True)
    distances = np.sqrt(np.maximum(
        squared + squared.T - 2.0 * np.dot(features, features.T), 0.0
    ))
    upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    same_id = pids[:, None] == pids[None, :]
    same_camera = camids[:, None] == camids[None, :]
    same_same = distances[upper & same_id & same_camera]
    same_cross = distances[upper & same_id & ~same_camera]
    different = distances[upper & ~same_id]
    positive = distances[upper & same_id]
    negative = different
    result = {
        "same_id_same_camera": _distribution(same_same),
        "same_id_cross_camera": _distribution(same_cross),
        "different_id": _distribution(different),
        "positive": _distribution(positive),
        "negative": _distribution(negative),
    }
    result.update({
        "same_id_same_camera_mean": result["same_id_same_camera"]["mean"],
        "same_id_same_camera_std": result["same_id_same_camera"]["std"],
        "same_id_cross_camera_mean": result["same_id_cross_camera"]["mean"],
        "same_id_cross_camera_std": result["same_id_cross_camera"]["std"],
        "different_id_mean": result["different_id"]["mean"],
        "different_id_std": result["different_id"]["std"],
        "cross_camera_gap": (
            result["same_id_cross_camera"]["mean"]
            - result["same_id_same_camera"]["mean"]
        ),
    })
    return result


def _checkpoint_state(path):
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint is not a mapping")
    for key in ("model", "state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            payload = candidate
            break
    if not payload or not all(torch.is_tensor(value) for value in payload.values()):
        raise ValueError("Checkpoint does not contain a model state dict")
    result = {}
    for key, value in payload.items():
        clean = key[7:] if key.startswith("module.") else key
        result[clean] = value
    return result


def build_checkpoint_model(local_cfg, checkpoint, device):
    state = _checkpoint_state(checkpoint)
    classifier = state.get("classifier.weight")
    if classifier is None:
        raise ValueError("classifier.weight is absent; num_classes is unknown")
    model = Baseline(
        int(classifier.shape[0]),
        local_cfg.MODEL.LAST_STRIDE,
        "",
        local_cfg.MODEL.NECK,
        local_cfg.TEST.NECK_FEAT,
        local_cfg.MODEL.NAME,
        "none",
        part_attention=bool(local_cfg.MODEL.PART_ATTENTION),
        part_attention_parts=int(local_cfg.MODEL.PART_ATTENTION_PARTS),
        multi_granularity_local=bool(
            local_cfg.MODEL.MULTI_GRANULARITY_LOCAL
        ),
        multi_granularity_scales=local_cfg.MODEL.MULTI_GRANULARITY_SCALES,
        multi_granularity_dim=int(local_cfg.MODEL.MULTI_GRANULARITY_DIM),
        multi_granularity_aggregation=(
            local_cfg.MODEL.MULTI_GRANULARITY_AGGREGATION
        ),
    )
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError("Strict checkpoint load failed: {}".format(incompatibility))
    model.to(device).eval()
    return model


def _sample_metadata(samples, maximum, seed):
    if len(samples) <= maximum:
        return list(samples)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(samples)), maximum))
    return [samples[index] for index in indices]


def analyze_checkpoint(config_file, checkpoint, output, seed, max_samples=4096):
    seed = validate_seed(seed)
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(config_file))
    local_cfg.freeze()
    dataset = init_dataset(
        local_cfg.DATASETS.NAMES, root=local_cfg.DATASETS.ROOT_DIR
    )
    samples = _sample_metadata(
        dataset.query + dataset.gallery, int(max_samples), seed
    )
    transforms = build_transforms(local_cfg, is_train=False)
    loader = DataLoader(
        ImageDataset(samples, transforms),
        batch_size=int(local_cfg.TEST.IMS_PER_BATCH),
        shuffle=False,
        num_workers=int(local_cfg.DATALOADER.NUM_WORKERS),
        collate_fn=val_collate_fn,
        worker_init_fn=seed_worker,
        generator=make_data_loader_generator(seed, "validation"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_checkpoint_model(local_cfg, checkpoint, device)
    feature_rows = []
    pid_rows = []
    camid_rows = []
    with torch.no_grad():
        for images, pids, camids in loader:
            features = model(images.to(device))
            if str(local_cfg.TEST.FEAT_NORM).lower() == "yes":
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            feature_rows.append(features.cpu().numpy())
            pid_rows.extend(int(value) for value in pids)
            camid_rows.extend(int(value) for value in camids)
    statistics = compute_distance_statistics(
        np.concatenate(feature_rows, axis=0), pid_rows, camid_rows
    )
    payload = {
        "schema_version": 1,
        "analysis_type": "post_hoc_checkpoint_feature_distance",
        "analysis_time": utc_now(),
        "analysis_seed": seed,
        "dataset": str(local_cfg.DATASETS.NAMES),
        "sample_count": len(samples),
        "population_count": len(dataset.query) + len(dataset.gallery),
        "sampling_rule": (
            "all query+gallery samples when <= max_samples; otherwise a sorted "
            "uniform sample without replacement using analysis_seed"
        ),
        "max_samples": int(max_samples),
        "checkpoint": str(Path(checkpoint).resolve()).replace("\\", "/"),
        "source_checkpoint_sha256": sha256_file(checkpoint),
    }
    payload.update(statistics)
    atomic_write_json(output, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze Re-ID distance distributions")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-samples", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        analyze_checkpoint(
            args.config_file, args.checkpoint, args.output, args.seed,
            max_samples=args.max_samples,
        )
    except BaseException as error:
        print("Distance analysis failed: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
