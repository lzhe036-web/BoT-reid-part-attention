#!/usr/bin/env python
"""Generate bounded, deterministic selected-checkpoint gating evidence."""

from __future__ import absolute_import

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.collate_batch import val_collate_fn
from data.datasets import ImageDataset, init_dataset
from data.transforms import build_transforms
from modeling import build_model
from utils.dynamic_gating_evidence import (
    DYNAMIC_GATING_SAMPLE_FIELDS,
    DYNAMIC_GATING_SELECTION_RULE,
    GatingEpochAccumulator,
    dynamic_gating_sample_fields,
    gating_scales,
)
from utils.experiment_schema import SCHEMA_VERSION
from utils.experiment_recording import sha256_file
from utils.reproducibility import make_data_loader_generator, seed_worker


SELECTION_RULE = DYNAMIC_GATING_SELECTION_RULE
SAMPLE_FIELDS = DYNAMIC_GATING_SAMPLE_FIELDS


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}".format(path.name, uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Selected checkpoint has no model state_dict")
    result = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            raise ValueError("Selected checkpoint state_dict contains non-tensors")
        normalized = key[7:] if key.startswith("module.") else key
        result[normalized] = value
    return result


def _stable_key(split, image_path, pid, camid, data_root):
    try:
        relative = Path(image_path).resolve().relative_to(Path(data_root).resolve()).as_posix()
    except ValueError:
        relative = Path(image_path).name
    identity = "{}|{}|{}|{}".format(split, relative, int(pid), int(camid))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_samples(configuration, limit=256):
    dataset = init_dataset(
        configuration.DATASETS.NAMES, root=configuration.DATASETS.ROOT_DIR,
        verbose=False,
    )
    candidates = []
    for split, entries in (("query", dataset.query), ("gallery", dataset.gallery)):
        for image_path, pid, camid in entries:
            key = _stable_key(
                split, image_path, pid, camid, configuration.DATASETS.ROOT_DIR
            )
            candidates.append((key, split, image_path, int(pid), int(camid)))
    candidates.sort(
        key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest()
    )
    return candidates[:min(int(limit), len(candidates))], dataset.num_train_pids


def generate_dynamic_gating_evidence(configuration, checkpoint_path, output_dir,
                                     training_epoch_statistics, limit=256,
                                     device=None):
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    selected, num_classes = select_samples(configuration, limit=limit)
    if not selected:
        raise ValueError("Deterministic gating sample selection returned no samples")
    transform = build_transforms(configuration, is_train=False)
    image_entries = [(item[2], item[3], item[4]) for item in selected]
    loader = DataLoader(
        ImageDataset(image_entries, transform),
        batch_size=int(configuration.TEST.IMS_PER_BATCH), shuffle=False,
        num_workers=int(configuration.DATALOADER.NUM_WORKERS),
        collate_fn=val_collate_fn,
        worker_init_fn=seed_worker,
        generator=make_data_loader_generator(configuration.SEED, "query"),
    )
    model_cfg = configuration.clone()
    model_cfg.defrost()
    model_cfg.MODEL.PRETRAIN_CHOICE = "none"
    model_cfg.MODEL.PRETRAIN_PATH = ""
    model_cfg.freeze()
    model = build_model(model_cfg, int(num_classes))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(_state_dict(checkpoint), strict=True)
    actual_device = device or (
        "cuda" if configuration.MODEL.DEVICE == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    model.to(actual_device)
    model.eval()
    scales = gating_scales(configuration)
    expected_weight_sum = 1.0 if scales == (2, 4) else float(len(scales))
    accumulator = GatingEpochAccumulator(
        configuration.MODEL.MULTI_GRANULARITY_GATING_TAU,
        scales=scales, weight_sum=expected_weight_sum,
    )
    rows = []
    offset = 0
    with torch.no_grad():
        for images, _pids, _camids in loader:
            images = images.to(actual_device)
            descriptors = model(images)
            expected_descriptor_dim = 2048 + 256 * len(scales)
            if descriptors.dim() != 2 or descriptors.size(1) != expected_descriptor_dim:
                raise ValueError("Inference descriptor contract changed")
            evidence = model._last_dynamic_gating
            probabilities = evidence["probabilities"].to("cpu", dtype=torch.float64)
            weights = evidence["weights"].to("cpu", dtype=torch.float64)
            if tuple(evidence.get("scales", ())) != tuple(scales):
                raise ValueError("Dynamic Gating active-scale evidence mismatch")
            if not torch.allclose(
                    weights, probabilities * expected_weight_sum,
                    rtol=1e-6, atol=1e-9):
                raise ValueError(
                    "Dynamic Gating applied weights do not match probabilities"
                )
            accumulator.update(probabilities)
            for local_index in range(probabilities.size(0)):
                key, split, _path, pid, camid = selected[offset + local_index]
                probability = probabilities[local_index]
                weight = weights[local_index]
                safe = probability.clamp_min(torch.finfo(probability.dtype).tiny)
                entropy = float(-(probability * safe.log()).sum().item())
                dominant = scales[int(probability.argmax().item())]
                row = {
                    "stable_sample_key": key, "dataset_split": split,
                    "pid": pid, "camid": camid,
                }
                for index, scale in enumerate(scales):
                    row["p{}".format(scale)] = float(probability[index])
                    row["w{}".format(scale)] = float(weight[index])
                row.update({
                    "entropy": entropy, "dominant_k": dominant,
                    "checkpoint_sha256": checkpoint_sha,
                })
                rows.append(row)
            offset += int(probabilities.size(0))
    if offset != len(selected):
        raise ValueError("Gating evidence sample count mismatch")

    samples_path = output_dir / "gating_samples.tsv"
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=dynamic_gating_sample_fields(scales), delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(samples_path, buffer.getvalue())
    samples_evidence = {
        "path": str(samples_path), "size_bytes": samples_path.stat().st_size,
        "sha256": sha256_file(samples_path),
        "source_checkpoint_sha256": checkpoint_sha,
        "selection_rule": SELECTION_RULE,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_sha,
        "selection_rule": SELECTION_RULE,
        "selected_sample_count": len(rows),
        "training_epoch_statistics": dict(training_epoch_statistics),
        "deterministic_sample_statistics": accumulator.summary(),
        "gating_samples": samples_evidence,
    }
    summary_path = output_dir / "dynamic_gating_summary.json"
    _atomic_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary_path, samples_path, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epoch-statistics", required=True)
    args = parser.parse_args(argv)
    from config import cfg
    configuration = cfg.clone()
    configuration.merge_from_file(args.config)
    configuration.freeze()
    statistics = json.loads(Path(args.epoch_statistics).read_text(encoding="utf-8"))
    paths = generate_dynamic_gating_evidence(
        configuration, args.checkpoint, args.output_dir, statistics
    )
    print(json.dumps([str(paths[0]), str(paths[1])]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
