#!/usr/bin/env python
"""Strict, reproducible G1-versus-G2 Dynamic Gating analysis.

The analysis reads only formal G1/G2 evidence, validates the selected
checkpoints and protocol, freezes a Market1501 query/gallery candidate list
*before* model inference, then re-extracts both gates on exactly those images.
It never trains, changes a model, substitutes a smoke run, or hand-fills a
metric.  Training-final-epoch evidence contains only aggregate moments; fields
that require per-training-sample values remain ``not_recorded``.
"""

from __future__ import absolute_import

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tarfile
import uuid
from collections import Counter
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from data.collate_batch import val_collate_fn
from data.datasets import ImageDataset
from data.transforms import build_transforms
from modeling import build_model
from tools.analyze_dynamic_gating import _state_dict
from utils.dynamic_gating_evidence import read_gating_epoch_records
from utils.experiment_recording import sha256_file
from utils.reproducibility import make_data_loader_generator, seed_worker


SCALES = (2, 4, 6)
G1_LABEL = "G1-global"
G2_LABEL = "G2-global-local"
G1_COMMIT = "352127379e2c4bd7475fddb79fb8e3754cb8a2b8"
G2_COMMIT = "fa4e7f88f7ab645e9ba6b9a8e6cffdd9056b36c8"
G1_SHA256 = "e57dd34a1b8d10ef6f544d55d4f627656815704ee8303a1c20e3ae735d4ca6aa"
G2_SHA256 = "49a766fb520cca5dfe9121f272994185db9fddee45709c9d61446c6781dc7d45"
SELECTION_PREFIX = "g1-vs-g2-gating-visual-v1|"
NOT_RECORDED = "not_recorded"
NOT_APPLICABLE = "N/A"
CATEGORIES = ("clear", "occluded", "misaligned", "side_view", "back_view", "blurred")
COLORS = {2: "#1f77b4", 4: "#ff7f0e", 6: "#2ca02c"}


class EvidenceError(RuntimeError):
    """Raised whenever a formal-evidence requirement is not satisfied."""


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    gate_input: str
    formal_commit: str
    checkpoint_sha256: str
    run_dir: Path
    output_dir: Path
    checkpoint: Path


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}".format(path.name, uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write_json(path, value):
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_csv(path, fields, rows, delimiter=","):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=delimiter, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, NOT_APPLICABLE) for field in fields})
    _atomic_text(path, buffer.getvalue())


def _markdown_value(value):
    return str(value if value not in (None, "") else NOT_APPLICABLE).replace("|", "\\|").replace("\n", " ")


def _write_markdown_from_csv(path, title, fields, rows, note):
    lines = ["# {}".format(title), "", note, "", "| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_value(row.get(field)) for field in fields) + " |")
    lines.append("")
    _atomic_text(path, "\n".join(lines))


def _file_evidence(path):
    path = Path(path)
    if not path.is_file():
        raise EvidenceError("Required file is absent: {}".format(path))
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EvidenceError("Invalid {} JSON: {}".format(label, path)) from error


def _nested(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return NOT_RECORDED
        value = value[key]
    return value


def _manifest_field(manifest, name):
    if name in manifest:
        return manifest[name]
    metrics = manifest.get("metrics", {})
    return metrics.get(name, NOT_RECORDED) if isinstance(metrics, dict) else NOT_RECORDED


def _exact_list(value, expected, label):
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if list(value) != list(expected):
        raise EvidenceError("{} mismatch: {!r} != {!r}".format(label, value, expected))


def _close(actual, expected, label, rel=1e-6, abs_tol=1e-9):
    try:
        valid = math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=abs_tol)
    except (TypeError, ValueError, OverflowError) as error:
        raise EvidenceError("{} is not numeric".format(label)) from error
    if not valid:
        raise EvidenceError("{} mismatch: {!r} != {!r}".format(label, actual, expected))


def _config_from_path(path):
    configuration = cfg.clone()
    try:
        configuration.merge_from_file(str(path))
    except Exception as error:
        raise EvidenceError("Cannot load recorded config {}".format(path)) from error
    configuration.freeze()
    return configuration


def _config_protocol(configuration):
    return {
        "dataset": str(configuration.DATASETS.NAMES).lower(),
        "dataset_root": str(configuration.DATASETS.ROOT_DIR),
        "seed": int(configuration.SEED),
        "backbone": str(configuration.MODEL.NAME),
        "gating_input": str(configuration.MODEL.MULTI_GRANULARITY_GATING_INPUT).lower(),
        "temperature": float(configuration.MODEL.MULTI_GRANULARITY_GATING_TAU),
        "normalization": str(configuration.MODEL.MULTI_GRANULARITY_GATING_NORMALIZATION).lower(),
        "scale_order": [int(value) for value in configuration.MODEL.MULTI_GRANULARITY_PART_SCALES],
        "input_test": [int(value) for value in configuration.INPUT.SIZE_TEST],
        "neck_feature": str(configuration.TEST.NECK_FEAT),
        "feature_normalization": str(configuration.TEST.FEAT_NORM),
        "reranking": str(configuration.TEST.RE_RANKING),
    }


def _manifest_artifact(manifest, key, run_dir):
    candidates = []
    if isinstance(manifest.get("artifacts"), dict):
        candidates.append(manifest["artifacts"].get(key))
    candidates.append(manifest.get(key))
    for value in candidates:
        if isinstance(value, dict):
            path = value.get("path")
            digest = value.get("sha256")
        else:
            path, digest = None, None
        if path:
            local = run_dir / Path(path).name
            if local.is_file():
                return local, digest
            if Path(path).is_file():
                return Path(path), digest
    fallback = {
        "source_config": "config_source.yml",
        "resolved_config": "config_resolved.yml",
        "dynamic_gating_summary": "dynamic_gating_summary.json",
        "gating_samples": "gating_samples.tsv",
        "gating_epoch_statistics": "dynamic_gating_epoch_stats.jsonl",
        "training_log": "log.txt",
    }.get(key)
    if fallback and (run_dir / fallback).is_file():
        return run_dir / fallback, None
    raise EvidenceError("Manifest has no accessible {} artifact".format(key))


def _validate_manifest_artifact(path, recorded_sha, label):
    evidence = _file_evidence(path)
    if recorded_sha not in (None, "", NOT_RECORDED, NOT_APPLICABLE) and evidence["sha256"] != recorded_sha:
        raise EvidenceError("{} SHA256 mismatch".format(label))
    return evidence


def _validate_historical_gating_samples(path, checkpoint_sha256, label):
    """Audit the original archived TSV without using it for fixed selection."""
    expected = ("stable_sample_key", "dataset_split", "pid", "camid", "p2", "p4", "p6", "w2", "w4", "w6", "entropy", "dominant_k", "checkpoint_sha256")
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or tuple(rows[0].keys()) != expected:
        raise EvidenceError("{} historical gating TSV schema/order is invalid".format(label))
    for row in rows:
        p = [float(row["p{}".format(scale)]) for scale in SCALES]
        w = [float(row["w{}".format(scale)]) for scale in SCALES]
        if row["checkpoint_sha256"] != checkpoint_sha256:
            raise EvidenceError("{} historical gating TSV checkpoint mismatch".format(label))
        if not math.isclose(sum(p), 1.0, rel_tol=1e-6, abs_tol=1e-8) or not math.isclose(sum(w), 3.0, rel_tol=1e-6, abs_tol=1e-8):
            raise EvidenceError("{} historical gating TSV p/w sums are invalid".format(label))
        if int(row["dominant_k"]) != SCALES[max(range(3), key=lambda item: w[item])]:
            raise EvidenceError("{} historical gating TSV dominant-K is invalid".format(label))
    return len(rows)


def validate_formal_run(spec):
    """Validate all immutable formal-run and selected-checkpoint contracts."""
    run_dir = Path(spec.run_dir)
    if not Path(spec.output_dir).is_dir():
        raise EvidenceError("{} formal output directory is absent".format(spec.label))
    if not Path(spec.checkpoint).is_file():
        raise EvidenceError("{} selected checkpoint is absent".format(spec.label))
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path, spec.label + " run manifest")
    if manifest.get("status") != "success" or manifest.get("run_kind") != "formal":
        raise EvidenceError("{} run is not a successful formal run".format(spec.label))
    if str(manifest.get("dataset", "")).lower() != "market1501":
        raise EvidenceError("{} run is not Market1501".format(spec.label))
    if manifest.get("commit") != spec.formal_commit:
        raise EvidenceError("{} formal training commit mismatch".format(spec.label))
    if str(manifest.get("output_dir", "")) != str(spec.output_dir):
        raise EvidenceError("{} manifest output_dir does not match the supplied formal output".format(spec.label))
    if int(manifest.get("seed", -1)) != 42:
        raise EvidenceError("{} seed is not 42".format(spec.label))
    _close(manifest.get("gating_temperature"), 1.0, spec.label + " temperature")
    if manifest.get("gating_normalization") != "scaled_softmax":
        raise EvidenceError("{} uses a non-scaled-softmax gate".format(spec.label))
    _exact_list(manifest.get("scale_order"), [2, 4, 6], spec.label + " scale order")
    _exact_list(manifest.get("gate_outputs"), ["w2", "w4", "w6"], spec.label + " gate output order")
    if manifest.get("gating_input") != spec.gate_input:
        raise EvidenceError("{} gate input mismatch".format(spec.label))
    if int(_manifest_field(manifest, "selected_epoch")) != 120:
        raise EvidenceError("{} selected epoch is not 120".format(spec.label))

    selected = manifest.get("selected_checkpoint")
    if not isinstance(selected, dict):
        selected = _nested(manifest, "artifacts", "selected_checkpoint")
    selected_sha = selected.get("sha256") if isinstance(selected, dict) else NOT_RECORDED
    if selected_sha != spec.checkpoint_sha256:
        raise EvidenceError("{} selected checkpoint SHA256 in manifest mismatches fixed identity".format(spec.label))
    if sha256_file(spec.checkpoint) != spec.checkpoint_sha256:
        raise EvidenceError("{} supplied checkpoint SHA256 mismatch".format(spec.label))

    artifacts, paths = {"run_manifest": _file_evidence(manifest_path)}, {}
    for key in ("source_config", "resolved_config", "dynamic_gating_summary", "gating_samples", "gating_epoch_statistics", "training_log"):
        path, recorded_sha = _manifest_artifact(manifest, key, run_dir)
        paths[key] = path
        artifacts[key] = _validate_manifest_artifact(path, recorded_sha, spec.label + " " + key)
    artifacts["checkpoint"] = _file_evidence(spec.checkpoint)
    source_config, resolved_config = _config_from_path(paths["source_config"]), _config_from_path(paths["resolved_config"])
    source_protocol, resolved_protocol = _config_protocol(source_config), _config_protocol(resolved_config)
    for protocol_name, protocol in (("source", source_protocol), ("resolved", resolved_protocol)):
        if protocol["dataset"] != "market1501" or protocol["seed"] != 42:
            raise EvidenceError("{} {} config dataset/seed mismatch".format(spec.label, protocol_name))
        if protocol["gating_input"] != spec.gate_input or protocol["normalization"] != "scaled_softmax":
            raise EvidenceError("{} {} gate-config mismatch".format(spec.label, protocol_name))
        _close(protocol["temperature"], 1.0, spec.label + " " + protocol_name + " temperature")
        if protocol["scale_order"] != [2, 4, 6]:
            raise EvidenceError("{} {} scale order mismatch".format(spec.label, protocol_name))
    summary = _read_json(paths["dynamic_gating_summary"], spec.label + " dynamic gating summary")
    if summary.get("source_checkpoint_sha256") != spec.checkpoint_sha256:
        raise EvidenceError("{} summary checkpoint SHA256 mismatch".format(spec.label))
    sample_evidence = summary.get("gating_samples", {})
    if not isinstance(sample_evidence, dict) or sample_evidence.get("sha256") != artifacts["gating_samples"]["sha256"]:
        raise EvidenceError("{} gating-samples summary SHA256 mismatch".format(spec.label))
    if sample_evidence.get("source_checkpoint_sha256") != spec.checkpoint_sha256:
        raise EvidenceError("{} sample TSV checkpoint SHA256 mismatch".format(spec.label))
    historical_sample_count = _validate_historical_gating_samples(
        paths["gating_samples"], spec.checkpoint_sha256, spec.label
    )
    history = read_gating_epoch_records(paths["gating_epoch_statistics"])
    if not history or int(history[-1].get("epoch", -1)) != 120:
        raise EvidenceError("{} final dynamic-gating epoch evidence is absent or not epoch 120".format(spec.label))
    required_epoch_fields = ("gating_sample_count", "mean_gate_entropy", "p2_mean", "p2_std", "p2_min", "p2_max", "p4_mean", "p4_std", "p4_min", "p4_max", "p6_mean", "p6_std", "p6_min", "p6_max", "applied_w2_mean", "applied_w2_std", "applied_w4_mean", "applied_w4_std", "applied_w6_mean", "applied_w6_std", "dominant_k2_ratio", "dominant_k4_ratio", "dominant_k6_ratio")
    if any(field not in history[-1] for field in required_epoch_fields):
        raise EvidenceError("{} final epoch gating statistics are incomplete".format(spec.label))
    return {
        "spec": spec, "manifest": manifest, "paths": paths, "artifacts": artifacts,
        "source_config": source_config, "resolved_config": resolved_config,
        "source_protocol": source_protocol, "resolved_protocol": resolved_protocol,
        "summary": summary, "history": history, "historical_sample_count": historical_sample_count,
    }


def validate_pair(g1, g2):
    for key in ("dataset", "dataset_root", "seed", "backbone", "temperature", "normalization", "scale_order", "input_test", "neck_feature", "feature_normalization", "reranking"):
        if g1["resolved_protocol"][key] != g2["resolved_protocol"][key]:
            raise EvidenceError("G1/G2 query-gallery protocol mismatch for {}".format(key))


def _market_directories(dataset_root):
    root = Path(dataset_root)
    candidates = (root, root / "market1501", root / "Market-1501-v15.09")
    for market in candidates:
        query, gallery = market / "query", market / "bounding_box_test"
        if query.is_dir() and gallery.is_dir():
            return market, query, gallery
    raise EvidenceError("Market1501 query/bounding_box_test directories are absent; __MACOSX is never searched")


def build_fixed_candidates(dataset_root, output_path, query_limit=256, gallery_limit=256):
    """Freeze query/gallery candidates without accepting or reading gate values."""
    market, query_dir, gallery_dir = _market_directories(dataset_root)
    expression = __import__("re").compile(r"^([\-\d]+)_c(\d+)")
    selected = []
    for split, directory, limit in (("query", query_dir, query_limit), ("gallery", gallery_dir, gallery_limit)):
        rows = []
        for image in sorted(directory.glob("*.jpg")):
            matched = expression.match(image.name)
            if not matched:
                continue
            pid, camid = int(matched.group(1)), int(matched.group(2)) - 1
            if pid == -1:
                continue
            relative = image.relative_to(market.parent).as_posix()
            stable = "{}|{}|{}|{}".format(split, relative, pid, camid)
            rows.append({
                "stable_sample_key": stable, "split": split, "relative_path": relative,
                "pid": pid, "camid": camid,
                "selection_hash": hashlib.sha256((SELECTION_PREFIX + stable).encode("utf-8")).hexdigest(),
                "image_sha256": sha256_file(image),
            })
        rows.sort(key=lambda row: row["selection_hash"])
        if not rows:
            raise EvidenceError("No valid {} images were found".format(split))
        for rank, row in enumerate(rows[:int(limit)], 1):
            row["selection_rank"] = rank
            selected.append(row)
    fields = ("stable_sample_key", "split", "relative_path", "pid", "camid", "selection_hash", "selection_rank", "image_sha256")
    _write_csv(output_path, fields, selected, delimiter="\t")
    return selected, market


def read_fixed_candidates(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    fields = ("stable_sample_key", "split", "relative_path", "pid", "camid", "selection_hash", "selection_rank", "image_sha256")
    if not rows or tuple(rows[0].keys()) != fields:
        raise EvidenceError("Fixed candidate manifest schema is invalid")
    for split in ("query", "gallery"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            raise EvidenceError("Fixed candidate manifest lacks {} samples".format(split))
        if [row["selection_hash"] for row in subset] != sorted(row["selection_hash"] for row in subset):
            raise EvidenceError("Fixed candidate manifest is not hash-sorted within {}".format(split))
    if len({row["stable_sample_key"] for row in rows}) != len(rows):
        raise EvidenceError("Fixed candidate manifest has duplicate stable keys")
    return rows


def _resolve_image(market_root, row):
    market_root = Path(market_root)
    relative = Path(row["relative_path"])
    candidates = (market_root.parent / relative, market_root / relative, market_root / Path(*relative.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise EvidenceError("Fixed candidate image is absent: {}".format(row["relative_path"]))


def _model_for_evidence(configuration, num_classes, checkpoint, device):
    model_cfg = configuration.clone()
    model_cfg.defrost()
    model_cfg.MODEL.PRETRAIN_CHOICE = "none"
    model_cfg.MODEL.PRETRAIN_PATH = ""
    model_cfg.freeze()
    model = build_model(model_cfg, int(num_classes))
    model.load_state_dict(_state_dict(torch.load(str(checkpoint), map_location="cpu")), strict=True)
    model.to(device)
    model.eval()
    return model


def extract_fixed_gates(validated, candidates, market_root, output_path, device=None):
    """Infer G1/G2 only on frozen candidates and export p/w in explicit order."""
    configuration = validated["resolved_config"]
    actual_device = device or ("cuda" if configuration.MODEL.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    paths = [_resolve_image(market_root, row) for row in candidates]
    entries = [(str(path), int(row["pid"]), int(row["camid"])) for path, row in zip(paths, candidates)]
    loader = DataLoader(
        ImageDataset(entries, build_transforms(configuration, is_train=False)),
        batch_size=int(configuration.TEST.IMS_PER_BATCH), shuffle=False,
        num_workers=int(configuration.DATALOADER.NUM_WORKERS), collate_fn=val_collate_fn,
        worker_init_fn=seed_worker, generator=make_data_loader_generator(configuration.SEED, "query"),
    )
    # Existing run summaries do not necessarily retain num_train_pids.  The
    # selected checkpoint classifier strictly determines it, and build_model's
    # classifier is replaced by the state_dict shape during a strict load only
    # when this value agrees.  Read it directly rather than infer from data.
    checkpoint_state = _state_dict(torch.load(str(validated["spec"].checkpoint), map_location="cpu"))
    classifier = checkpoint_state.get("classifier.weight")
    if classifier is None or classifier.dim() != 2:
        raise EvidenceError("Selected checkpoint has no classifier.weight to establish class count")
    num_classes = int(classifier.size(0))
    model = _model_for_evidence(configuration, num_classes, validated["spec"].checkpoint, actual_device)
    rows, offset = [], 0
    with torch.no_grad():
        for images, _pids, _camids in loader:
            model(images.to(actual_device))
            evidence = model._last_dynamic_gating
            if not isinstance(evidence, dict):
                raise EvidenceError("{} model did not export Dynamic Gating values".format(validated["spec"].label))
            probabilities = evidence["probabilities"].detach().to(dtype=torch.float64, device="cpu")
            weights = evidence["weights"].detach().to(dtype=torch.float64, device="cpu")
            if probabilities.dim() != 2 or tuple(probabilities.shape[1:]) != (3,) or tuple(weights.shape) != tuple(probabilities.shape):
                raise EvidenceError("{} gating tensor shape is not [B,3]".format(validated["spec"].label))
            for index in range(probabilities.size(0)):
                p = [float(probabilities[index, item]) for item in range(3)]
                w = [float(weights[index, item]) for item in range(3)]
                if not all(math.isfinite(value) and value >= 0.0 for value in p + w):
                    raise EvidenceError("{} inferred non-finite/negative gate".format(validated["spec"].label))
                if not math.isclose(sum(p), 1.0, rel_tol=1e-6, abs_tol=1e-8):
                    raise EvidenceError("{} probabilities do not sum to one".format(validated["spec"].label))
                if not math.isclose(sum(w), 3.0, rel_tol=1e-6, abs_tol=1e-8):
                    raise EvidenceError("{} applied weights do not sum to three".format(validated["spec"].label))
                if any(not math.isclose(w[item], 3.0 * p[item], rel_tol=1e-6, abs_tol=1e-8) for item in range(3)):
                    raise EvidenceError("{} applied weights are not scaled-softmax weights".format(validated["spec"].label))
                candidate = candidates[offset + index]
                dominant_index = max(range(3), key=lambda item: w[item])
                rows.append({
                    "stable_sample_key": candidate["stable_sample_key"], "split": candidate["split"],
                    "relative_path": candidate["relative_path"], "pid": candidate["pid"], "camid": candidate["camid"],
                    "selection_hash": candidate["selection_hash"], "image_sha256": candidate["image_sha256"],
                    "p2": p[0], "p4": p[1], "p6": p[2], "w2": w[0], "w4": w[1], "w6": w[2],
                    "dominant_k": SCALES[dominant_index], "checkpoint_sha256": validated["spec"].checkpoint_sha256,
                })
            offset += probabilities.size(0)
    if offset != len(candidates):
        raise EvidenceError("Fixed gate extraction sample count mismatch")
    fields = ("stable_sample_key", "split", "relative_path", "pid", "camid", "selection_hash", "image_sha256", "p2", "p4", "p6", "w2", "w4", "w6", "dominant_k", "checkpoint_sha256")
    _write_csv(output_path, fields, rows, delimiter="\t")
    return rows


def read_fixed_gates(path, expected_sha):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    fields = ("stable_sample_key", "split", "relative_path", "pid", "camid", "selection_hash", "image_sha256", "p2", "p4", "p6", "w2", "w4", "w6", "dominant_k", "checkpoint_sha256")
    if not rows or tuple(rows[0].keys()) != fields:
        raise EvidenceError("Fixed-gating TSV schema is invalid")
    result = {}
    for row in rows:
        key = row["stable_sample_key"]
        if key in result or row["checkpoint_sha256"] != expected_sha:
            raise EvidenceError("Fixed-gating TSV has duplicate key or checkpoint mismatch")
        p, w = [float(row["p{}".format(scale)]) for scale in SCALES], [float(row["w{}".format(scale)]) for scale in SCALES]
        if not math.isclose(sum(p), 1.0, rel_tol=1e-6, abs_tol=1e-8) or not math.isclose(sum(w), 3.0, rel_tol=1e-6, abs_tol=1e-8):
            raise EvidenceError("Fixed-gating row violates p/w sum semantics")
        if int(row["dominant_k"]) != SCALES[max(range(3), key=lambda item: w[item])]:
            raise EvidenceError("Fixed-gating row has invalid dominant K")
        result[key] = dict(row, **{"p": p, "w": w, "dominant": int(row["dominant_k"])})
    return result


def _quantile(values, fraction):
    values = sorted(values)
    location = (len(values) - 1) * fraction
    left, right = int(math.floor(location)), int(math.ceil(location))
    return values[left] if left == right else values[left] + (values[right] - values[left]) * (location - left)


def _bootstrap_mean(values, seed=42, replicates=1000):
    if not values:
        return NOT_RECORDED, NOT_RECORDED
    generator = np.random.RandomState(int(seed))
    values = np.asarray(values, dtype=np.float64)
    means = np.mean(values[generator.randint(0, len(values), size=(int(replicates), len(values)))], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _describe(values, seed=42, replicates=1000):
    if not values:
        return {field: NOT_RECORDED for field in ("count", "mean", "std", "min", "max", "median", "q25", "q75", "ci95_low", "ci95_high")}
    values = [float(value) for value in values]
    low, high = _bootstrap_mean(values, seed, replicates)
    return {
        "count": len(values), "mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)), "max": float(np.max(values)), "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75)),
        "ci95_low": low, "ci95_high": high,
    }


WEIGHT_FIELDS = ("scope", "model", "quantity", "component", "count", "mean", "std", "min", "max", "median", "q25", "q75", "dominant_count", "dominant_ratio", "ci95_low", "ci95_high", "status")
COLLAPSE_FIELDS = ("scope", "comparison", "model", "metric", "count", "mean", "std", "min", "max", "median", "q25", "q75", "ci95_low", "ci95_high", "paired_intersection", "g1_only", "g2_only", "complete_pairing", "status")


def _sample_weight_rows(label, sample_map, scope, seed, replicates):
    rows = []
    for quantity, key in (("probability", "p"), ("applied_weight", "w")):
        for index, scale in enumerate(SCALES):
            values = [sample[key][index] for sample in sample_map.values()]
            row = {"scope": scope, "model": label, "quantity": quantity, "component": "{}{}".format("p" if quantity == "probability" else "w", scale), "status": "measured"}
            row.update(_describe(values, seed, replicates))
            dominant = sum(1 for sample in sample_map.values() if sample["dominant"] == scale)
            row["dominant_count"], row["dominant_ratio"] = dominant, dominant / float(len(sample_map))
            rows.append(row)
    return rows


def _training_weight_rows(label, history, scope):
    final = history[-1]
    count = int(final["gating_sample_count"])
    rows = []
    for quantity, prefix in (("probability", "p"), ("applied_weight", "applied_w")):
        for scale in SCALES:
            if quantity == "probability":
                mean, std, minimum, maximum = final["p{}_mean".format(scale)], final["p{}_std".format(scale)], final["p{}_min".format(scale)], final["p{}_max".format(scale)]
            else:
                mean, std = final["applied_w{}_mean".format(scale)], final["applied_w{}_std".format(scale)]
                minimum, maximum = 3.0 * float(final["p{}_min".format(scale)]), 3.0 * float(final["p{}_max".format(scale)])
            ratio = float(final["dominant_k{}_ratio".format(scale)])
            dominant_count = int(round(ratio * count))
            rows.append({
                "scope": scope, "model": label, "quantity": quantity,
                "component": "{}{}".format("p" if quantity == "probability" else "w", scale),
                "count": count, "mean": mean, "std": std, "min": minimum, "max": maximum,
                "median": NOT_RECORDED, "q25": NOT_RECORDED, "q75": NOT_RECORDED,
                "dominant_count": dominant_count, "dominant_ratio": ratio,
                "ci95_low": NOT_RECORDED, "ci95_high": NOT_RECORDED,
                "status": "aggregate_epoch_evidence_no_raw_quantiles",
            })
    return rows


def _collapse_per_sample(sample_map):
    rows = {}
    for key, sample in sample_map.items():
        p = np.asarray(sample["p"], dtype=np.float64)
        entropy = float(-(p * np.log(np.maximum(p, np.finfo(np.float64).tiny))).sum())
        ordered = sorted(p, reverse=True)
        # Equivalent concentration metric: Gini coefficient for three
        # non-negative gate probabilities, bounded in [0, 2/3].
        gini = float(sum(abs(float(left) - float(right)) for left in p for right in p) / (2.0 * len(p) * float(p.sum())))
        rows[key] = {
            "entropy": entropy, "normalized_entropy": entropy / math.log(3.0),
            "maximum_probability": float(ordered[0]), "first_second_margin": float(ordered[0] - ordered[1]),
            "effective_active_scales": float(math.exp(entropy)), "normalized_effective_scales": float(math.exp(entropy) / 3.0),
            "uniform_total_variation": float(0.5 * np.abs(p - 1.0 / 3.0).sum()),
            "gini_concentration": gini, "dominant": sample["dominant"],
        }
    return rows


def _collapse_rows(g1_map, g2_map, seed, replicates):
    intersection = sorted(set(g1_map) & set(g2_map))
    g1_only, g2_only = len(set(g1_map) - set(g2_map)), len(set(g2_map) - set(g1_map))
    complete = len(intersection) == len(g1_map) == len(g2_map)
    if not intersection:
        raise EvidenceError("G1/G2 fixed samples have no stable-key intersection")
    metrics1, metrics2 = _collapse_per_sample({key: g1_map[key] for key in intersection}), _collapse_per_sample({key: g2_map[key] for key in intersection})
    rows = []
    common = {"scope": "fixed_query_gallery_paired", "comparison": "G1_vs_G2", "paired_intersection": len(intersection), "g1_only": g1_only, "g2_only": g2_only, "complete_pairing": complete, "status": "measured"}
    for model, values in ((G1_LABEL, metrics1), (G2_LABEL, metrics2)):
        for metric in ("entropy", "normalized_entropy", "maximum_probability", "first_second_margin", "effective_active_scales", "normalized_effective_scales", "uniform_total_variation", "gini_concentration"):
            row = dict(common, model=model, metric=metric)
            row.update(_describe([value[metric] for value in values.values()], seed, replicates))
            rows.append(row)
        counts = Counter(value["dominant"] for value in values.values())
        for scale in SCALES:
            row = dict(common, model=model, metric="dominant_k{}_ratio".format(scale))
            ratio = counts[scale] / float(len(values))
            row.update({"count": len(values), "mean": ratio, "std": NOT_APPLICABLE, "min": NOT_APPLICABLE, "max": NOT_APPLICABLE, "median": NOT_APPLICABLE, "q25": NOT_APPLICABLE, "q75": NOT_APPLICABLE, "ci95_low": NOT_APPLICABLE, "ci95_high": NOT_APPLICABLE})
            rows.append(row)
    # Paired differences are stored as their own rows so every comparison is
    # explicitly G2 minus G1, never a chain through another experiment.
    for metric in ("entropy", "normalized_entropy", "maximum_probability", "first_second_margin", "effective_active_scales", "normalized_effective_scales", "uniform_total_variation", "gini_concentration"):
        row = dict(common, model="G2_minus_G1", metric=metric)
        row.update(_describe([metrics2[key][metric] - metrics1[key][metric] for key in intersection], seed, replicates))
        rows.append(row)
    for scale in SCALES:
        ratio1 = sum(1 for key in intersection if metrics1[key]["dominant"] == scale) / float(len(intersection))
        ratio2 = sum(1 for key in intersection if metrics2[key]["dominant"] == scale) / float(len(intersection))
        rows.append(dict(common, model="G2_minus_G1", metric="dominant_k{}_ratio".format(scale), count=len(intersection), mean=ratio2-ratio1, std=NOT_APPLICABLE, min=NOT_APPLICABLE, max=NOT_APPLICABLE, median=NOT_APPLICABLE, q25=NOT_APPLICABLE, q75=NOT_APPLICABLE, ci95_low=NOT_APPLICABLE, ci95_high=NOT_APPLICABLE))
    return rows, metrics1, metrics2, intersection, {"g1_count": len(g1_map), "g2_count": len(g2_map), "intersection_count": len(intersection), "g1_only": g1_only, "g2_only": g2_only, "complete_pairing": complete}


def _training_collapse_rows(label, history):
    final, count = history[-1], int(history[-1]["gating_sample_count"])
    rows = []
    mean_entropy = float(final["mean_gate_entropy"])
    values = {
        "entropy": mean_entropy, "normalized_entropy": mean_entropy / math.log(3.0),
        "effective_active_scales": math.exp(mean_entropy), "normalized_effective_scales": math.exp(mean_entropy) / 3.0,
    }
    for metric, value in values.items():
        rows.append({"scope": "training_final_epoch_all_training_samples", "comparison": "separate_formal_runs", "model": label, "metric": metric, "count": count, "mean": value, "std": NOT_RECORDED, "min": NOT_RECORDED, "max": NOT_RECORDED, "median": NOT_RECORDED, "q25": NOT_RECORDED, "q75": NOT_RECORDED, "ci95_low": NOT_RECORDED, "ci95_high": NOT_RECORDED, "paired_intersection": NOT_APPLICABLE, "g1_only": NOT_APPLICABLE, "g2_only": NOT_APPLICABLE, "complete_pairing": NOT_APPLICABLE, "status": "aggregate_epoch_evidence"})
    for metric in ("maximum_probability", "first_second_margin", "uniform_total_variation", "gini_concentration"):
        rows.append({"scope": "training_final_epoch_all_training_samples", "comparison": "separate_formal_runs", "model": label, "metric": metric, "count": count, "mean": NOT_RECORDED, "std": NOT_RECORDED, "min": NOT_RECORDED, "max": NOT_RECORDED, "median": NOT_RECORDED, "q25": NOT_RECORDED, "q75": NOT_RECORDED, "ci95_low": NOT_RECORDED, "ci95_high": NOT_RECORDED, "paired_intersection": NOT_APPLICABLE, "g1_only": NOT_APPLICABLE, "g2_only": NOT_APPLICABLE, "complete_pairing": NOT_APPLICABLE, "status": "not_recorded_in_aggregate_epoch_evidence"})
    for scale in SCALES:
        rows.append({"scope": "training_final_epoch_all_training_samples", "comparison": "separate_formal_runs", "model": label, "metric": "dominant_k{}_ratio".format(scale), "count": count, "mean": float(final["dominant_k{}_ratio".format(scale)]), "std": NOT_APPLICABLE, "min": NOT_APPLICABLE, "max": NOT_APPLICABLE, "median": NOT_APPLICABLE, "q25": NOT_APPLICABLE, "q75": NOT_APPLICABLE, "ci95_low": NOT_RECORDED, "ci95_high": NOT_RECORDED, "paired_intersection": NOT_APPLICABLE, "g1_only": NOT_APPLICABLE, "g2_only": NOT_APPLICABLE, "complete_pairing": NOT_APPLICABLE, "status": "aggregate_epoch_evidence"})
    return rows


def _save(figure, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    png, pdf = directory / (stem + ".png"), directory / (stem + ".pdf")
    figure.savefig(str(png), dpi=300, bbox_inches="tight")
    figure.savefig(str(pdf), bbox_inches="tight")
    plt.close(figure)
    return [str(png), str(pdf)]


def _plot_distributions(g1, g2, metrics1, metrics2, intersection, directory):
    outputs = []
    # Applied-weight histograms, shared native [0,3] range and bins.
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 8.5), sharex=True)
    for index, scale in enumerate(SCALES):
        values1, values2 = [g1[key]["w"][index] for key in intersection], [g2[key]["w"][index] for key in intersection]
        axes[index].hist(values1, bins=30, range=(0.0, 3.0), density=True, histtype="step", linewidth=1.7, color=COLORS[scale], label="G1 w{}".format(scale))
        axes[index].hist(values2, bins=30, range=(0.0, 3.0), density=True, histtype="step", linewidth=1.7, linestyle="--", color=COLORS[scale], label="G2 w{}".format(scale))
        axes[index].set_ylabel("density")
        axes[index].legend(); axes[index].grid(alpha=0.2)
    axes[-1].set_xlabel("native applied weight (scaled-softmax; sum=3)")
    figure.suptitle("G1/G2 applied gate-weight distributions on paired fixed samples")
    outputs += _save(figure, directory, "applied_weight_histograms")

    figure, axis = plt.subplots(figsize=(9.6, 4.8))
    values, labels, colors = [], [], []
    for index, scale in enumerate(SCALES):
        values += [[g1[key]["w"][index] for key in intersection], [g2[key]["w"][index] for key in intersection]]
        labels += ["G1 w{}".format(scale), "G2 w{}".format(scale)]
        colors += [COLORS[scale], COLORS[scale]]
    positions = np.arange(1, len(values) + 1)
    violins = axis.violinplot(values, positions=positions, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violins["bodies"], colors): body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.28)
    boxes = axis.boxplot(values, positions=positions, widths=0.45, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors): patch.set_facecolor(color); patch.set_alpha(0.55)
    axis.set_ylim(0.0, 3.0); axis.set_ylabel("native applied weight"); axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", alpha=0.2)
    axis.set_title("G1/G2 applied weight violin and box summary")
    outputs += _save(figure, directory, "applied_weight_boxplots")

    counts1, counts2 = Counter(g1[key]["dominant"] for key in intersection), Counter(g2[key]["dominant"] for key in intersection)
    figure, axis = plt.subplots(figsize=(6.8, 4.5)); positions = np.arange(3); width = 0.35
    axis.bar(positions - width/2, [counts1[scale]/len(intersection) for scale in SCALES], width, label="G1", color="#4c78a8")
    axis.bar(positions + width/2, [counts2[scale]/len(intersection) for scale in SCALES], width, label="G2", color="#f58518")
    axis.set_xticks(positions); axis.set_xticklabels(["K{}".format(scale) for scale in SCALES]); axis.set_ylim(0.0, 1.0); axis.set_ylabel("dominant-scale ratio"); axis.legend(); axis.grid(axis="y", alpha=0.2); axis.set_title("Dominant K on paired fixed samples")
    outputs += _save(figure, directory, "dominant_scale_ratios")

    metric_defs = (("entropy", "Gate entropy", (0.0, math.log(3.0))), ("maximum_probability", "Maximum probability", (0.0, 1.0)), ("first_second_margin", "First-second probability margin", (0.0, 1.0)))
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for axis, (metric, title, bounds) in zip(axes, metric_defs):
        boxes = axis.boxplot([[metrics1[key][metric] for key in intersection], [metrics2[key][metric] for key in intersection]], labels=["G1", "G2"], patch_artist=True, showfliers=False)
        boxes["boxes"][0].set_facecolor("#4c78a8"); boxes["boxes"][1].set_facecolor("#f58518")
        axis.set_ylim(*bounds); axis.set_title(title); axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Gate-concentration distributions on paired fixed samples")
    outputs += _save(figure, directory, "entropy_maximum_margin")

    figure, axis = plt.subplots(figsize=(8.6, 4.6))
    for index, scale in enumerate(SCALES):
        delta = [g2[key]["w"][index] - g1[key]["w"][index] for key in intersection]
        axis.scatter(range(len(intersection)), delta, s=8, alpha=0.65, color=COLORS[scale], label="Δw{} (G2−G1)".format(scale))
    axis.axhline(0.0, color="black", linewidth=1.0); axis.set_xlabel("fixed hash-sorted paired sample ordinal"); axis.set_ylabel("native applied-weight delta"); axis.legend(); axis.grid(alpha=0.2); axis.set_title("Per-sample G2−G1 applied-weight changes")
    outputs += _save(figure, directory, "paired_applied_weight_deltas")

    matrix = np.zeros((3, 3), dtype=int)
    for key in intersection: matrix[SCALES.index(g1[key]["dominant"]), SCALES.index(g2[key]["dominant"])] += 1
    figure, axis = plt.subplots(figsize=(5.5, 4.6)); image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    for i in range(3):
        for j in range(3): axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    axis.set_xticks(range(3)); axis.set_yticks(range(3)); axis.set_xticklabels(["G2 K{}".format(scale) for scale in SCALES]); axis.set_yticklabels(["G1 K{}".format(scale) for scale in SCALES]); axis.set_title("G1→G2 dominant-K transition"); figure.colorbar(image, ax=axis, label="sample count")
    outputs += _save(figure, directory, "dominant_k_transition")

    # Barycentric (ternary-equivalent) probability scatter without an external
    # ternary package.  The triangle uses p2/p4/p6 with a common geometry.
    figure, axis = plt.subplots(figsize=(6.2, 5.6)); vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3)/2]])
    for label, sample_map, color, marker in (("G1", g1, "#4c78a8", "o"), ("G2", g2, "#f58518", "^")):
        coordinates = np.asarray([np.dot(np.asarray(sample_map[key]["p"]), vertices) for key in intersection])
        axis.scatter(coordinates[:, 0], coordinates[:, 1], s=10, alpha=0.45, label=label, color=color, marker=marker)
    axis.plot([0, 1, 0.5, 0], [0, 0, math.sqrt(3)/2, 0], color="black"); axis.text(-0.04, -0.04, "p2"); axis.text(1.01, -0.04, "p4"); axis.text(0.48, math.sqrt(3)/2 + .03, "p6"); axis.set_aspect("equal"); axis.axis("off"); axis.legend(); axis.set_title("Three-way gate probability distribution")
    outputs += _save(figure, directory, "ternary_probability_distribution")
    return outputs, matrix


def _contact_sheet(candidates, market_root, destination, columns=8):
    paths = [_resolve_image(market_root, row) for row in candidates]
    rows = int(math.ceil(len(paths) / float(columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.0, rows * 3.0))
    axes = np.asarray(axes).reshape(-1)
    for axis, candidate, path in zip(axes, candidates, paths):
        axis.imshow(plt.imread(str(path))); axis.axis("off")
        axis.set_title("{}\np{} c{}".format(candidate["split"], candidate["pid"], candidate["camid"]), fontsize=6)
    for axis in axes[len(paths):]: axis.axis("off")
    figure.suptitle("Blind annotation contact sheet: no G1/G2 gate values displayed", y=1.0)
    return _save(figure, destination, "blind_annotation_contact_sheet")


ANNOTATION_FIELDS = ("stable_sample_key", "image_type", "annotation_method", "annotation_version", "short_reason")
SAMPLE_GATING_FIELDS = ("stable_sample_key", "relative_path", "split", "pid", "camid", "image_type", "g1_w2", "g1_w4", "g1_w6", "g1_dominant_k", "g2_w2", "g2_w4", "g2_w6", "g2_dominant_k", "delta_w2", "delta_w4", "delta_w6")
TYPE_STAT_FIELDS = ("image_type", "model", "metric", "count", "mean", "std", "min", "max", "median", "q25", "q75", "ci95_low", "ci95_high", "dominant_ratio")


def _write_annotation_template(path):
    if not Path(path).exists(): _write_csv(path, ANNOTATION_FIELDS, [], delimiter="\t")
    _atomic_text(Path(path).with_name("ANNOTATION_README.md"), """# 盲标注说明

Market1501 不提供 clear/occluded/misaligned/side_view/back_view/blurred 的官方标签。只能先查看不含门控值的 contact sheet，再填写本 TSV；允许相同 stable_sample_key 以多个 image_type 行表示多标签。不得依据文件名、G1/G2 权重、dominant K、检索结果或预期结论修改类别。类别内展示固定按 selection_hash 升序取前 10 张。
""")


def _read_annotations(path, candidates):
    if not Path(path).is_file() or Path(path).stat().st_size == 0: return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows: return []
    if tuple(rows[0].keys()) != ANNOTATION_FIELDS: raise EvidenceError("Annotation TSV schema is invalid")
    valid, seen = {row["stable_sample_key"] for row in candidates}, set()
    for row in rows:
        key = (row["stable_sample_key"], row["image_type"])
        if row["stable_sample_key"] not in valid or row["image_type"] not in CATEGORIES or not all(row[field] for field in ("annotation_method", "annotation_version", "short_reason")) or key in seen:
            raise EvidenceError("Annotation TSV has invalid key/category/method/version/reason or duplicate")
        seen.add(key)
    return rows


def _type_outputs(annotations, candidates, g1, g2, market_root, sample_dir, table_dir, seed, replicates):
    if not annotations:
        _write_csv(Path(table_dir).parent / "manifests" / "sample_gating_weights.tsv", SAMPLE_GATING_FIELDS, [], delimiter="\t")
        _write_csv(Path(table_dir) / "image_type_gating_statistics.csv", TYPE_STAT_FIELDS, [])
        return [], [], "not_recorded: blind annotations are empty"
    candidate_by_key = {row["stable_sample_key"]: row for row in candidates}
    annotation_by_type = {category: [] for category in CATEGORIES}
    for row in annotations: annotation_by_type[row["image_type"]].append(row["stable_sample_key"])
    sample_rows, summary_rows, figures = [], [], []
    for category in CATEGORIES:
        selected = sorted((candidate_by_key[key] for key in set(annotation_by_type[category])), key=lambda row: row["selection_hash"])
        if 0 < len(selected) < 5:
            # Required fail-closed boundary: do not pretend a 1–4 sample
            # category supports a type-level interpretation.
            raise EvidenceError("{} has fewer than five blind-annotated fixed samples".format(category))
        selected = selected[:10]
        if not selected: continue
        for candidate in selected:
            key = candidate["stable_sample_key"]
            row = {field: candidate[field] for field in ("stable_sample_key", "relative_path", "split", "pid", "camid")}
            row["image_type"] = category
            for index, scale in enumerate(SCALES):
                row["g1_w{}".format(scale)], row["g2_w{}".format(scale)] = g1[key]["w"][index], g2[key]["w"][index]
                row["delta_w{}".format(scale)] = g2[key]["w"][index] - g1[key]["w"][index]
            row["g1_dominant_k"], row["g2_dominant_k"] = g1[key]["dominant"], g2[key]["dominant"]
            sample_rows.append(row)
        figure, axes = plt.subplots(len(selected), 1, figsize=(9.0, 4.6 * len(selected)))
        axes = [axes] if len(selected) == 1 else axes
        for axis, candidate in zip(axes, selected):
            key = candidate["stable_sample_key"]; axis.imshow(plt.imread(str(_resolve_image(market_root, candidate)))); axis.axis("off")
            axis.set_title("{} | {} pid={} camid={}\nG1 w=({:.4f},{:.4f},{:.4f}) dominant K{} | G2 w=({:.4f},{:.4f},{:.4f}) dominant K{} | Δw=({:+.4f},{:+.4f},{:+.4f})".format(candidate["relative_path"], category, candidate["pid"], candidate["camid"], *g1[key]["w"], g1[key]["dominant"], *g2[key]["w"], g2[key]["dominant"], *(np.asarray(g2[key]["w"]) - np.asarray(g1[key]["w"]))), fontsize=7, loc="left")
        figure.suptitle("Blind-annotated fixed samples: {}".format(category), y=1.0); figures += _save(figure, Path(sample_dir) / category, "g1_vs_g2_contact_sheet")
        for label, sample_map in ((G1_LABEL, g1), (G2_LABEL, g2)):
            subset = {row["stable_sample_key"]: sample_map[row["stable_sample_key"]] for row in selected}
            metrics = _collapse_per_sample(subset)
            for index, scale in enumerate(SCALES):
                values = [sample["w"][index] for sample in subset.values()]
                statistic = _describe(values, seed, replicates)
                summary_rows.append(dict({"image_type": category, "model": label, "metric": "w{}".format(scale), "dominant_ratio": sum(1 for sample in subset.values() if sample["dominant"] == scale) / float(len(subset))}, **statistic))
            summary_rows.append(dict({"image_type": category, "model": label, "metric": "normalized_entropy", "dominant_ratio": NOT_APPLICABLE}, **_describe([item["normalized_entropy"] for item in metrics.values()], seed, replicates)))
    _write_csv(Path(table_dir).parent / "manifests" / "sample_gating_weights.tsv", SAMPLE_GATING_FIELDS, sample_rows, delimiter="\t")
    _write_csv(Path(table_dir) / "image_type_gating_statistics.csv", TYPE_STAT_FIELDS, summary_rows)
    return sample_rows, summary_rows, figures


def _bootstrap_difference(left, right, seed, replicates):
    if not left or not right:
        return NOT_RECORDED, NOT_RECORDED
    generator = np.random.RandomState(int(seed))
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    samples = (left[generator.randint(0, len(left), size=(int(replicates), len(left)))].mean(axis=1) - right[generator.randint(0, len(right), size=(int(replicates), len(right)))].mean(axis=1))
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def _k6_type_comparisons(annotations, candidates, g2, table_dir, seed, replicates):
    fields = ("comparison", "clear_count", "reference_count", "metric", "clear_mean", "reference_mean", "delta_clear_minus_reference", "ci95_low", "ci95_high", "status")
    destination = Path(table_dir) / "k6_image_type_comparisons.csv"
    if not annotations:
        _write_csv(destination, fields, [])
        return []
    candidate_by_key = {row["stable_sample_key"]: row for row in candidates}
    types = {category: set() for category in CATEGORIES}
    for annotation in annotations: types[annotation["image_type"]].add(annotation["stable_sample_key"])
    selected = {
        category: sorted((candidate_by_key[key] for key in keys), key=lambda row: row["selection_hash"])[:10]
        for category, keys in types.items()
    }
    clear = [row["stable_sample_key"] for row in selected["clear"]]
    other_pool = sorted({row["stable_sample_key"] for category in CATEGORIES if category != "clear" for row in selected[category]} - set(clear))
    comparisons = [("clear_vs_nonclear", other_pool), ("clear_vs_blurred", [row["stable_sample_key"] for row in selected["blurred"]]), ("clear_vs_occluded", [row["stable_sample_key"] for row in selected["occluded"]]), ("clear_vs_misaligned", [row["stable_sample_key"] for row in selected["misaligned"]])]
    rows = []
    for name, reference in comparisons:
        status = "measured" if len(clear) >= 5 and len(reference) >= 5 else "insufficient_blind_annotated_fixed_samples"
        for metric, resolve in (("g2_w6", lambda item: item["w"][2]), ("g2_dominant_k6_ratio", lambda item: 1.0 if item["dominant"] == 6 else 0.0)):
            left, right = [resolve(g2[key]) for key in clear], [resolve(g2[key]) for key in reference]
            low, high = _bootstrap_difference(left, right, seed, replicates) if status == "measured" else (NOT_RECORDED, NOT_RECORDED)
            rows.append({"comparison": name, "clear_count": len(clear), "reference_count": len(reference), "metric": metric, "clear_mean": float(np.mean(left)) if left else NOT_RECORDED, "reference_mean": float(np.mean(right)) if right else NOT_RECORDED, "delta_clear_minus_reference": float(np.mean(left)-np.mean(right)) if left and right else NOT_RECORDED, "ci95_low": low, "ci95_high": high, "status": status})
    _write_csv(destination, fields, rows)
    return rows


def _archive(output_dir, archive_path):
    output_dir, archive_path = Path(output_dir), Path(archive_path)
    with tarfile.open(str(archive_path), "w:gz") as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file(): archive.add(str(path), arcname=str(Path(output_dir.name) / path.relative_to(output_dir)))
    return _file_evidence(archive_path)


def _readme(output_dir, pairing, annotation_status):
    _atomic_text(Path(output_dir) / "README.md", """# G1 vs G2 Dynamic Gating 分析

所有固定样本均先由 query/gallery 的 SHA256 规则冻结，随后才运行 G1/G2 gate 推理。`p2+p4+p6=1`；两者均为 scaled-softmax，因此 `w2+w4+w6=3`。训练末轮与固定 query/gallery 样本的统计范围在表中分开标记，训练聚合记录未保存分位数或 bootstrap 所需的原始值时标为 `not_recorded`。

配对状态：`{}`。图像类型不是 Market1501 官方标签；当前盲标注状态：`{}`。没有足够样本时，不能据此主张 K6 偏好清晰细节。
""".format(json.dumps(pairing, ensure_ascii=False), annotation_status))


def run_analysis(g1_spec, g2_spec, dataset_root, output_dir, device=None, query_limit=256, gallery_limit=256, bootstrap_seed=42, bootstrap_replicates=1000, prepare_annotations_only=False, resume=False, archive_path=None):
    output_dir = Path(output_dir)
    if output_dir.exists() and not resume:
        raise EvidenceError("Output directory exists; use --resume only after reviewing its frozen manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, figures, manifests, samples = (output_dir / "tables", output_dir / "figures", output_dir / "manifests", output_dir / "samples")
    for directory in (tables, figures, manifests, samples): directory.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES: (samples / category).mkdir(exist_ok=True)

    g1, g2 = validate_formal_run(g1_spec), validate_formal_run(g2_spec)
    validate_pair(g1, g2)
    candidate_path = manifests / "fixed_candidate_samples.tsv"
    if candidate_path.exists(): candidates = read_fixed_candidates(candidate_path); market_root, _query, _gallery = _market_directories(dataset_root)
    else: candidates, market_root = build_fixed_candidates(dataset_root, candidate_path, query_limit, gallery_limit)
    _write_annotation_template(manifests / "image_type_annotations.tsv")
    contact_paths = _contact_sheet(candidates, market_root, figures / "blind_annotation")

    manifest = {
        "analysis": "reproducible_g1_vs_g2_dynamic_gating", "selection_prefix": SELECTION_PREFIX,
        "candidate_manifest": _file_evidence(candidate_path), "candidate_counts": dict(Counter(row["split"] for row in candidates)),
        "g1": {"formal_commit": G1_COMMIT, "checkpoint_sha256": G1_SHA256, "gate_input": "global", "inputs": g1["artifacts"]},
        "g2": {"formal_commit": G2_COMMIT, "checkpoint_sha256": G2_SHA256, "gate_input": "concat_global_local", "inputs": g2["artifacts"]},
        "contact_sheets": contact_paths, "prepare_annotations_only": bool(prepare_annotations_only),
    }
    _write_json(output_dir / "analysis_manifest.json", manifest)
    if prepare_annotations_only:
        _readme(output_dir, NOT_RECORDED, "awaiting_manual_blind_annotation")
        return manifest

    g1_path, g2_path = manifests / "g1_fixed_gating_samples.tsv", manifests / "g2_fixed_gating_samples.tsv"
    if resume and g1_path.is_file() and g2_path.is_file():
        g1_map, g2_map = read_fixed_gates(g1_path, G1_SHA256), read_fixed_gates(g2_path, G2_SHA256)
    else:
        g1_rows = extract_fixed_gates(g1, candidates, market_root, g1_path, device=device)
        g2_rows = extract_fixed_gates(g2, candidates, market_root, g2_path, device=device)
        g1_map, g2_map = read_fixed_gates(g1_path, G1_SHA256), read_fixed_gates(g2_path, G2_SHA256)
        if len(g1_rows) != len(g2_rows): raise EvidenceError("G1/G2 inference counts differ")
    expected_fixed_keys = {row["stable_sample_key"] for row in candidates}
    if set(g1_map) != set(g2_map) or set(g1_map) != expected_fixed_keys:
        # Both were derived from the same frozen manifest; any mismatch signals
        # a data-path or extraction fault and is not safe to interpret.
        raise EvidenceError("G1/G2 fixed sample pairing does not exactly match the frozen candidate manifest")
    weight_rows = _training_weight_rows(G1_LABEL, g1["history"], "training_final_epoch_all_training_samples") + _training_weight_rows(G2_LABEL, g2["history"], "training_final_epoch_all_training_samples")
    weight_rows += _sample_weight_rows(G1_LABEL, g1_map, "fixed_query_gallery_samples", bootstrap_seed, bootstrap_replicates) + _sample_weight_rows(G2_LABEL, g2_map, "fixed_query_gallery_samples", bootstrap_seed, bootstrap_replicates)
    _write_csv(tables / "gating_weight_statistics.csv", WEIGHT_FIELDS, weight_rows)
    _write_markdown_from_csv(tables / "gating_weight_statistics.md", "G1/G2 Dynamic Gating 权重统计", WEIGHT_FIELDS, weight_rows, "Markdown 由同一机器生成行写出；`training_final_epoch_all_training_samples` 为聚合证据，未记录的分位数和 bootstrap CI 不会被推断。")
    _write_json(tables / "gating_weight_statistics.json", {"rows": weight_rows})
    collapse_rows, metrics1, metrics2, intersection, pairing = _collapse_rows(g1_map, g2_map, bootstrap_seed, bootstrap_replicates)
    collapse_rows += _training_collapse_rows(G1_LABEL, g1["history"]) + _training_collapse_rows(G2_LABEL, g2["history"])
    _write_csv(tables / "gating_collapse_comparison.csv", COLLAPSE_FIELDS, collapse_rows)
    _write_markdown_from_csv(tables / "gating_collapse_comparison.md", "G1/G2 门控分布与集中度比较", COLLAPSE_FIELDS, collapse_rows, "固定样本表按 stable_sample_key 完全配对；跨模型归一化 entropy 使用 H/log(3)。不能由本表单独推出因果关系。")
    _write_json(tables / "gating_collapse_comparison.json", {"pairing": pairing, "rows": collapse_rows})
    figure_paths, transition = _plot_distributions(g1_map, g2_map, metrics1, metrics2, intersection, figures)
    _write_csv(tables / "dominant_k_transition.csv", ("g1_dominant_k", "g2_dominant_k", "count"), [{"g1_dominant_k": left, "g2_dominant_k": right, "count": int(transition[i, j])} for i, left in enumerate(SCALES) for j, right in enumerate(SCALES)])
    paired_rows = []
    for key in intersection:
        base, current = g1_map[key], g2_map[key]
        paired_rows.append({"stable_sample_key": key, "split": base["split"], "relative_path": base["relative_path"], "pid": base["pid"], "camid": base["camid"], "g1_w2": base["w"][0], "g1_w4": base["w"][1], "g1_w6": base["w"][2], "g1_dominant_k": base["dominant"], "g2_w2": current["w"][0], "g2_w4": current["w"][1], "g2_w6": current["w"][2], "g2_dominant_k": current["dominant"], "delta_w2": current["w"][0]-base["w"][0], "delta_w4": current["w"][1]-base["w"][1], "delta_w6": current["w"][2]-base["w"][2]})
    _write_csv(tables / "paired_sample_weight_deltas.csv", tuple(paired_rows[0].keys()), paired_rows)
    annotations = _read_annotations(manifests / "image_type_annotations.tsv", candidates)
    sample_rows, type_rows, type_figures = _type_outputs(annotations, candidates, g1_map, g2_map, market_root, samples, tables, bootstrap_seed, bootstrap_replicates)
    k6_rows = _k6_type_comparisons(annotations, candidates, g2_map, tables, bootstrap_seed, bootstrap_replicates)
    _write_markdown_from_csv(tables / "k6_image_type_comparisons.md", "G2 K6 与盲标注图像类型比较", ("comparison", "clear_count", "reference_count", "metric", "clear_mean", "reference_mean", "delta_clear_minus_reference", "ci95_low", "ci95_high", "status"), k6_rows, "该表仅描述固定盲标注样本中的相关性；样本不足时不能主张 K6 偏好清晰细节。")
    annotation_status = "measured" if annotations else "not_recorded: blind annotations are empty"
    manifest.update({"fixed_gating_samples": {"g1": _file_evidence(g1_path), "g2": _file_evidence(g2_path)}, "pairing": pairing, "figures": figure_paths, "type_figures": type_figures, "annotation_status": annotation_status, "type_sample_count": len(sample_rows), "type_summary_count": len(type_rows), "k6_type_comparison_count": len(k6_rows)})
    _write_json(output_dir / "analysis_manifest.json", manifest)
    _readme(output_dir, pairing, annotation_status)
    if archive_path:
        manifest["archive"] = _archive(output_dir, archive_path)
        _write_json(output_dir / "analysis_manifest.json", manifest)
    return manifest


def _default_specs(args):
    return (
        RunSpec("g1", G1_LABEL, "global", G1_COMMIT, G1_SHA256, Path(args.g1_run), Path(args.g1_output), Path(args.g1_checkpoint)),
        RunSpec("g2", G2_LABEL, "concat_global_local", G2_COMMIT, G2_SHA256, Path(args.g2_run), Path(args.g2_output), Path(args.g2_checkpoint)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-run", required=True); parser.add_argument("--g1-output", required=True); parser.add_argument("--g1-checkpoint", required=True)
    parser.add_argument("--g2-run", required=True); parser.add_argument("--g2-output", required=True); parser.add_argument("--g2-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--archive-path", default=None)
    parser.add_argument("--device", default=None); parser.add_argument("--query-limit", type=int, default=256); parser.add_argument("--gallery-limit", type=int, default=256)
    parser.add_argument("--bootstrap-seed", type=int, default=42); parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--prepare-annotations-only", action="store_true"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.query_limit, args.gallery_limit, args.bootstrap_replicates) <= 0: parser.error("limits and bootstrap replicates must be positive")
    g1, g2 = _default_specs(args)
    result = run_analysis(g1, g2, args.dataset_root, args.output_dir, device=args.device, query_limit=args.query_limit, gallery_limit=args.gallery_limit, bootstrap_seed=args.bootstrap_seed, bootstrap_replicates=args.bootstrap_replicates, prepare_annotations_only=args.prepare_annotations_only, resume=args.resume, archive_path=args.archive_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
