#!/usr/bin/env python
# encoding: utf-8
"""Compare Baseline and C2-L03 feature-distance distributions.

The script deliberately keeps feature extraction and pair analysis separate:
one deterministic query+gallery sample order is used for both checkpoints, pair
indices are derived only from pid/camid metadata, and distances are computed in
chunks instead of materializing an N x N matrix.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shlex
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np


PAIR_TYPES = (
    "same-id same-camera",
    "same-id different-camera",
    "different-id",
)
PAIR_TYPE_TO_CODE = {name: index for index, name in enumerate(PAIR_TYPES)}
MAX_SAMPLED_DIFFERENT_ID_PAIRS = 1000000
PID_FILTER_POLICIES = ("auto", "positive-only", "none")
MARKET_DATASET_NAMES = {"market1501", "market-1501"}


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract retrieval features from a Baseline and C2-L03 checkpoint, "
            "then compare three mutually exclusive unordered-pair distances."
        )
    )
    parser.add_argument("--baseline-config-file", required=True)
    parser.add_argument("--baseline-weight", required=True)
    parser.add_argument(
        "--c2-config-file",
        "--config-file",
        dest="c2_config_file",
        required=True,
        help="C2-L03 config; --config-file is a compatibility alias.",
    )
    parser.add_argument(
        "--c2-weight",
        "--weight",
        dest="c2_weight",
        required=True,
        help="C2-L03 checkpoint; --weight is a compatibility alias.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional dataset registration-name override (market1501 or dukemtmc).",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Optional DATASETS.ROOT_DIR override.",
    )
    parser.add_argument(
        "--pid-filter",
        choices=PID_FILTER_POLICIES,
        default="auto",
        help=(
            "Evaluation identity filter. auto applies pid > 0 to Market1501 "
            "and leaves other datasets unchanged."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-different-id-pairs",
        type=int,
        default=200000,
        help=(
            "Positive uniform no-replacement sample limit (maximum 1,000,000). "
            "All same-ID pairs are always retained."
        ),
    )
    parser.add_argument(
        "--distance-chunk-size",
        type=int,
        default=4096,
        help="Number of pairs per distance block; the conservative default limits RAM.",
    )
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="Override TEST DataLoader workers; -1 uses the config.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Still write CSV/JSON/Markdown, but skip PNG plots.",
    )
    return parser


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_rows(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(("|".join(str(value) for value in row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def classify_pair(pid_i, camid_i, pid_j, camid_j):
    if pid_i != pid_j:
        return "different-id"
    if camid_i == camid_j:
        return "same-id same-camera"
    return "same-id different-camera"


def validate_pair_indices(pair_i, pair_j, pair_type_codes, pids, camids):
    if not (len(pair_i) == len(pair_j) == len(pair_type_codes)):
        raise ValueError("Pair arrays have inconsistent lengths.")
    if np.any(pair_i >= pair_j):
        raise ValueError("Every pair must satisfy i < j; self/reversed pairs found.")
    if len(pair_i) != len(set(zip(pair_i.tolist(), pair_j.tolist()))):
        raise ValueError("Duplicate unordered pairs found.")
    for index in range(len(pair_i)):
        expected = PAIR_TYPE_TO_CODE[
            classify_pair(
                pids[pair_i[index]],
                camids[pair_i[index]],
                pids[pair_j[index]],
                camids[pair_j[index]],
            )
        ]
        if int(pair_type_codes[index]) != expected:
            raise ValueError("Pair type mismatch at pair row {}.".format(index))


def _all_different_id_pairs(pids):
    pair_i = []
    pair_j = []
    for i in range(len(pids) - 1):
        for j in range(i + 1, len(pids)):
            if pids[i] != pids[j]:
                pair_i.append(i)
                pair_j.append(j)
    return pair_i, pair_j


def _sample_different_id_pairs(pids, sample_size, seed):
    """Uniformly sample unordered different-id pairs by rejection sampling."""
    rng = random.Random(seed)
    selected = set()
    n = len(pids)
    max_attempts = max(sample_size * 100, 10000)
    attempts = 0
    while len(selected) < sample_size:
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        if i > j:
            i, j = j, i
        if pids[i] == pids[j]:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("Unable to sample enough different-id pairs.")
            continue
        selected.add((i, j))
        attempts += 1
        if attempts > max_attempts and len(selected) < sample_size:
            raise RuntimeError("Different-id pair sampling exceeded safety limit.")
    pairs = sorted(selected)
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def generate_pair_indices(pids, camids, max_different_id_pairs=200000, seed=42):
    pids = np.asarray(pids)
    camids = np.asarray(camids)
    if len(pids) != len(camids):
        raise ValueError("pids and camids must have the same length.")

    by_pid = defaultdict(list)
    for index, pid in enumerate(pids.tolist()):
        by_pid[pid].append(index)

    same_i = []
    same_j = []
    same_codes = []
    same_id_pair_count = 0
    for indices in by_pid.values():
        count = len(indices)
        same_id_pair_count += count * (count - 1) // 2
        for left in range(count - 1):
            i = indices[left]
            for right in range(left + 1, count):
                j = indices[right]
                pair_type = classify_pair(pids[i], camids[i], pids[j], camids[j])
                same_i.append(i)
                same_j.append(j)
                same_codes.append(PAIR_TYPE_TO_CODE[pair_type])

    total_pairs = len(pids) * (len(pids) - 1) // 2
    different_candidate_count = total_pairs - same_id_pair_count
    if max_different_id_pairs <= 0 or max_different_id_pairs >= different_candidate_count:
        diff_i, diff_j = _all_different_id_pairs(pids)
        sampled = False
    else:
        diff_i, diff_j = _sample_different_id_pairs(
            pids, max_different_id_pairs, seed
        )
        sampled = True

    pair_i = np.asarray(same_i + diff_i, dtype=np.int64)
    pair_j = np.asarray(same_j + diff_j, dtype=np.int64)
    pair_type_codes = np.asarray(
        same_codes + [PAIR_TYPE_TO_CODE["different-id"]] * len(diff_i),
        dtype=np.int8,
    )
    validate_pair_indices(pair_i, pair_j, pair_type_codes, pids, camids)
    return pair_i, pair_j, pair_type_codes, {
        "different_id_candidate_count": int(different_candidate_count),
        "different_id_used_count": int(len(diff_i)),
        "different_id_sampled": sampled,
        "sampling_seed": int(seed) if sampled else None,
    }


def l2_normalize(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A zero-norm feature cannot be L2-normalized.")
    return features / norms


def pairwise_squared_euclidean(features, pair_i, pair_j, chunk_size=4096):
    features = np.asarray(features, dtype=np.float32)
    result = np.empty(len(pair_i), dtype=np.float32)
    for start in range(0, len(pair_i), chunk_size):
        end = min(start + chunk_size, len(pair_i))
        delta = features[pair_i[start:end]] - features[pair_j[start:end]]
        result[start:end] = np.einsum("ij,ij->i", delta, delta)
    np.maximum(result, 0.0, out=result)
    return result


def summarize_distances(distances):
    values = np.asarray(distances, dtype=np.float64)
    if values.size == 0:
        return {
            key: None
            for key in (
                "count",
                "mean",
                "std",
                "median",
                "q25",
                "q75",
                "q05",
                "q95",
            )
        }
    q05, q25, median, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "q05": float(q05),
        "q95": float(q95),
    }


def _load_cfg(config_file, dataset_override="", data_root_override=""):
    from config.defaults import _C

    local_cfg = _C.clone()
    local_cfg.merge_from_file(config_file)
    local_cfg.defrost()
    if dataset_override:
        local_cfg.DATASETS.NAMES = dataset_override
    if data_root_override:
        local_cfg.DATASETS.ROOT_DIR = data_root_override
    local_cfg.freeze()
    return local_cfg


def _flatten_cfg(node, prefix=""):
    flattened = {}
    for key in node.keys():
        value = node[key]
        dotted_key = "{}.{}".format(prefix, key) if prefix else str(key)
        if hasattr(value, "keys"):
            flattened.update(_flatten_cfg(value, dotted_key))
        elif isinstance(value, (list, tuple)):
            flattened[dotted_key] = tuple(value)
        else:
            flattened[dotted_key] = value
    return flattened


def _fairness_signature(cfg):
    allowed_differences = {
        "MODEL.CROSS_CAMERA_POSITIVE_ONLY",
        "MODEL.CROSS_CAMERA_POSITIVE_LAMBDA",
        "OUTPUT_DIR",
    }
    return {
        key: value
        for key, value in _flatten_cfg(cfg).items()
        if key not in allowed_differences
    }


def _optional_model_flag(cfg, key):
    return bool(cfg.MODEL[key]) if key in cfg.MODEL else False


def _validate_protocols(baseline_cfg, c2_cfg):
    baseline_signature = _fairness_signature(baseline_cfg)
    c2_signature = _fairness_signature(c2_cfg)
    if baseline_signature != c2_signature:
        differing_keys = sorted(
            key
            for key in set(baseline_signature) | set(c2_signature)
            if baseline_signature.get(key) != c2_signature.get(key)
        )
        differences = {
            key: {
                "baseline": baseline_signature.get(key),
                "c2_l03": c2_signature.get(key),
            }
            for key in differing_keys
        }
        raise ValueError(
            "Baseline and C2 configs differ outside the C2 switch/lambda "
            "and OUTPUT_DIR:\n{}".format(
                json.dumps(differences, ensure_ascii=False, indent=2),
            )
        )
    if baseline_cfg.TEST.NECK_FEAT != "after":
        raise ValueError("Distance analysis requires TEST.NECK_FEAT='after'.")
    if baseline_cfg.TEST.FEAT_NORM != "yes":
        raise ValueError("Distance analysis requires TEST.FEAT_NORM='yes'.")
    if baseline_cfg.TEST.RE_RANKING != "no":
        raise ValueError("Distance analysis requires re-ranking disabled.")
    if bool(baseline_cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY):
        raise ValueError("Baseline config unexpectedly enables C2.")
    if not bool(c2_cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY):
        raise ValueError("C2 config does not enable CROSS_CAMERA_POSITIVE_ONLY.")
    if abs(float(c2_cfg.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA) - 0.3) > 1e-12:
        raise ValueError("C2-L03 requires auxiliary loss weight lambda=0.3.")
    if str(c2_cfg.MODEL.CROSS_CAMERA_POSITIVE_MODE).lower() != "mean":
        raise ValueError("C2-L03 requires CROSS_CAMERA_POSITIVE_MODE='mean'.")
    if _optional_model_flag(baseline_cfg, "CAMERA_AWARE_TRIPLET") or (
        _optional_model_flag(c2_cfg, "CAMERA_AWARE_TRIPLET")
    ):
        raise ValueError("Full CAAT/CAMERA_AWARE_TRIPLET is not C2-L03.")
    if _optional_model_flag(baseline_cfg, "SAME_CAMERA_POSITIVE_ONLY") or (
        _optional_model_flag(c2_cfg, "SAME_CAMERA_POSITIVE_ONLY")
    ):
        raise ValueError("Same-camera positive only is not C2-L03.")
    if not bool(c2_cfg.MODEL.PART_ATTENTION) or int(
        c2_cfg.MODEL.PART_ATTENTION_PARTS
    ) != 6:
        raise ValueError("Current C2-L03 requires Part Attention with K=6.")
    if str(c2_cfg.MODEL.METRIC_LOSS_TYPE) != "triplet":
        raise ValueError("Current C2-L03 requires METRIC_LOSS_TYPE='triplet'.")


def _prepare_new_output_dir(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise FileExistsError("Output path exists and is not a directory: {}".format(path))
        with os.scandir(path) as entries:
            is_nonempty = next(entries, None) is not None
        if is_nonempty:
            raise FileExistsError(
                "Output directory is not empty; choose a new directory so existing "
                "analysis results are not overwritten: {}".format(path)
            )
    else:
        os.makedirs(path)


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _load_checkpoint(model, checkpoint_path, device):
    import torch

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _unwrap_state_dict(checkpoint)
    current_state = model.state_dict()
    compatible = {}
    skipped = []
    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key in current_state and tuple(current_state[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing = [key for key in current_state if key not in compatible]
    critical_missing = [key for key in missing if not key.startswith("classifier.")]
    if critical_missing:
        raise RuntimeError(
            "Checkpoint is missing incompatible model parameters: {}".format(
                ", ".join(critical_missing[:20])
            )
        )
    model.load_state_dict(compatible, strict=False)
    return {"missing": missing, "skipped": skipped}


def _normalized_dataset_name(dataset_name):
    if isinstance(dataset_name, (tuple, list)):
        if len(dataset_name) != 1:
            raise ValueError(
                "Distance analysis requires exactly one dataset, got: {}".format(
                    dataset_name
                )
            )
        dataset_name = dataset_name[0]
    return str(dataset_name).strip().lower()


def _resolve_pid_filter(dataset_name, requested_policy):
    if requested_policy not in PID_FILTER_POLICIES:
        raise ValueError(
            "Unknown pid filter policy '{}'; expected one of {}.".format(
                requested_policy, ", ".join(PID_FILTER_POLICIES)
            )
        )
    if requested_policy != "auto":
        return requested_policy
    if _normalized_dataset_name(dataset_name) in MARKET_DATASET_NAMES:
        return "positive-only"
    return "none"


def _build_sample_manifest(samples, split_names):
    if len(samples) != len(split_names):
        raise ValueError("Sample and split-name counts differ.")
    return [
        {
            "sample_index": index,
            "image_path": sample[0],
            "pid": int(sample[1]),
            "camid": int(sample[2]),
            "split": split_names[index],
        }
        for index, sample in enumerate(samples)
    ]


def _filter_eval_samples(query, gallery, dataset_name, requested_policy="auto"):
    effective_policy = _resolve_pid_filter(dataset_name, requested_policy)
    entries = [(sample, "query") for sample in query]
    entries.extend((sample, "gallery") for sample in gallery)

    loader_output_counts = {
        "query": len(query),
        "gallery": len(gallery),
        "total": len(entries),
    }
    kept_entries = []
    excluded_by_pid = Counter()
    excluded_by_split = Counter()
    for sample, split_name in entries:
        pid = int(sample[1])
        if effective_policy == "positive-only" and pid <= 0:
            excluded_by_pid[pid] += 1
            excluded_by_split[split_name] += 1
            continue
        kept_entries.append((sample, split_name))

    if not kept_entries:
        raise RuntimeError("PID filtering removed every evaluation sample.")

    samples = [sample for sample, _ in kept_entries]
    split_names = [split_name for _, split_name in kept_entries]
    analysis_counts = {
        "query": split_names.count("query"),
        "gallery": split_names.count("gallery"),
        "total": len(samples),
    }
    filter_report = {
        "requested_policy": requested_policy,
        "effective_policy": effective_policy,
        "predicate": "pid > 0" if effective_policy == "positive-only" else "none",
        "applied_before_feature_extraction": True,
        "applied_before_pair_generation": True,
        "loader_output_counts": loader_output_counts,
        "analysis_counts": analysis_counts,
        "excluded_nonpositive_count": int(sum(excluded_by_pid.values())),
        "excluded_by_split": {
            "query": int(excluded_by_split.get("query", 0)),
            "gallery": int(excluded_by_split.get("gallery", 0)),
        },
        "excluded_by_pid": {
            str(pid): int(count) for pid, count in sorted(excluded_by_pid.items())
        },
    }
    return samples, split_names, filter_report


def _audit_raw_image_sources(dataset):
    audit = {}
    for split_name, directory_attribute in (
        ("query", "query_dir"),
        ("gallery", "gallery_dir"),
    ):
        directory = getattr(dataset, directory_attribute, "")
        if not directory or not os.path.isdir(directory):
            audit[split_name] = {"available": False, "directory": directory}
            continue
        pid_counts = Counter()
        unparsed_count = 0
        image_count = 0
        for filename in sorted(os.listdir(directory)):
            if not filename.lower().endswith(".jpg"):
                continue
            image_count += 1
            match = re.match(r"^([-\d]+)_c\d+", filename)
            if match is None:
                unparsed_count += 1
                continue
            pid_counts[int(match.group(1))] += 1
        audit[split_name] = {
            "available": True,
            "directory": os.path.abspath(directory),
            "jpg_count": int(image_count),
            "pid_minus_one_count": int(pid_counts.get(-1, 0)),
            "pid_zero_count": int(pid_counts.get(0, 0)),
            "positive_pid_count": int(
                sum(count for pid, count in pid_counts.items() if pid > 0)
            ),
            "unparsed_count": int(unparsed_count),
        }
    return audit


def _build_sample_filter_consistency_checks(filter_report):
    checks = {}
    raw_audit = filter_report.get("raw_source_audit", {})
    loader_counts = filter_report["loader_output_counts"]
    analysis_counts = filter_report["analysis_counts"]
    excluded_by_split = filter_report["excluded_by_split"]
    for split_name in ("query", "gallery"):
        raw_split = raw_audit.get(split_name, {})
        if raw_split.get("available"):
            raw_to_loader = (
                int(raw_split["jpg_count"])
                - int(raw_split["pid_minus_one_count"])
                == int(loader_counts[split_name])
            )
        else:
            raw_to_loader = None
        loader_to_analysis = (
            int(loader_counts[split_name])
            - int(excluded_by_split[split_name])
            == int(analysis_counts[split_name])
        )
        checks[split_name] = {
            "raw_jpg_minus_pid_minus_one_equals_loader_output": raw_to_loader,
            "loader_output_minus_filter_exclusions_equals_analysis": (
                loader_to_analysis
            ),
        }
    return checks


def _assert_manifest_pid_policy(manifest, effective_policy):
    if effective_policy != "positive-only":
        return
    invalid_rows = [row for row in manifest if int(row["pid"]) <= 0]
    if invalid_rows:
        raise RuntimeError(
            "positive-only PID filter left {} non-positive samples.".format(
                len(invalid_rows)
            )
        )


def _build_eval_data(
    cfg,
    num_workers_override=-1,
    batch_size_override=0,
    pid_filter_policy="auto",
):
    from torch.utils.data import DataLoader

    from data.collate_batch import val_collate_fn
    from data.datasets import ImageDataset, init_dataset
    from data.transforms import build_transforms

    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    samples, split_names, filter_report = _filter_eval_samples(
        dataset.query,
        dataset.gallery,
        cfg.DATASETS.NAMES,
        requested_policy=pid_filter_policy,
    )
    filter_report["raw_source_audit"] = _audit_raw_image_sources(dataset)
    filter_report["consistency_checks"] = _build_sample_filter_consistency_checks(
        filter_report
    )
    if _normalized_dataset_name(cfg.DATASETS.NAMES) in MARKET_DATASET_NAMES:
        failed_checks = [
            "{}:{}".format(split_name, check_name)
            for split_name, split_checks in filter_report[
                "consistency_checks"
            ].items()
            for check_name, passed in split_checks.items()
            if passed is False
        ]
        if failed_checks:
            raise RuntimeError(
                "Market sample-filter count chain failed: {}".format(
                    ", ".join(failed_checks)
                )
            )
    val_set = ImageDataset(samples, build_transforms(cfg, is_train=False))
    workers = (
        int(cfg.DATALOADER.NUM_WORKERS)
        if num_workers_override < 0
        else num_workers_override
    )
    batch_size = (
        int(cfg.TEST.IMS_PER_BATCH)
        if batch_size_override <= 0
        else batch_size_override
    )
    loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=val_collate_fn,
    )
    manifest = _build_sample_manifest(samples, split_names)
    _assert_manifest_pid_policy(manifest, filter_report["effective_policy"])
    return dataset, loader, manifest, filter_report


def _build_model(cfg, num_classes):
    from modeling.baseline import Baseline

    return Baseline(
        num_classes,
        cfg.MODEL.LAST_STRIDE,
        cfg.MODEL.PRETRAIN_PATH,
        cfg.MODEL.NECK,
        cfg.TEST.NECK_FEAT,
        cfg.MODEL.NAME,
        "self",
        part_attention=cfg.MODEL.PART_ATTENTION,
        part_attention_parts=cfg.MODEL.PART_ATTENTION_PARTS,
    )


def _extract_features(cfg, loader, num_classes, checkpoint_path, device):
    import torch

    model = _build_model(cfg, num_classes)
    checkpoint_report = _load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()
    batches = []
    observed_pids = []
    observed_camids = []
    with torch.no_grad():
        for images, pids, camids in loader:
            features = model(images.to(device))
            batches.append(features.detach().cpu())
            observed_pids.extend(int(value) for value in pids)
            observed_camids.extend(int(value) for value in camids)
    features = torch.cat(batches, dim=0).numpy()
    return features, observed_pids, observed_camids, checkpoint_report


def _write_samples(path, manifest):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_index", "image_path", "pid", "camid", "split"),
        )
        writer.writeheader()
        writer.writerows(manifest)


def _write_pairs_and_distances(
    output_dir,
    pair_i,
    pair_j,
    pair_type_codes,
    baseline_distances,
    c2_distances,
    sampling_seed,
):
    pair_index_path = os.path.join(output_dir, "pair_indices.csv")
    distance_path = os.path.join(output_dir, "pair_distances.csv")
    with open(pair_index_path, "w", encoding="utf-8", newline="") as pair_handle, open(
        distance_path, "w", encoding="utf-8", newline=""
    ) as distance_handle:
        pair_writer = csv.writer(pair_handle)
        distance_writer = csv.writer(distance_handle)
        pair_writer.writerow(
            (
                "pair_index",
                "sample_i",
                "sample_j",
                "pair_type",
                "sampled",
                "sampling_seed",
            )
        )
        distance_writer.writerow(
            (
                "pair_index",
                "sample_i",
                "sample_j",
                "pair_type",
                "baseline_distance",
                "c2_l03_distance",
                "delta",
            )
        )
        for index in range(len(pair_i)):
            pair_type = PAIR_TYPES[int(pair_type_codes[index])]
            sampled = pair_type == "different-id" and sampling_seed is not None
            pair_writer.writerow(
                (
                    index,
                    int(pair_i[index]),
                    int(pair_j[index]),
                    pair_type,
                    str(sampled).lower(),
                    sampling_seed if sampled else "",
                )
            )
            distance_writer.writerow(
                (
                    index,
                    int(pair_i[index]),
                    int(pair_j[index]),
                    pair_type,
                    "{:.10g}".format(float(baseline_distances[index])),
                    "{:.10g}".format(float(c2_distances[index])),
                    "{:.10g}".format(
                        float(c2_distances[index] - baseline_distances[index])
                    ),
                )
            )


def _build_summary(pair_type_codes, baseline_distances, c2_distances):
    rows = []
    for pair_type in PAIR_TYPES:
        code = PAIR_TYPE_TO_CODE[pair_type]
        mask = pair_type_codes == code
        for model_name, distances in (
            ("Baseline", baseline_distances),
            ("C2-L03", c2_distances),
        ):
            summary = summarize_distances(distances[mask])
            summary.update({"pair_type": pair_type, "model": model_name})
            rows.append(summary)
    return rows


def _build_separation_summary(pair_type_codes, baseline_distances, c2_distances):
    cross_camera_mask = (
        pair_type_codes == PAIR_TYPE_TO_CODE["same-id different-camera"]
    )
    different_id_mask = pair_type_codes == PAIR_TYPE_TO_CODE["different-id"]
    rows = []
    for model_name, distances in (
        ("Baseline", baseline_distances),
        ("C2-L03", c2_distances),
    ):
        cross_camera = summarize_distances(distances[cross_camera_mask])
        different_id = summarize_distances(distances[different_id_mask])
        rows.append(
            {
                "model": model_name,
                "mean_gap": (
                    None
                    if cross_camera["mean"] is None or different_id["mean"] is None
                    else different_id["mean"] - cross_camera["mean"]
                ),
                "median_gap": (
                    None
                    if cross_camera["median"] is None
                    or different_id["median"] is None
                    else different_id["median"] - cross_camera["median"]
                ),
                "definition": (
                    "different-id distance minus same-id different-camera distance"
                ),
            }
        )
    rows.append(
        {
            "model": "C2-L03 minus Baseline",
            "mean_gap": (
                None
                if rows[0]["mean_gap"] is None or rows[1]["mean_gap"] is None
                else rows[1]["mean_gap"] - rows[0]["mean_gap"]
            ),
            "median_gap": (
                None
                if rows[0]["median_gap"] is None
                or rows[1]["median_gap"] is None
                else rows[1]["median_gap"] - rows[0]["median_gap"]
            ),
            "definition": "change in the separation gap",
        }
    )
    return rows


def _write_summary(output_dir, rows, separation_rows):
    fields = (
        "pair_type",
        "model",
        "count",
        "mean",
        "std",
        "median",
        "q25",
        "q75",
        "q05",
        "q95",
    )
    with open(
        os.path.join(output_dir, "distance_summary.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(
        os.path.join(output_dir, "distance_summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    separation_fields = ("model", "mean_gap", "median_gap", "definition")
    with open(
        os.path.join(output_dir, "separation_gap_summary.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=separation_fields)
        writer.writeheader()
        writer.writerows(separation_rows)
    with open(
        os.path.join(output_dir, "separation_gap_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(separation_rows, handle, ensure_ascii=False, indent=2)

    lines = [
        "# Distance distribution summary",
        "",
        "| Pair type | Model | Count | Mean | Std | Median | Q25 | Q75 | Q05 | Q95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = []
        for field in ("mean", "std", "median", "q25", "q75", "q05", "q95"):
            value = row[field]
            values.append("TBD" if value is None else "{:.6f}".format(value))
        lines.append(
            "| {pair_type} | {model} | {count} | {stats} |".format(
                pair_type=row["pair_type"],
                model=row["model"],
                count=row["count"],
                stats=" | ".join(values),
            )
        )
    lines.extend(
        (
            "",
            "> Distances are squared Euclidean distances over L2-normalized "
            "BNNeck-after features. Conclusions require reviewing all three "
            "distributions, not only their means.",
            "",
            "## Cross-camera separation gap",
            "",
            "Gap = different-ID distance - same-ID different-camera distance.",
            "",
            "| Model | Mean gap | Median gap | Definition |",
            "|---|---:|---:|---|",
        )
    )
    for row in separation_rows:
        mean_gap = (
            "TBD" if row["mean_gap"] is None else "{:.6f}".format(row["mean_gap"])
        )
        median_gap = (
            "TBD"
            if row["median_gap"] is None
            else "{:.6f}".format(row["median_gap"])
        )
        lines.append(
            "| {model} | {mean_gap} | {median_gap} | {definition} |".format(
                model=row["model"],
                mean_gap=mean_gap,
                median_gap=median_gap,
                definition=row["definition"],
            )
        )
    lines.extend(
        (
            "",
            "> A larger positive gap is favorable, but it is only descriptive "
            "evidence and must be interpreted together with the full "
            "distributions and retrieval metrics.",
            "",
        )
    )
    with open(
        os.path.join(output_dir, "distance_summary.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines))


def _checkpoint_label(path):
    normalized = os.path.normpath(path)
    parent = os.path.basename(os.path.dirname(normalized))
    filename = os.path.basename(normalized)
    return os.path.join(parent, filename) if parent else filename


def _plot_annotation(context):
    if not context:
        return (
            "Dataset/checkpoints: synthetic test | Feature: BNNeck-after, L2 | "
            "Metric: squared Euclidean | Pair sampling: supplied test pairs"
        )
    return (
        "Dataset: {dataset} | Checkpoints: Baseline={baseline_checkpoint}; "
        "C2-L03={c2_checkpoint}\n"
        "Feature: BNNeck-after | Normalization: L2 | Metric: squared Euclidean | "
        "PID filter: {pid_filter} | Pairs: {sampling}"
    ).format(**context)


def _plot_distributions(
    output_dir,
    pair_type_codes,
    baseline_distances,
    c2_distances,
    context=None,
):
    os.environ.setdefault(
        "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "bot_reid_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, pair_type in zip(axes, PAIR_TYPES):
        mask = pair_type_codes == PAIR_TYPE_TO_CODE[pair_type]
        baseline_values = baseline_distances[mask]
        c2_values = c2_distances[mask]
        combined = np.concatenate((baseline_values, c2_values))
        bins = np.histogram_bin_edges(combined, bins=60)
        axis.hist(
            baseline_values,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Baseline (n={})".format(len(baseline_values)),
        )
        axis.hist(
            c2_values,
            bins=bins,
            density=True,
            alpha=0.5,
            label="C2-L03 (n={})".format(len(c2_values)),
        )
        axis.set_title(pair_type)
        axis.set_xlabel("Squared Euclidean distance")
        axis.set_ylabel("Density")
        axis.legend()
    figure.suptitle(
        "Baseline vs C2-L03: BNNeck-after, L2-normalized, identical unordered pairs"
    )
    figure.text(
        0.5,
        0.015,
        _plot_annotation(context),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 0.94))
    figure.savefig(
        os.path.join(output_dir, "distance_histogram.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, pair_type in zip(axes, PAIR_TYPES):
        mask = pair_type_codes == PAIR_TYPE_TO_CODE[pair_type]
        axis.boxplot(
            [baseline_distances[mask], c2_distances[mask]],
            showfliers=False,
        )
        axis.set_xticks([1, 2])
        axis.set_xticklabels(["Baseline", "C2-L03"])
        axis.set_title(pair_type)
        axis.set_ylabel("Squared Euclidean distance")
    figure.suptitle(
        "Baseline vs C2-L03: identical pair indices; outlier markers hidden"
    )
    figure.text(
        0.5,
        0.015,
        _plot_annotation(context),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 0.94))
    figure.savefig(
        os.path.join(output_dir, "distance_boxplot.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.distance_chunk_size <= 0:
        raise ValueError("--distance-chunk-size must be positive.")
    if not 0 < args.max_different_id_pairs <= MAX_SAMPLED_DIFFERENT_ID_PAIRS:
        raise ValueError(
            "--max-different-id-pairs must be between 1 and {:,}.".format(
                MAX_SAMPLED_DIFFERENT_ID_PAIRS
            )
        )

    baseline_cfg = _load_cfg(
        args.baseline_config_file, args.dataset, args.data_root
    )
    c2_cfg = _load_cfg(args.c2_config_file, args.dataset, args.data_root)
    _validate_protocols(baseline_cfg, c2_cfg)

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Feature extraction requires the project PyTorch environment."
        ) from error
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    _prepare_new_output_dir(args.output_dir)
    dataset, loader, manifest, filter_report = _build_eval_data(
        baseline_cfg,
        args.num_workers,
        args.batch_size,
        args.pid_filter,
    )
    baseline_features, baseline_pids, baseline_camids, baseline_load_report = (
        _extract_features(
            baseline_cfg,
            loader,
            dataset.num_train_pids,
            args.baseline_weight,
            device,
        )
    )
    c2_features, c2_pids, c2_camids, c2_load_report = _extract_features(
        c2_cfg, loader, dataset.num_train_pids, args.c2_weight, device
    )

    expected_pids = [row["pid"] for row in manifest]
    expected_camids = [row["camid"] for row in manifest]
    _assert_manifest_pid_policy(manifest, filter_report["effective_policy"])
    if baseline_pids != expected_pids or c2_pids != expected_pids:
        raise RuntimeError("Feature extraction pid order differs from sample manifest.")
    if baseline_camids != expected_camids or c2_camids != expected_camids:
        raise RuntimeError("Feature extraction camid order differs from sample manifest.")
    if baseline_features.shape != c2_features.shape:
        raise RuntimeError("Baseline and C2 feature shapes differ.")

    baseline_features = l2_normalize(baseline_features)
    c2_features = l2_normalize(c2_features)
    pids = np.asarray(expected_pids)
    camids = np.asarray(expected_camids)
    pair_i, pair_j, pair_type_codes, sampling = generate_pair_indices(
        pids, camids, args.max_different_id_pairs, args.seed
    )
    baseline_distances = pairwise_squared_euclidean(
        baseline_features, pair_i, pair_j, args.distance_chunk_size
    )
    c2_distances = pairwise_squared_euclidean(
        c2_features, pair_i, pair_j, args.distance_chunk_size
    )
    rows = _build_summary(pair_type_codes, baseline_distances, c2_distances)
    separation_rows = _build_separation_summary(
        pair_type_codes, baseline_distances, c2_distances
    )

    _write_samples(os.path.join(args.output_dir, "samples.csv"), manifest)
    with open(
        os.path.join(args.output_dir, "sample_filter.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(filter_report, handle, ensure_ascii=False, indent=2)
    _write_pairs_and_distances(
        args.output_dir,
        pair_i,
        pair_j,
        pair_type_codes,
        baseline_distances,
        c2_distances,
        sampling["sampling_seed"],
    )
    _write_summary(args.output_dir, rows, separation_rows)
    if not args.no_plots:
        if sampling["different_id_sampled"]:
            pair_sampling = (
                "all same-ID; different-ID uniform no-replacement "
                "n={count}, seed={seed}"
            ).format(
                count=sampling["different_id_used_count"],
                seed=sampling["sampling_seed"],
            )
        else:
            pair_sampling = "all same-ID and all different-ID unordered pairs"
        _plot_distributions(
            args.output_dir,
            pair_type_codes,
            baseline_distances,
            c2_distances,
            context={
                "dataset": str(baseline_cfg.DATASETS.NAMES),
                "baseline_checkpoint": _checkpoint_label(args.baseline_weight),
                "c2_checkpoint": _checkpoint_label(args.c2_weight),
                "pid_filter": filter_report["predicate"],
                "sampling": pair_sampling,
            },
        )

    pair_hash_rows = zip(
        pair_i.tolist(), pair_j.tolist(), pair_type_codes.tolist()
    )
    metadata = {
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "dataset": str(baseline_cfg.DATASETS.NAMES),
        "data_root": str(baseline_cfg.DATASETS.ROOT_DIR),
        "sample_count": len(manifest),
        "sample_filter": filter_report,
        "sample_order_hash": _sha256_rows(
            (
                row["sample_index"],
                row["image_path"],
                row["pid"],
                row["camid"],
                row["split"],
            )
            for row in manifest
        ),
        "pair_index_hash": _sha256_rows(pair_hash_rows),
        "pair_count": int(len(pair_i)),
        "sampling": sampling,
        "separation_gap": separation_rows,
        "feature_type": "BNNeck-after",
        "feature_normalization": "L2",
        "distance_metric": "squared Euclidean",
        "re_ranking": False,
        "camera_mean_debias": False,
        "feature_shape": list(baseline_features.shape),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "seed": args.seed,
        "analysis_script": {
            "path": os.path.abspath(__file__),
            "sha256": _sha256_file(__file__),
        },
        "standard_deviation": "population (ddof=0)",
        "quantile_method": "NumPy default linear interpolation",
        "baseline": {
            "config": args.baseline_config_file,
            "weight": args.baseline_weight,
            "weight_sha256": _sha256_file(args.baseline_weight),
            "checkpoint_load_report": baseline_load_report,
        },
        "c2_l03": {
            "config": args.c2_config_file,
            "weight": args.c2_weight,
            "weight_sha256": _sha256_file(args.c2_weight),
            "checkpoint_load_report": c2_load_report,
        },
        "quality_checks": {
            "identical_sample_order": True,
            "identical_pair_indices": True,
            "self_pairs_excluded": True,
            "unordered_pairs_unique": True,
            "pair_types_mutually_exclusive": True,
            "all_analysis_pids_positive": bool(
                all(int(row["pid"]) > 0 for row in manifest)
            ),
            "filter_applied_before_feature_extraction": bool(
                filter_report["applied_before_feature_extraction"]
            ),
            "filter_applied_before_pair_generation": bool(
                filter_report["applied_before_pair_generation"]
            ),
        },
    }
    with open(
        os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print("Distance analysis written to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
