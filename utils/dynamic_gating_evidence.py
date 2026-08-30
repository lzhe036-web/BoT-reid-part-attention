# encoding: utf-8
"""Observation-only evidence helpers for per-sample dynamic gating."""

from __future__ import absolute_import

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from utils.experiment_schema import GATING_STAT_FIELDS, SCHEMA_VERSION


DYNAMIC_GATING_SELECTION_RULE = (
    "sha256(stable_sample_key) ascending; first 256 query+gallery samples"
)
DYNAMIC_GATING_SAMPLE_FIELDS = (
    "stable_sample_key", "dataset_split", "pid", "camid", "p2", "p4", "p6",
    "w2", "w4", "w6", "entropy", "dominant_k", "checkpoint_sha256",
)
GATING_VALUE_RTOL = 1e-6
GATING_VALUE_ATOL = 1e-9


class DynamicGatingEvidenceError(RuntimeError):
    pass


def gating_scales(configuration):
    """Return active gate scales without changing part-feature extraction."""
    try:
        value = _config_value(
            configuration, "MODEL.MULTI_GRANULARITY_GATING_INPUT"
        )
    except (AttributeError, KeyError):
        value = "global"
    return (2, 6) if str(value) == "concat_z2_z6" else (2, 4, 6)


def gating_stat_fields(scales):
    fields = ["gating_temperature", "gating_sample_count"]
    for scale in scales:
        fields.extend((
            "p{}_mean".format(scale), "p{}_std".format(scale),
            "p{}_min".format(scale), "p{}_max".format(scale),
            "applied_w{}_mean".format(scale),
            "applied_w{}_std".format(scale),
        ))
    fields.append("mean_gate_entropy")
    fields.extend("dominant_k{}_ratio".format(scale) for scale in scales)
    return tuple(fields)


def dynamic_gating_sample_fields(scales):
    fields = ["stable_sample_key", "dataset_split", "pid", "camid"]
    fields.extend("p{}".format(scale) for scale in scales)
    fields.extend("w{}".format(scale) for scale in scales)
    fields.extend(("entropy", "dominant_k", "checkpoint_sha256"))
    return tuple(fields)


class GatingEpochAccumulator(object):
    """Exact sample-weighted moments without retaining per-sample tensors."""

    def __init__(self, temperature, scales=(2, 4, 6), weight_sum=None):
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("gating temperature must be finite and positive")
        self.temperature = temperature
        self.scales = tuple(int(scale) for scale in scales)
        if not self.scales or len(set(self.scales)) != len(self.scales):
            raise ValueError("gating scales must be non-empty and unique")
        self.weight_sum = float(
            len(self.scales) if weight_sum is None else weight_sum
        )
        self.reset()

    def reset(self):
        self.count = 0
        width = len(self.scales)
        self.sum_p = [0.0] * width
        self.sum_p2 = [0.0] * width
        self.min_p = [float("inf")] * width
        self.max_p = [float("-inf")] * width
        self.sum_w = [0.0] * width
        self.sum_w2 = [0.0] * width
        self.entropy_sum = 0.0
        self.dominant_counts = [0] * width

    def update(self, probabilities):
        if not torch.is_tensor(probabilities):
            probabilities = torch.as_tensor(probabilities)
        values = probabilities.detach().to(device="cpu", dtype=torch.float64)
        if values.dim() != 2 or values.size(1) != len(self.scales):
            raise DynamicGatingEvidenceError(
                "Gating probabilities must have shape [B,{}], got {}".format(
                    len(self.scales), tuple(values.shape)
                )
            )
        if values.size(0) == 0:
            return
        if not torch.isfinite(values).all():
            raise DynamicGatingEvidenceError("Gating probabilities must be finite")
        if bool((values < 0.0).any()):
            raise DynamicGatingEvidenceError("Gating probabilities must be non-negative")
        row_sums = values.sum(dim=1)
        if not torch.allclose(
                row_sums, torch.ones_like(row_sums),
                rtol=GATING_VALUE_RTOL, atol=GATING_VALUE_ATOL):
            max_abs_error = float((row_sums - 1.0).abs().max().item())
            raise DynamicGatingEvidenceError(
                "Gating probabilities must sum to one for every sample; "
                "max_abs_error={:.12g}, rtol={}, atol={}".format(
                    max_abs_error, GATING_VALUE_RTOL, GATING_VALUE_ATOL
                )
            )

        weights = values * self.weight_sum
        batch_count = int(values.size(0))
        self.count += batch_count
        for index in range(len(self.scales)):
            column = values[:, index]
            weight_column = weights[:, index]
            self.sum_p[index] += float(column.sum().item())
            self.sum_p2[index] += float((column * column).sum().item())
            self.min_p[index] = min(self.min_p[index], float(column.min().item()))
            self.max_p[index] = max(self.max_p[index], float(column.max().item()))
            self.sum_w[index] += float(weight_column.sum().item())
            self.sum_w2[index] += float(
                (weight_column * weight_column).sum().item()
            )
        safe_values = values.clamp_min(torch.finfo(values.dtype).tiny)
        entropy = -(values * safe_values.log()).sum(dim=1)
        self.entropy_sum += float(entropy.sum().item())
        dominant = values.argmax(dim=1)
        for index in range(len(self.scales)):
            self.dominant_counts[index] += int((dominant == index).sum().item())

    @staticmethod
    def _std(total, square_total, count):
        mean = total / float(count)
        variance = max(0.0, square_total / float(count) - mean * mean)
        return math.sqrt(variance)

    def summary(self):
        if self.count <= 0:
            raise DynamicGatingEvidenceError(
                "Cannot record a gating epoch with zero samples"
            )
        result = {
            "gating_temperature": self.temperature,
            "gating_sample_count": self.count,
            "mean_gate_entropy": self.entropy_sum / float(self.count),
        }
        for index, scale in enumerate(self.scales):
            label = str(scale)
            result["p{}_mean".format(label)] = (
                self.sum_p[index] / float(self.count)
            )
            result["p{}_std".format(label)] = self._std(
                self.sum_p[index], self.sum_p2[index], self.count
            )
            result["p{}_min".format(label)] = self.min_p[index]
            result["p{}_max".format(label)] = self.max_p[index]
            result["applied_w{}_mean".format(label)] = (
                self.sum_w[index] / float(self.count)
            )
            result["applied_w{}_std".format(label)] = self._std(
                self.sum_w[index], self.sum_w2[index], self.count
            )
            result["dominant_k{}_ratio".format(label)] = (
                self.dominant_counts[index] / float(self.count)
            )
        for field in gating_stat_fields(self.scales):
            value = result[field]
            if isinstance(value, float) and not math.isfinite(value):
                raise DynamicGatingEvidenceError(
                    "Non-finite gating statistic {}".format(field)
                )
        return result


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        ".{}.tmp.{}".format(path.name, os.getpid())
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def append_gating_epoch_record(output_dir, epoch, global_iteration,
                               epoch_length, statistics, scales=(2, 4, 6)):
    path = Path(output_dir) / "dynamic_gating_epoch_stats.jsonl"
    record = {
        "epoch": int(epoch),
        "global_iteration": int(global_iteration),
        "epoch_length": int(epoch_length),
    }
    record["gating_scales"] = list(scales)
    record.update({
        field: statistics[field] for field in gating_stat_fields(scales)
    })
    if record["epoch"] <= 0 or record["global_iteration"] <= 0:
        raise DynamicGatingEvidenceError("Epoch evidence counters must be positive")
    if record["epoch_length"] <= 0:
        raise DynamicGatingEvidenceError("Ignite epoch_length must be positive")
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    _atomic_write_text(
        path,
        existing + json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return path


def read_gating_epoch_records(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise DynamicGatingEvidenceError(
                "Invalid gating epoch JSON at line {}".format(line_number)
            ) from error
        rows.append(row)
    return rows


def _close(actual, expected, label):
    try:
        matches = math.isclose(
            float(actual), float(expected), rel_tol=GATING_VALUE_RTOL,
            abs_tol=GATING_VALUE_ATOL,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise DynamicGatingEvidenceError(
            "{} is missing or non-numeric".format(label)
        ) from error
    if not matches:
        raise DynamicGatingEvidenceError(
            "{} mismatch: {!r} != {!r}".format(label, actual, expected)
        )


def _config_value(configuration, dotted):
    value = configuration
    for part in dotted.split("."):
        if isinstance(value, dict):
            value = value[part]
        else:
            value = getattr(value, part)
    return value


def _validate_dataset_manifest(configuration, recorded_manifest):
    from data.build import collect_dataset_protocol
    current, _num_classes = collect_dataset_protocol(configuration)
    fields = (
        "dataset_name", "data_root", "train_image_count", "query_image_count",
        "gallery_image_count", "train_pid_count", "query_pid_count",
        "gallery_pid_count", "train_camera_count", "query_camera_count",
        "gallery_camera_count", "split_manifest_sha256",
        "combined_manifest_sha256",
    )
    for field in fields:
        if current.get(field) != recorded_manifest.get(field):
            raise DynamicGatingEvidenceError(
                "Dataset manifest changed for gating selection: {}".format(field)
            )


def _default_selection_resolver(configuration):
    from tools.analyze_dynamic_gating import select_samples
    return select_samples(configuration, limit=256)[0]


def validate_dynamic_gating_evidence(
        summary_path, samples_path, selected_checkpoint_sha256,
        resolved_config, gating_epoch_statistics, dataset_manifest,
        selection_resolver=None, dataset_validator=None):
    """Strictly validate bounded Dynamic Gating evidence before success."""
    summary_path = Path(summary_path).resolve()
    samples_path = Path(samples_path).resolve()
    if not summary_path.is_file() or not samples_path.is_file():
        raise DynamicGatingEvidenceError("Dynamic Gating summary/TSV is missing")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as error:
        raise DynamicGatingEvidenceError("Invalid Dynamic Gating summary JSON") from error
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise DynamicGatingEvidenceError("Dynamic Gating summary schema is not v5")
    if summary.get("source_checkpoint_sha256") != selected_checkpoint_sha256:
        raise DynamicGatingEvidenceError("Gating summary checkpoint SHA mismatch")
    if summary.get("selection_rule") != DYNAMIC_GATING_SELECTION_RULE:
        raise DynamicGatingEvidenceError("Gating selection rule mismatch")
    selected_count = summary.get("selected_sample_count")
    if isinstance(selected_count, bool) or not isinstance(selected_count, int) \
            or selected_count <= 0 or selected_count > 256:
        raise DynamicGatingEvidenceError("Invalid selected gating sample count")

    samples_evidence = summary.get("gating_samples")
    if not isinstance(samples_evidence, dict):
        raise DynamicGatingEvidenceError("Summary lacks gating_samples evidence")
    actual_sha = hashlib.sha256(samples_path.read_bytes()).hexdigest()
    if Path(str(samples_evidence.get("path", ""))).resolve() != samples_path:
        raise DynamicGatingEvidenceError("Summary gating TSV path mismatch")
    if samples_evidence.get("size_bytes") != samples_path.stat().st_size:
        raise DynamicGatingEvidenceError("Summary gating TSV size mismatch")
    if samples_evidence.get("sha256") != actual_sha:
        raise DynamicGatingEvidenceError("Summary gating TSV SHA256 mismatch")
    if samples_evidence.get("source_checkpoint_sha256") != selected_checkpoint_sha256:
        raise DynamicGatingEvidenceError("Summary TSV checkpoint SHA mismatch")
    if samples_evidence.get("selection_rule") != DYNAMIC_GATING_SELECTION_RULE:
        raise DynamicGatingEvidenceError("Summary TSV selection rule mismatch")

    scales = gating_scales(resolved_config)
    sample_fields = dynamic_gating_sample_fields(scales)
    expected_weight_sum = 1.0 if scales == (2, 6) else float(len(scales))
    text = samples_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise DynamicGatingEvidenceError("Gating TSV contains an empty line")
    reader = csv.DictReader(lines, delimiter="\t")
    if tuple(reader.fieldnames or ()) != sample_fields:
        raise DynamicGatingEvidenceError("Gating TSV header mismatch")
    rows = list(reader)
    if len(rows) != selected_count:
        raise DynamicGatingEvidenceError("Gating TSV/sample-count mismatch")

    probabilities = []
    keys = []
    parsed_rows = []
    for line_number, row in enumerate(rows, 2):
        if None in row or set(row) != set(sample_fields):
            raise DynamicGatingEvidenceError(
                "Gating TSV has extra/missing columns at line {}".format(line_number)
            )
        key = row["stable_sample_key"].strip()
        if not key:
            raise DynamicGatingEvidenceError("Gating sample key is empty")
        if key in keys:
            raise DynamicGatingEvidenceError("Duplicate gating sample key")
        keys.append(key)
        if row["dataset_split"] not in ("query", "gallery"):
            raise DynamicGatingEvidenceError("Invalid gating dataset split")
        try:
            pid, camid = int(row["pid"]), int(row["camid"])
            p = [float(row["p{}".format(scale)]) for scale in scales]
            w = [float(row["w{}".format(scale)]) for scale in scales]
            entropy = float(row["entropy"])
            dominant = int(row["dominant_k"])
        except (TypeError, ValueError) as error:
            raise DynamicGatingEvidenceError(
                "Unparseable gating TSV value at line {}".format(line_number)
            ) from error
        if not all(math.isfinite(value) and value >= 0.0 for value in p):
            raise DynamicGatingEvidenceError("Gating probabilities must be finite/non-negative")
        if not all(math.isfinite(value) for value in w + [entropy]):
            raise DynamicGatingEvidenceError("Gating weights/entropy must be finite")
        _close(sum(p), 1.0, "probability sum")
        for index in range(len(scales)):
            _close(
                w[index], expected_weight_sum * p[index],
                "applied gating weight"
            )
        recomputed_entropy = -sum(
            value * math.log(max(value, float.fromhex("0x1.0p-1022")))
            for value in p
        )
        _close(entropy, recomputed_entropy, "gating entropy")
        expected_dominant = scales[
            max(range(len(scales)), key=lambda index: p[index])
        ]
        if dominant not in scales or dominant != expected_dominant:
            raise DynamicGatingEvidenceError("Dominant K mismatch")
        if row["checkpoint_sha256"] != selected_checkpoint_sha256:
            raise DynamicGatingEvidenceError("Per-sample checkpoint SHA mismatch")
        probabilities.append(p)
        parsed_rows.append((key, row["dataset_split"], pid, camid))

    if dataset_validator is None:
        _validate_dataset_manifest(resolved_config, dataset_manifest)
    else:
        dataset_validator(resolved_config, dataset_manifest)
    resolver = selection_resolver or _default_selection_resolver
    expected = resolver(resolved_config)
    if isinstance(expected, tuple) and len(expected) == 2 \
            and isinstance(expected[1], int):
        expected = expected[0]
    expected_rows = [
        (str(item[0]), str(item[1]), int(item[3]), int(item[4]))
        for item in expected
    ]
    if parsed_rows != expected_rows:
        raise DynamicGatingEvidenceError(
            "Gating sample selection/order does not match the dataset"
        )
    expected_key_order = sorted(
        keys, key=lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest()
    )
    if keys != expected_key_order:
        raise DynamicGatingEvidenceError("Gating sample selection order mismatch")

    temperature = float(_config_value(
        resolved_config, "MODEL.MULTI_GRANULARITY_GATING_TAU"
    ))
    accumulator = GatingEpochAccumulator(
        temperature, scales=scales, weight_sum=expected_weight_sum
    )
    accumulator.update(probabilities)
    recomputed = accumulator.summary()
    deterministic = summary.get("deterministic_sample_statistics")
    training = summary.get("training_epoch_statistics")
    if not isinstance(deterministic, dict) or not isinstance(training, dict):
        raise DynamicGatingEvidenceError("Summary statistics are missing")
    for field in gating_stat_fields(scales):
        _close(deterministic.get(field), recomputed[field], "deterministic {}".format(field))
        _close(training.get(field), gating_epoch_statistics.get(field), "training {}".format(field))
    return {
        "summary": summary,
        "sample_count": selected_count,
        "statistics": recomputed,
        "samples_sha256": actual_sha,
    }
