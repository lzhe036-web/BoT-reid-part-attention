# encoding: utf-8
"""Observation-only evidence helpers for per-sample dynamic gating."""

from __future__ import absolute_import

import json
import math
import os
from pathlib import Path

import torch


GATING_STAT_FIELDS = (
    "gating_temperature", "gating_sample_count",
    "p2_mean", "p2_std", "p2_min", "p2_max",
    "p4_mean", "p4_std", "p4_min", "p4_max",
    "p6_mean", "p6_std", "p6_min", "p6_max",
    "applied_w2_mean", "applied_w2_std",
    "applied_w4_mean", "applied_w4_std",
    "applied_w6_mean", "applied_w6_std",
    "mean_gate_entropy", "dominant_k2_ratio", "dominant_k4_ratio",
    "dominant_k6_ratio",
)


class DynamicGatingEvidenceError(RuntimeError):
    pass


class GatingEpochAccumulator(object):
    """Exact sample-weighted moments without retaining per-sample tensors."""

    def __init__(self, temperature):
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("gating temperature must be finite and positive")
        self.temperature = temperature
        self.reset()

    def reset(self):
        self.count = 0
        self.sum_p = [0.0, 0.0, 0.0]
        self.sum_p2 = [0.0, 0.0, 0.0]
        self.min_p = [float("inf"), float("inf"), float("inf")]
        self.max_p = [float("-inf"), float("-inf"), float("-inf")]
        self.sum_w = [0.0, 0.0, 0.0]
        self.sum_w2 = [0.0, 0.0, 0.0]
        self.entropy_sum = 0.0
        self.dominant_counts = [0, 0, 0]

    def update(self, probabilities):
        if not torch.is_tensor(probabilities):
            probabilities = torch.as_tensor(probabilities)
        values = probabilities.detach().to(device="cpu", dtype=torch.float64)
        if values.dim() != 2 or values.size(1) != 3:
            raise DynamicGatingEvidenceError(
                "Gating probabilities must have shape [B,3], got {}".format(
                    tuple(values.shape)
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
                row_sums, torch.ones_like(row_sums), rtol=1e-7, atol=1e-9):
            raise DynamicGatingEvidenceError(
                "Gating probabilities must sum to one for every sample"
            )

        weights = values * 3.0
        batch_count = int(values.size(0))
        self.count += batch_count
        for index in range(3):
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
        for index in range(3):
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
        labels = ("2", "4", "6")
        for index, label in enumerate(labels):
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
        for field in GATING_STAT_FIELDS:
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
                               epoch_length, statistics):
    path = Path(output_dir) / "dynamic_gating_epoch_stats.jsonl"
    record = {
        "epoch": int(epoch),
        "global_iteration": int(global_iteration),
        "epoch_length": int(epoch_length),
    }
    record.update({field: statistics[field] for field in GATING_STAT_FIELDS})
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
