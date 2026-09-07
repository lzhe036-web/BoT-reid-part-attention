#!/usr/bin/env python
"""Export observation-only evidence for the G2 global-plus-local gate.

G2 takes the controller input ``[g, z2, z4, z6]`` and produces three
scaled-softmax weights for the existing local descriptors ``z2``, ``z4`` and
``z6``.  There is no fourth gate weight for ``g``.  Consequently this tool
exports (1) the actual per-sample K=2/K=4/K=6 gate outputs and (2) the
controller parameter-block magnitudes for the four input blocks.  The latter
is an input-sensitivity proxy, not a per-sample branch weight.
"""

from __future__ import absolute_import

import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from tools.analyze_dynamic_gating import _state_dict, generate_dynamic_gating_evidence
from utils.dynamic_gating_evidence import read_gating_epoch_records
from utils.experiment_recording import sha256_file


SCALES = (2, 4, 6)


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}".format(path.name, uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write_csv(path, fieldnames, rows):
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _gate_input_description(gating_input):
    descriptions = {
        "concat_global_local": "concat([g, z2, z4, z6])",
        "concat_global_local_diff46": "concat([g, z2, z4, z6, abs(z4-z6)])",
    }
    try:
        return descriptions[gating_input]
    except KeyError:
        raise ValueError("Unsupported G2 controller-input mode: {!r}".format(gating_input))


def _controller_blocks(configuration):
    local_dim = int(configuration.MODEL.MULTI_GRANULARITY_PART_DIM)
    gating_input = str(configuration.MODEL.MULTI_GRANULARITY_GATING_INPUT)
    blocks = [("g", 2048), ("z2", local_dim), ("z4", local_dim), ("z6", local_dim)]
    if gating_input == "concat_global_local_diff46":
        blocks.append(("delta46_abs_z4_minus_z6", local_dim))
    elif gating_input != "concat_global_local":
        raise ValueError("Unsupported G2 controller-input mode: {!r}".format(gating_input))
    return tuple(blocks)


def _load_configuration(config_path, expected_gating_tau=None,
                        expected_gating_input=None):
    configuration = cfg.clone()
    configuration.merge_from_file(str(config_path))
    configuration.freeze()
    if not configuration.MODEL.MULTI_GRANULARITY_DYNAMIC_GATING:
        raise ValueError("The supplied config does not enable Dynamic Gating")
    gating_input = str(configuration.MODEL.MULTI_GRANULARITY_GATING_INPUT)
    if gating_input not in ("concat_global_local", "concat_global_local_diff46"):
        raise ValueError(
            "This G2 analyzer does not support MULTI_GRANULARITY_GATING_INPUT="
            "{!r}".format(gating_input)
        )
    if expected_gating_input is not None and gating_input != expected_gating_input:
        raise ValueError("This analyzer requires gating input {!r}, got {!r}".format(
            expected_gating_input, gating_input
        ))
    if expected_gating_tau is not None and float(
            configuration.MODEL.MULTI_GRANULARITY_GATING_TAU
    ) != float(expected_gating_tau):
        raise ValueError(
            "This G2 analyzer requires MULTI_GRANULARITY_GATING_TAU={!r}, "
            "got {!r}".format(
                float(expected_gating_tau),
                float(configuration.MODEL.MULTI_GRANULARITY_GATING_TAU),
            )
        )
    return configuration


def _block_rows(state, configuration, checkpoint_sha256):
    key = "multi_granularity_dynamic_gate.controller.weight"
    if key not in state:
        raise ValueError("Checkpoint has no dynamic-gate controller weight")
    controller = state[key].detach().to(dtype=torch.float64, device="cpu")
    blocks = _controller_blocks(configuration)
    expected = sum(width for _name, width in blocks)
    if controller.dim() != 2 or controller.size(0) != len(SCALES):
        raise ValueError("Unexpected G2 controller shape {}".format(tuple(controller.shape)))
    if controller.size(1) != expected:
        raise ValueError("Unexpected G2 controller input width {} != {}".format(
            controller.size(1), expected
        ))
    boundaries, offset = [], 0
    for block, width in blocks:
        boundaries.append((block, offset, offset + width))
        offset += width
    rows = []
    for target_index, target_scale in enumerate(SCALES):
        for block, start, end in boundaries:
            values = controller[target_index, start:end]
            l2_norm = float(torch.linalg.vector_norm(values).item())
            width = int(end - start)
            rows.append({
                "checkpoint_sha256": checkpoint_sha256,
                "target_gate": "w{}".format(target_scale),
                "input_block": block,
                "input_width": width,
                "l2_norm": l2_norm,
                "rms_weight": float(torch.sqrt(torch.mean(values.square())).item()),
                "mean_abs_weight": float(torch.mean(torch.abs(values)).item()),
            })
    return rows, tuple(boundaries)


def _plot_block_magnitudes(rows, output_path):
    labels = tuple(dict.fromkeys(row["input_block"] for row in rows))
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.22
    figure, axis = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    for index, scale in enumerate(SCALES):
        values = [
            next(
                row["rms_weight"] for row in rows
                if row["target_gate"] == "w{}".format(scale)
                and row["input_block"] == label
            )
            for label in labels
        ]
        axis.bar(positions + (index - 1) * width, values, width,
                 label="controller output w{}".format(scale))
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("G2 controller input block")
    axis.set_ylabel("Controller coefficient RMS magnitude")
    axis.set_title("G2 controller parameter-block magnitudes")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(str(output_path), bbox_inches="tight")
    figure.savefig(str(output_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(figure)


def _history_rows(records):
    fields = [
        "epoch", "gating_sample_count", "mean_gate_entropy",
        "p2_mean", "p4_mean", "p6_mean",
        "applied_w2_mean", "applied_w4_mean", "applied_w6_mean",
        "dominant_k2_ratio", "dominant_k4_ratio", "dominant_k6_ratio",
    ]
    return [{field: record.get(field, "not_recorded") for field in fields}
            for record in records], fields


def _plot_history(rows, output_path):
    if not rows:
        return False
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), dpi=180)
    for scale in SCALES:
        axes[0].plot(epochs, [float(row["p{}_mean".format(scale)]) for row in rows],
                     label="p{} mean".format(scale))
        axes[1].plot(epochs, [float(row["applied_w{}_mean".format(scale)]) for row in rows],
                     label="w{} mean".format(scale))
    axes[0].set_title("Training gate probability means")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean probability")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_title("Training applied gate-weight means")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean applied weight")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(str(output_path), bbox_inches="tight")
    figure.savefig(str(output_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(figure)
    return True


def _sample_weight_rows(samples_path):
    with Path(samples_path).open("r", encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle, delimiter="\t"))
    if not samples:
        raise ValueError("No deterministic G2 gating samples were exported")
    rows = []
    series = []
    for scale in SCALES:
        values = np.asarray([float(row["w{}".format(scale)]) for row in samples])
        series.append(values)
        rows.append({
            "gate_weight": "w{}".format(scale),
            "sample_count": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    return rows, series


def _plot_sample_weight_distribution(series, output_path):
    figure, axis = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    axis.boxplot(series, labels=["w2", "w4", "w6"], showmeans=True)
    axis.set_xlabel("Applied local-scale gate weight")
    axis.set_ylabel("Weight across deterministic test samples")
    axis.set_title("G2 test-stage applied gate-weight distribution")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(str(output_path), bbox_inches="tight")
    figure.savefig(str(output_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(figure)


def _sha256_text(path):
    return sha256_file(Path(path)) if Path(path).is_file() else "not_created"


def analyze(config_path, checkpoint_path, output_dir, epoch_stats_path,
            sample_limit=256, device=None, expected_gating_tau=None,
            expected_gating_input=None):
    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    epoch_stats_path = Path(epoch_stats_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    if not epoch_stats_path.is_file():
        raise FileNotFoundError("Epoch gating statistics not found: {}".format(epoch_stats_path))
    output_dir.mkdir(parents=True, exist_ok=False)
    configuration = _load_configuration(
        config_path, expected_gating_tau, expected_gating_input
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = _state_dict(checkpoint)
    controller_parameter_count = sum(
        int(value.numel()) for name, value in state.items()
        if name.startswith("multi_granularity_dynamic_gate.controller.")
    )
    if controller_parameter_count <= 0:
        raise ValueError("Checkpoint has no dynamic-gate controller parameters")
    block_rows, _boundaries = _block_rows(state, configuration, checkpoint_sha)
    block_csv = output_dir / "g2_controller_input_block_norms.csv"
    _write_csv(block_csv, list(block_rows[0].keys()), block_rows)
    block_png = output_dir / "g2_controller_input_block_norms.png"
    _plot_block_magnitudes(block_rows, block_png)

    epoch_records = read_gating_epoch_records(epoch_stats_path)
    if not epoch_records:
        raise ValueError("No Dynamic Gating epoch records found")
    history_rows, history_fields = _history_rows(epoch_records)
    history_csv = output_dir / "g2_gate_training_history.csv"
    _write_csv(history_csv, history_fields, history_rows)
    history_png = output_dir / "g2_gate_training_history.png"
    _plot_history(history_rows, history_png)

    summary_path, samples_path, evidence_summary = generate_dynamic_gating_evidence(
        configuration, checkpoint_path, output_dir,
        epoch_records[-1], limit=sample_limit, device=device,
    )
    weight_rows, series = _sample_weight_rows(samples_path)
    weights_csv = output_dir / "g2_gate_test_weight_summary.csv"
    _write_csv(weights_csv, list(weight_rows[0].keys()), weight_rows)
    weights_png = output_dir / "g2_gate_test_weight_distribution.png"
    _plot_sample_weight_distribution(series, weights_png)

    manifest = {
        "analysis_type": "G2 linear Dynamic Gating observation",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "epoch_statistics_path": str(epoch_stats_path),
        "epoch_statistics_sha256": sha256_file(epoch_stats_path),
        "gating_input": _gate_input_description(str(
            configuration.MODEL.MULTI_GRANULARITY_GATING_INPUT
        )),
        "gating_input_mode": str(configuration.MODEL.MULTI_GRANULARITY_GATING_INPUT),
        "controller_input_blocks": [name for name, _width in _controller_blocks(configuration)],
        "controller_input_dim": sum(width for _name, width in _controller_blocks(configuration)),
        "controller_parameter_count": controller_parameter_count,
        "retrieval_feature_dim": 2816,
        "delta46_definition": (
            "torch.abs(z4-z6); unweighted; graph-connected; controller input only"
            if str(configuration.MODEL.MULTI_GRANULARITY_GATING_INPUT) == "concat_global_local_diff46"
            else "not_applicable"
        ),
        "gating_temperature": float(
            configuration.MODEL.MULTI_GRANULARITY_GATING_TAU
        ),
        "controller_output_semantics": (
            "three scaled-softmax weights applied to z2, z4, z6; no direct g weight"
        ),
        "controller_block_plot_semantics": (
            "RMS magnitude of learned linear-controller coefficients by input block; "
            "not a per-sample branch-weight allocation"
        ),
        "test_weight_protocol": evidence_summary["selection_rule"],
        "test_weight_sample_count": evidence_summary["selected_sample_count"],
        "files": {
            "controller_block_norms_csv": {"path": str(block_csv), "sha256": _sha256_text(block_csv)},
            "controller_block_norms_png": {"path": str(block_png), "sha256": _sha256_text(block_png)},
            "controller_block_norms_pdf": {"path": str(block_png.with_suffix(".pdf")), "sha256": _sha256_text(block_png.with_suffix(".pdf"))},
            "training_history_csv": {"path": str(history_csv), "sha256": _sha256_text(history_csv)},
            "training_history_png": {"path": str(history_png), "sha256": _sha256_text(history_png)},
            "training_history_pdf": {"path": str(history_png.with_suffix(".pdf")), "sha256": _sha256_text(history_png.with_suffix(".pdf"))},
            "test_gate_samples_tsv": {"path": str(samples_path), "sha256": _sha256_text(samples_path)},
            "test_weight_summary_csv": {"path": str(weights_csv), "sha256": _sha256_text(weights_csv)},
            "test_weight_distribution_png": {"path": str(weights_png), "sha256": _sha256_text(weights_png)},
            "test_weight_distribution_pdf": {"path": str(weights_png.with_suffix(".pdf")), "sha256": _sha256_text(weights_png.with_suffix(".pdf"))},
            "dynamic_gating_summary_json": {"path": str(summary_path), "sha256": _sha256_text(summary_path)},
        },
    }
    manifest_path = output_dir / "g2_gating_analysis_manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epoch-stats", required=True)
    parser.add_argument("--sample-limit", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--expected-gating-tau", type=float, default=None)
    parser.add_argument("--expected-gating-input", default=None)
    args = parser.parse_args(argv)
    if args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    manifest = analyze(
        args.config_file, args.weight, args.output_dir, args.epoch_stats,
        sample_limit=args.sample_limit, device=args.device,
        expected_gating_tau=args.expected_gating_tau,
        expected_gating_input=args.expected_gating_input,
    )
    print(str(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
