#!/usr/bin/env python
"""Export checkpoint-bound dense and hard Top-2 G2-F gate evidence.

The inherited G2 analyzer remains responsible for the ordinary strict gate
evidence.  This G2-F layer replays the same deterministic sample list to add
the pre-sparsification probabilities, actual mask, Top-2 combination and
selection-boundary tie indicator required to audit hard selection.
"""

from __future__ import absolute_import

import argparse
import csv
import json
import math
import os
import sys
import uuid
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

from data.collate_batch import val_collate_fn
from data.datasets import ImageDataset
from data.transforms import build_transforms
from modeling import build_model
from tools import analyze_g2_global_local_gating as g2_analysis
from tools.analyze_dynamic_gating import _state_dict, select_samples
from utils.experiment_recording import sha256_file
from utils.reproducibility import make_data_loader_generator, seed_worker


SCALES = (2, 4, 6)
TAU = 0.5
ANALYSIS_MANIFEST = "g2_f_top2_tau0p5_analysis_manifest.json"
PER_SAMPLE_FIELDS = (
    "stable_sample_key", "dataset_split", "pid", "camid",
    "p_dense2", "p_dense4", "p_dense6",
    "p_top2_2", "p_top2_4", "p_top2_6",
    "w2", "w4", "w6", "mask2", "mask4", "mask6",
    "top2_combination", "disabled_scale", "dominant_k",
    "selection_boundary_tie", "checkpoint_sha256",
)


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}".format(path.name, uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write(path, fields, rows, delimiter="\t"):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=delimiter,
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    _atomic_text(path, buffer.getvalue())


def _configuration(path):
    configuration = g2_analysis._load_configuration(path, expected_gating_tau=TAU)
    expected = {
        "MULTI_GRANULARITY_GATING_SPARSIFICATION": "topk",
        "MULTI_GRANULARITY_GATING_TOPK": 2,
        "MULTI_GRANULARITY_GATING_TIE_BREAK": "scale_order",
    }
    for key, value in expected.items():
        actual = getattr(configuration.MODEL, key)
        if actual != value:
            raise ValueError("G2-F analysis requires {}={!r}, got {!r}".format(
                key, value, actual
            ))
    return configuration


def _combination(mask):
    selected = [scale for index, scale in enumerate(SCALES) if bool(mask[index])]
    if len(selected) != 2:
        raise ValueError("Hard Top-2 mask must select exactly two scales")
    disabled = [scale for scale in SCALES if scale not in selected]
    return "+".join("K{}".format(scale) for scale in selected), "K{}".format(disabled[0])


def _validate(values, dense, sparse, weights, mask, tie):
    if not (torch.isfinite(values).all() and torch.isfinite(dense).all()
            and torch.isfinite(sparse).all() and torch.isfinite(weights).all()):
        raise ValueError("G2-F gate evidence contains non-finite values")
    if not torch.all(dense >= 0) or not torch.all(sparse >= 0) or not torch.all(weights >= 0):
        raise ValueError("G2-F gate evidence contains a negative probability or weight")
    if not torch.allclose(dense.sum(1), torch.ones_like(dense.sum(1)), rtol=1e-6, atol=1e-7):
        raise ValueError("Dense G2-F gate probabilities must sum to one")
    if not torch.allclose(sparse.sum(1), torch.ones_like(sparse.sum(1)), rtol=1e-6, atol=1e-7):
        raise ValueError("Top-2 G2-F gate probabilities must sum to one")
    if not torch.allclose(weights, 3.0 * sparse, rtol=1e-6, atol=1e-7):
        raise ValueError("G2-F weights must equal 3*p_top2")
    if not torch.allclose(weights.sum(1), torch.full_like(weights.sum(1), 3.0),
                          rtol=1e-6, atol=1e-7):
        raise ValueError("G2-F weights must sum to three")
    if not torch.equal(mask.sum(1), torch.full_like(mask.sum(1), 2)):
        raise ValueError("Every G2-F sample must select exactly two scales")
    if not torch.equal(sparse.masked_select(~mask), torch.zeros_like(sparse.masked_select(~mask))):
        raise ValueError("Unselected G2-F weights must be exactly zero")
    if tie.dim() != 1 or tie.size(0) != sparse.size(0):
        raise ValueError("G2-F selection-boundary tie evidence has an invalid shape")


def _replay_rows(configuration, checkpoint_path, sample_limit, device):
    selected, num_classes = select_samples(configuration, limit=sample_limit)
    if not selected:
        raise ValueError("Deterministic G2-F sample selection returned no samples")
    image_entries = [(entry[2], entry[3], entry[4]) for entry in selected]
    loader = DataLoader(
        ImageDataset(image_entries, build_transforms(configuration, is_train=False)),
        batch_size=int(configuration.TEST.IMS_PER_BATCH), shuffle=False,
        num_workers=int(configuration.DATALOADER.NUM_WORKERS),
        collate_fn=val_collate_fn, worker_init_fn=seed_worker,
        generator=make_data_loader_generator(configuration.SEED, "query"),
    )
    model_cfg = configuration.clone()
    model_cfg.defrost()
    model_cfg.MODEL.PRETRAIN_CHOICE = "none"
    model_cfg.MODEL.PRETRAIN_PATH = ""
    model_cfg.freeze()
    model = build_model(model_cfg, int(num_classes))
    model.load_state_dict(_state_dict(torch.load(str(checkpoint_path), map_location="cpu")),
                          strict=True)
    actual_device = device or (
        "cuda" if configuration.MODEL.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model.to(actual_device)
    model.eval()
    checkpoint_sha = sha256_file(checkpoint_path)
    rows, offset = [], 0
    with torch.no_grad():
        for images, _pids, _camids in loader:
            descriptor = model(images.to(actual_device))
            if descriptor.dim() != 2 or descriptor.size(1) != 2816:
                raise ValueError("G2-F inference descriptor contract changed")
            evidence = model._last_dynamic_gating
            dense = evidence["dense_probabilities"].to("cpu", dtype=torch.float64)
            sparse = evidence["probabilities"].to("cpu", dtype=torch.float64)
            weights = evidence["weights"].to("cpu", dtype=torch.float64)
            mask = evidence["selection_mask"].to("cpu", dtype=torch.bool)
            tie = evidence["selection_boundary_tie"].to("cpu", dtype=torch.bool)
            _validate(evidence["logits"].to("cpu", dtype=torch.float64), dense, sparse,
                      weights, mask, tie)
            for index in range(sparse.size(0)):
                key, split, _image, pid, camid = selected[offset + index]
                combination, disabled = _combination(mask[index].tolist())
                row = {"stable_sample_key": key, "dataset_split": split,
                       "pid": int(pid), "camid": int(camid)}
                for position, scale in enumerate(SCALES):
                    row["p_dense{}".format(scale)] = float(dense[index, position])
                    row["p_top2_{}".format(scale)] = float(sparse[index, position])
                    row["w{}".format(scale)] = float(weights[index, position])
                    row["mask{}".format(scale)] = int(mask[index, position].item())
                row.update({"top2_combination": combination, "disabled_scale": disabled,
                            "dominant_k": SCALES[int(sparse[index].argmax().item())],
                            "selection_boundary_tie": int(tie[index].item()),
                            "checkpoint_sha256": checkpoint_sha})
                rows.append(row)
            offset += sparse.size(0)
    if offset != len(selected):
        raise ValueError("G2-F replay sample count mismatch")
    return rows, checkpoint_sha


def _statistics(rows):
    fields = ("quantity", "scale", "sample_count", "mean", "std", "min", "q05",
              "median", "q95", "max", "dominant_ratio", "selected_ratio", "disabled_ratio")
    result = []
    for quantity, prefix in (("dense_probability", "p_dense"),
                             ("top2_probability", "p_top2_"),
                             ("actual_fusion_weight", "w")):
        for scale in SCALES:
            values = np.asarray([float(row[prefix + str(scale)]) for row in rows], dtype=np.float64)
            selected = np.asarray([int(row["mask{}".format(scale)]) for row in rows], dtype=np.float64)
            result.append({
                "quantity": quantity, "scale": scale, "sample_count": int(values.size),
                "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                "min": float(values.min()), "q05": float(np.quantile(values, .05)),
                "median": float(np.median(values)), "q95": float(np.quantile(values, .95)),
                "max": float(values.max()),
                "dominant_ratio": float(sum(int(row["dominant_k"]) == scale for row in rows) / len(rows)),
                "selected_ratio": float(selected.mean()), "disabled_ratio": float(1.0 - selected.mean()),
            })
    return fields, result


def _combination_statistics(rows):
    fields = ("top2_combination", "sample_count", "ratio")
    return fields, [{"top2_combination": combination,
                     "sample_count": sum(row["top2_combination"] == combination for row in rows),
                     "ratio": sum(row["top2_combination"] == combination for row in rows) / float(len(rows))}
                    for combination in ("K2+K4", "K2+K6", "K4+K6")]


def _plot_distribution(rows, output_base):
    figure, axis = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    bins = np.linspace(0.0, 3.0, 31)
    for scale in SCALES:
        axis.hist([float(row["w{}".format(scale)]) for row in rows], bins=bins,
                  histtype="step", linewidth=1.8, label="w{} = 3 p_top2{}".format(scale, scale))
    axis.set(xlim=(0.0, 3.0), xlabel="Actual fusion weight", ylabel="Sample count",
             title="G2-F hard Top-2 applied gate weights (zero retained)")
    axis.legend(); axis.grid(alpha=.25); figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(output_base) + "." + suffix, bbox_inches="tight")
    plt.close(figure)


def _plot_combinations(rows, output_base):
    names = ("K2+K4", "K2+K6", "K4+K6")
    counts = [sum(row["top2_combination"] == name for row in rows) for name in names]
    figure, axis = plt.subplots(figsize=(6.8, 4.8), dpi=180)
    axis.bar(names, counts)
    axis.set(xlabel="Actually enabled Top-2 combination", ylabel="Sample count",
             title="G2-F hard Top-2 combination frequency")
    axis.grid(axis="y", alpha=.25); figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(output_base) + "." + suffix, bbox_inches="tight")
    plt.close(figure)


def analyze(config_path, checkpoint_path, output_dir, epoch_stats_path,
            sample_limit=256, device=None):
    configuration = _configuration(config_path)
    inherited_path = g2_analysis.analyze(
        config_path, checkpoint_path, output_dir, epoch_stats_path,
        sample_limit=sample_limit, device=device, expected_gating_tau=TAU,
    )
    inherited = json.loads(Path(inherited_path).read_text(encoding="utf-8"))
    rows, checkpoint_sha = _replay_rows(configuration, checkpoint_path, sample_limit, device)
    inherited_rows_path = Path(inherited["files"]["test_gate_samples_tsv"]["path"])
    with inherited_rows_path.open("r", encoding="utf-8", newline="") as handle:
        inherited_rows = list(csv.DictReader(handle, delimiter="\t"))
    if [row["stable_sample_key"] for row in inherited_rows] != [row["stable_sample_key"] for row in rows]:
        raise ValueError("G2-F extended replay does not match inherited deterministic sample order")
    for raw, extended in zip(inherited_rows, rows):
        for scale in SCALES:
            if not math.isclose(float(raw["p{}".format(scale)]), float(extended["p_top2_{}".format(scale)]), rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError("G2-F inherited and extended Top-2 probabilities disagree")
    output_dir = Path(output_dir)
    per_sample = output_dir / "per_sample_gating.tsv"
    _write(per_sample, PER_SAMPLE_FIELDS, rows)
    sample_manifest = output_dir / "sample_manifest.tsv"
    _write(sample_manifest, ("stable_sample_key", "dataset_split", "pid", "camid", "checkpoint_sha256"), rows)
    statistic_fields, statistic_rows = _statistics(rows)
    statistics = output_dir / "gate_statistics.csv"
    _write(statistics, statistic_fields, statistic_rows, delimiter=",")
    combination_fields, combination_rows = _combination_statistics(rows)
    if not math.isclose(sum(row["ratio"] for row in combination_rows), 1.0, abs_tol=1e-12):
        raise ValueError("G2-F Top-2 combination ratios do not sum to one")
    combinations = output_dir / "top2_combination_statistics.csv"
    _write(combinations, combination_fields, combination_rows, delimiter=",")
    distribution_base = output_dir / "g2_f_top2_weight_distribution"
    combination_base = output_dir / "g2_f_top2_combination_frequency"
    _plot_distribution(rows, distribution_base)
    _plot_combinations(rows, combination_base)
    inherited.update({
        "analysis_type": "G2-F hard Top-2 sparse Dynamic Gating",
        "gate_temperature": TAU, "gate_sparsification": "topk", "gate_topk": 2,
        "gate_tie_break": "scale_order", "weight_scale": 3.0,
        "initial_zero_controller_selection": "K2+K4",
        "selection_boundary_tie_count": sum(int(row["selection_boundary_tie"]) for row in rows),
        "selection_boundary_tie_ratio": sum(int(row["selection_boundary_tie"]) for row in rows) / float(len(rows)),
        "probability_semantics": "p_dense is pre-sparsification; p_top2 is actual; w=3*p_top2",
    })
    inherited["files"].update({
        "per_sample_gating_tsv": {"path": str(per_sample), "sha256": sha256_file(per_sample)},
        "sample_manifest_tsv": {"path": str(sample_manifest), "sha256": sha256_file(sample_manifest)},
        "gate_statistics_csv": {"path": str(statistics), "sha256": sha256_file(statistics)},
        "top2_combination_statistics_csv": {"path": str(combinations), "sha256": sha256_file(combinations)},
        "g2_f_top2_weight_distribution_png": {"path": str(distribution_base) + ".png", "sha256": sha256_file(str(distribution_base) + ".png")},
        "g2_f_top2_weight_distribution_pdf": {"path": str(distribution_base) + ".pdf", "sha256": sha256_file(str(distribution_base) + ".pdf")},
        "g2_f_top2_combination_frequency_png": {"path": str(combination_base) + ".png", "sha256": sha256_file(str(combination_base) + ".png")},
        "g2_f_top2_combination_frequency_pdf": {"path": str(combination_base) + ".pdf", "sha256": sha256_file(str(combination_base) + ".pdf")},
    })
    manifest = output_dir / ANALYSIS_MANIFEST
    _atomic_text(manifest, json.dumps(inherited, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epoch-stats", required=True)
    parser.add_argument("--sample-limit", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    print(analyze(args.config_file, args.weight, args.output_dir, args.epoch_stats,
                  args.sample_limit, args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
