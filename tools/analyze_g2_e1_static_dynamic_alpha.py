#!/usr/bin/env python
"""Export E1 static-concatenation plus gated-residual checkpoint evidence.

The inherited G2 exporter remains the authoritative source of p/w values.  E1
adds separately named, reproducible coefficient artifacts so that c=1+alpha*w
is never confused with a probability or with the original dynamic weight.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_g2_global_local_gating as g2_analysis
from utils.experiment_recording import sha256_file


ALPHA = 0.5
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


def _write_csv(path, fields, rows):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _configuration(path):
    configuration = g2_analysis._load_configuration(path)
    if configuration.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL is not True:
        raise ValueError("E1 analysis requires static-dynamic residual fusion")
    if float(configuration.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA) != ALPHA:
        raise ValueError("E1 analysis requires fixed alpha=0.5")
    if float(configuration.MODEL.MULTI_GRANULARITY_GATING_TAU) != 1.0:
        raise ValueError("E1 analysis requires gate temperature tau_g=1.0")
    return configuration


def _read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("E1 inherited gate sample export is empty")
    return rows


def _coefficient_rows(samples, checkpoint_sha):
    fields = (
        "stable_sample_key", "dataset_split", "pid", "camid", "p2", "p4", "p6",
        "w2", "w4", "w6", "residual2", "residual4", "residual6", "c2", "c4", "c6",
        "alpha", "dominant_k", "checkpoint_sha256",
    )
    rows = []
    for sample in samples:
        probabilities = [float(sample["p{}".format(scale)]) for scale in SCALES]
        weights = [float(sample["w{}".format(scale)]) for scale in SCALES]
        if (not all(math.isfinite(value) and value >= 0.0 for value in probabilities + weights)
                or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-8)
                or any(not math.isclose(weight, 3.0 * probability, rel_tol=1e-6, abs_tol=1e-8)
                       for weight, probability in zip(weights, probabilities))):
            raise ValueError("Inherited E1 p/w samples violate scaled-softmax semantics")
        residual = [ALPHA * value for value in weights]
        coefficients = [1.0 + value for value in residual]
        row = {key: sample[key] for key in ("stable_sample_key", "dataset_split", "pid", "camid")}
        for index, scale in enumerate(SCALES):
            row["p{}".format(scale)] = probabilities[index]
            row["w{}".format(scale)] = weights[index]
            row["residual{}".format(scale)] = residual[index]
            row["c{}".format(scale)] = coefficients[index]
        row.update({"alpha": ALPHA, "dominant_k": sample["dominant_k"],
                    "checkpoint_sha256": checkpoint_sha})
        rows.append(row)
    return fields, rows


def _summary(rows):
    fields = ("quantity", "scale", "sample_count", "mean", "std", "min", "q05", "median", "q95", "max")
    result = []
    for quantity, prefix in (("probability", "p"), ("dynamic_weight", "w"),
                             ("residual_alpha_times_w", "residual"),
                             ("final_local_coefficient", "c")):
        for scale in SCALES:
            values = np.asarray([float(row[prefix + str(scale)]) for row in rows], dtype=np.float64)
            result.append({"quantity": quantity, "scale": scale, "sample_count": int(values.size),
                           "mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
                           "min": float(np.min(values)), "q05": float(np.quantile(values, .05)),
                           "median": float(np.median(values)), "q95": float(np.quantile(values, .95)),
                           "max": float(np.max(values))})
    return fields, result


def _plot(rows, prefix, title, ylabel, output_base, xlim=None):
    figure, axis = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    for scale in SCALES:
        values = [float(row[prefix + str(scale)]) for row in rows]
        axis.hist(values, bins=30, density=True, histtype="step", linewidth=1.8,
                  label="{}{}".format(prefix, scale))
    axis.set_title(title); axis.set_xlabel("Value"); axis.set_ylabel(ylabel)
    if xlim is not None:
        axis.set_xlim(*xlim)
    axis.grid(alpha=.25); axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(Path(str(output_base) + "." + suffix)), bbox_inches="tight")
    plt.close(figure)


def analyze(config_path, checkpoint_path, output_dir, epoch_stats_path,
            sample_limit=256, device=None):
    configuration = _configuration(config_path)
    inherited_manifest_path = g2_analysis.analyze(
        config_path, checkpoint_path, output_dir, epoch_stats_path,
        sample_limit=sample_limit, device=device,
    )
    inherited = json.loads(Path(inherited_manifest_path).read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    samples_path = Path(inherited["files"]["test_gate_samples_tsv"]["path"])
    fields, rows = _coefficient_rows(_read_rows(samples_path), checkpoint_sha)
    coefficients_path = Path(output_dir) / "e1_per_sample_gating.csv"
    _write_csv(coefficients_path, fields, rows)
    selection_path = Path(output_dir) / "sample_selection.csv"
    _write_csv(selection_path, ("stable_sample_key", "dataset_split", "pid", "camid", "checkpoint_sha256"), [
        {key: row[key] for key in ("stable_sample_key", "dataset_split", "pid", "camid", "checkpoint_sha256")}
        for row in rows
    ])
    summary_fields, summary_rows = _summary(rows)
    summary_path = Path(output_dir) / "e1_gate_coefficient_summary.csv"
    _write_csv(summary_path, summary_fields, summary_rows)
    probability_base = Path(output_dir) / "e1_probability_distribution"
    coefficient_base = Path(output_dir) / "e1_final_coefficient_distribution"
    _plot(rows, "p", "E1 test-stage gate probability distribution (p)", "Density", probability_base, (0.0, 1.0))
    _plot(rows, "c", "E1 final local coefficient distribution (c=1+0.5w)", "Density", coefficient_base)
    inherited.update({
        "analysis_type": "E1 static concatenation plus gated residual observation",
        "fusion_mode": "static_concat_plus_gated_residual",
        "static_dynamic_alpha": ALPHA,
        "gate_temperature": float(configuration.MODEL.MULTI_GRANULARITY_GATING_TAU),
        "formula": "concat(g, (1+alpha*w2)z2, (1+alpha*w4)z4, (1+alpha*w6)z6)",
        "coefficient_semantics": "p is probability, w=3p, residual=alpha*w, c=1+alpha*w; c is not a probability",
    })
    inherited["files"].update({
        "e1_per_sample_gating_csv": {"path": str(coefficients_path), "sha256": sha256_file(coefficients_path)},
        "sample_selection_csv": {"path": str(selection_path), "sha256": sha256_file(selection_path)},
        "e1_gate_coefficient_summary_csv": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "e1_probability_distribution_png": {"path": str(probability_base) + ".png", "sha256": sha256_file(str(probability_base) + ".png")},
        "e1_probability_distribution_pdf": {"path": str(probability_base) + ".pdf", "sha256": sha256_file(str(probability_base) + ".pdf")},
        "e1_final_coefficient_distribution_png": {"path": str(coefficient_base) + ".png", "sha256": sha256_file(str(coefficient_base) + ".png")},
        "e1_final_coefficient_distribution_pdf": {"path": str(coefficient_base) + ".pdf", "sha256": sha256_file(str(coefficient_base) + ".pdf")},
    })
    manifest_path = Path(output_dir) / "e1_static_dynamic_analysis_manifest.json"
    _atomic_text(manifest_path, json.dumps(inherited, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    Path(inherited_manifest_path).unlink()
    return manifest_path


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
