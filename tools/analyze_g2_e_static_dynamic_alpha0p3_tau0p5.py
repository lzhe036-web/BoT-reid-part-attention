#!/usr/bin/env python
"""Export G2-E p/w/residual/c evidence from one selected checkpoint.

The inherited G2 analyzer remains the source of machine-extracted p and w
samples.  This layer adds named residual and final-coefficient evidence, so a
coefficient is never misrepresented as a probability.
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


ALPHA = 0.3
SCALES = (2, 4, 6)
ANALYSIS_MANIFEST = "g2_e_static_dynamic_alpha0p3_tau0p5_analysis_manifest.json"


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}".format(path.name, uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write_delimited(path, fields, rows, delimiter="\t"):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=delimiter,
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _configuration(path):
    configuration = g2_analysis._load_configuration(path, expected_gating_tau=0.5)
    if configuration.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL is not True:
        raise ValueError("G2-E analysis requires static-dynamic residual fusion")
    if not math.isclose(float(configuration.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA),
                        ALPHA, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("G2-E analysis requires fixed alpha=0.3")
    return configuration


def _read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("Inherited gate sample export is empty")
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
        if (not all(math.isfinite(value) and value >= 0.0
                    for value in probabilities + weights)
                or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-8)
                or any(not math.isclose(weight, 3.0 * probability, rel_tol=1e-6,
                                         abs_tol=1e-8)
                       for weight, probability in zip(weights, probabilities))):
            raise ValueError("Inherited G2-E samples violate scaled-softmax semantics")
        residual = [ALPHA * value for value in weights]
        coefficients = [1.0 + value for value in residual]
        row = {key: sample[key] for key in
               ("stable_sample_key", "dataset_split", "pid", "camid")}
        for index, scale in enumerate(SCALES):
            row["p{}".format(scale)] = probabilities[index]
            row["w{}".format(scale)] = weights[index]
            row["residual{}".format(scale)] = residual[index]
            row["c{}".format(scale)] = coefficients[index]
        row.update({"alpha": ALPHA, "dominant_k": sample["dominant_k"],
                    "checkpoint_sha256": checkpoint_sha})
        rows.append(row)
    return fields, rows


def _statistics(rows):
    fields = ("quantity", "scale", "sample_count", "mean", "std", "min",
              "q05", "median", "q95", "max", "dominant_ratio")
    output = []
    for quantity, prefix in (("probability", "p"), ("dynamic_weight", "w"),
                             ("residual_alpha_times_w", "residual"),
                             ("final_local_coefficient", "c")):
        for scale in SCALES:
            values = np.asarray([float(row[prefix + str(scale)]) for row in rows],
                                dtype=np.float64)
            output.append({
                "quantity": quantity, "scale": scale, "sample_count": int(values.size),
                "mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)), "q05": float(np.quantile(values, .05)),
                "median": float(np.median(values)), "q95": float(np.quantile(values, .95)),
                "max": float(np.max(values)),
                "dominant_ratio": float(sum(int(row["dominant_k"]) == scale for row in rows)
                                        / float(len(rows))),
            })
    return fields, output


def _plot(rows, prefix, title, ylabel, output_base, xlim=None):
    figure, axis = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    for scale in SCALES:
        values = [float(row[prefix + str(scale)]) for row in rows]
        axis.hist(values, bins=30, density=True, histtype="step", linewidth=1.8,
                  label="{}{}".format(prefix, scale))
    axis.set_title(title)
    axis.set_xlabel("Value")
    axis.set_ylabel(ylabel)
    if xlim is not None:
        axis.set_xlim(*xlim)
    axis.grid(alpha=.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(Path(str(output_base) + "." + suffix)), bbox_inches="tight")
    plt.close(figure)


def analyze(config_path, checkpoint_path, output_dir, epoch_stats_path,
            sample_limit=256, device=None):
    configuration = _configuration(config_path)
    inherited_manifest_path = g2_analysis.analyze(
        config_path, checkpoint_path, output_dir, epoch_stats_path,
        sample_limit=sample_limit, device=device, expected_gating_tau=0.5,
    )
    inherited = json.loads(Path(inherited_manifest_path).read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    samples_path = Path(inherited["files"]["test_gate_samples_tsv"]["path"])
    fields, rows = _coefficient_rows(_read_rows(samples_path), checkpoint_sha)
    per_sample_path = Path(output_dir) / "per_sample_gating.tsv"
    _write_delimited(per_sample_path, fields, rows)
    selection_path = Path(output_dir) / "sample_manifest.tsv"
    _write_delimited(selection_path,
                     ("stable_sample_key", "dataset_split", "pid", "camid",
                      "checkpoint_sha256"),
                     [{key: row[key] for key in
                       ("stable_sample_key", "dataset_split", "pid", "camid",
                        "checkpoint_sha256")} for row in rows])
    stats_fields, stats_rows = _statistics(rows)
    statistics_path = Path(output_dir) / "gate_statistics.csv"
    _write_delimited(statistics_path, stats_fields, stats_rows, delimiter=",")
    probability_base = Path(output_dir) / "g2_e_probability_distribution"
    coefficient_base = Path(output_dir) / "g2_e_final_coefficient_distribution"
    _plot(rows, "p", "G2-E gate probability distribution (p)", "Density",
          probability_base, (0.0, 1.0))
    _plot(rows, "c", "G2-E final local coefficient (c=1+0.3w)", "Density",
          coefficient_base)
    inherited.update({
        "analysis_type": "G2-E static concatenation plus gated residual observation",
        "fusion_mode": "static_concat_plus_gated_residual",
        "static_dynamic_alpha": ALPHA,
        "gate_temperature": float(configuration.MODEL.MULTI_GRANULARITY_GATING_TAU),
        "formula": "concat(g,(1+0.3*w2)z2,(1+0.3*w4)z4,(1+0.3*w6)z6)",
        "coefficient_semantics": (
            "p is probability; w=3p is the dynamic residual branch weight; "
            "residual=0.3*w; c=1+0.3*w is not a probability"
        ),
    })
    inherited["files"].update({
        "per_sample_gating_tsv": {"path": str(per_sample_path),
                                  "sha256": sha256_file(per_sample_path)},
        "sample_manifest_tsv": {"path": str(selection_path),
                                "sha256": sha256_file(selection_path)},
        "gate_statistics_csv": {"path": str(statistics_path),
                                "sha256": sha256_file(statistics_path)},
        "g2_e_probability_distribution_png": {"path": str(probability_base) + ".png",
                                               "sha256": sha256_file(str(probability_base) + ".png")},
        "g2_e_probability_distribution_pdf": {"path": str(probability_base) + ".pdf",
                                               "sha256": sha256_file(str(probability_base) + ".pdf")},
        "g2_e_final_coefficient_distribution_png": {"path": str(coefficient_base) + ".png",
                                                     "sha256": sha256_file(str(coefficient_base) + ".png")},
        "g2_e_final_coefficient_distribution_pdf": {"path": str(coefficient_base) + ".pdf",
                                                     "sha256": sha256_file(str(coefficient_base) + ".pdf")},
    })
    manifest_path = Path(output_dir) / ANALYSIS_MANIFEST
    _atomic_text(manifest_path, json.dumps(inherited, ensure_ascii=False, indent=2,
                                            sort_keys=True) + "\n")
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
