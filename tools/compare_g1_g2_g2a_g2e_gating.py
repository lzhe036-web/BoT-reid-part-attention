#!/usr/bin/env python
"""Strict shared-sample G1/G2/G2-A/G2-E gate comparison (no training)."""

from __future__ import absolute_import

import argparse
import csv
import json
import math
import os
import uuid
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VERSIONS = ("G1", "G2", "G2-A-tau0p5", "G2-E-alpha0p3-tau0p5")
SCALES = (2, 4, 6)


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}".format(path.name, uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write(path, fields, rows, delimiter=","):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=delimiter,
                            lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _read_samples(path, version, require_coefficients=False):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("{} samples are empty".format(version))
    required = {"stable_sample_key", "p2", "p4", "p6", "w2", "w4", "w6",
                "dominant_k", "checkpoint_sha256"}
    if require_coefficients:
        required.update({"c2", "c4", "c6", "alpha"})
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("{} samples lack {}".format(version, sorted(missing)))
    keys = []
    for row in rows:
        probabilities = [float(row["p{}".format(scale)]) for scale in SCALES]
        weights = [float(row["w{}".format(scale)]) for scale in SCALES]
        if (not all(math.isfinite(value) and value >= 0 for value in probabilities + weights)
                or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-8)
                or any(not math.isclose(weight, 3.0 * probability, rel_tol=1e-6,
                                         abs_tol=1e-8)
                       for weight, probability in zip(weights, probabilities))):
            raise ValueError("{} violates p/w scaled-softmax semantics".format(version))
        keys.append(row["stable_sample_key"])
    if len(set(keys)) != len(keys):
        raise ValueError("{} has duplicate stable sample keys".format(version))
    return rows


def _read_result(path, version):
    with Path(path).open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    required = ("rank1_percent", "rank5_percent", "rank10_percent", "map_percent")
    metrics = record.get("metrics", {})
    if any(key not in metrics or not math.isfinite(float(metrics[key])) for key in required):
        raise ValueError("{} result lacks finite formal metrics".format(version))
    selected = record.get("selected_checkpoint", {})
    if not isinstance(selected.get("sha256"), str) or len(selected["sha256"]) != 64:
        raise ValueError("{} result lacks selected checkpoint SHA256".format(version))
    return {"version": version, "selected_epoch": int(selected["epoch"]),
            "checkpoint_sha256": selected["sha256"], **{key: float(metrics[key]) for key in required}}


def _summary(version, rows):
    output = {"version": version, "sample_count": len(rows),
              "dominant_tie_rule": "torch.argmax(p2,p4,p6): K2 then K4 then K6"}
    for scale in SCALES:
        output["p{}_mean".format(scale)] = float(np.mean([float(row["p{}".format(scale)]) for row in rows]))
        output["w{}_mean".format(scale)] = float(np.mean([float(row["w{}".format(scale)]) for row in rows]))
        output["dominant_k{}_ratio".format(scale)] = float(
            sum(int(row["dominant_k"]) == scale for row in rows) / float(len(rows))
        )
    return output


def _plot(all_rows, prefix, title, output_base, xlim):
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), dpi=180, sharex=True)
    for axis, scale in zip(axes, SCALES):
        for version in VERSIONS:
            values = [float(row[prefix + str(scale)]) for row in all_rows[version]]
            axis.hist(values, bins=30, histtype="step", linewidth=1.4, label=version)
        axis.set(title="K{}".format(scale), xlabel=prefix, ylabel="sample count", xlim=xlim)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    figure.suptitle(title)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(Path(str(output_base) + "." + suffix)), bbox_inches="tight")
    plt.close(figure)


def compare(sample_paths, result_paths, output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite comparison output: {}".format(output_dir))
    all_rows = {}
    for index, version in enumerate(VERSIONS):
        all_rows[version] = _read_samples(sample_paths[index], version,
                                          require_coefficients=index == 3)
    reference_keys = [row["stable_sample_key"] for row in all_rows[VERSIONS[0]]]
    for version in VERSIONS[1:]:
        actual = [row["stable_sample_key"] for row in all_rows[version]]
        if actual != reference_keys:
            raise ValueError("{} sample keys/order differ; refusing intersection or reorder".format(version))
    metrics = [_read_result(path, version) for path, version in zip(result_paths, VERSIONS)]
    for record, version in zip(metrics, VERSIONS):
        values = all_rows[version]
        if any(row["checkpoint_sha256"] != record["checkpoint_sha256"] for row in values):
            raise ValueError("{} samples do not bind to its formal selected checkpoint".format(version))
    output_dir.mkdir(parents=True)
    summaries = [_summary(version, all_rows[version]) for version in VERSIONS]
    _write(output_dir / "gate_statistics.csv", sorted(summaries[0]), summaries)
    direct = metrics[2]
    comparison_metrics = []
    for record in metrics:
        row = dict(record)
        for field in ("rank1_percent", "rank5_percent", "rank10_percent", "map_percent"):
            row["delta_vs_g2a_{}_pp".format(field)] = float(record[field] - direct[field])
        comparison_metrics.append(row)
    metric_fields = list(comparison_metrics[0])
    _write(output_dir / "retrieval_metrics.csv", metric_fields, comparison_metrics)
    combined = []
    for version in VERSIONS:
        for row in all_rows[version]:
            combined.append({"version": version, **row})
    combined_fields = sorted({key for row in combined for key in row})
    _write(output_dir / "per_sample_gating.tsv", combined_fields, combined, delimiter="\t")
    figures = output_dir / "figures"; figures.mkdir()
    _plot(all_rows, "w", "Shared-sample residual-branch weights (w=3p)",
          figures / "four_version_weight_distributions", (0., 3.))
    e_rows = all_rows[VERSIONS[-1]]
    e_only = {VERSIONS[-1]: e_rows}
    # Keep coefficient scale separate from common w comparison.
    figure, axis = plt.subplots(figsize=(7, 4), dpi=180)
    for scale in SCALES:
        axis.hist([float(row["c{}".format(scale)]) for row in e_rows], bins=30,
                  histtype="step", linewidth=1.6, label="c{}=1+0.3w{}".format(scale, scale))
    axis.set(title="G2-E final local coefficients", xlabel="c", ylabel="sample count")
    axis.legend(); axis.grid(alpha=.2); figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(figures / ("g2_e_final_coefficients." + suffix)), bbox_inches="tight")
    plt.close(figure)
    examples = sorted((row for row in e_rows if int(row["dominant_k"]) in (4, 6)),
                      key=lambda row: (int(row["dominant_k"]), row["stable_sample_key"]))
    _write(output_dir / "sample_selection.csv",
           ["selection_rule", "dominant_k", "stable_sample_key", "p2", "p4", "p6", "w2", "w4", "w6", "c2", "c4", "c6"],
           [{"selection_rule": "stable_sample_key ascending; first six per E dominant class",
             **{key: row[key] for key in ("dominant_k", "stable_sample_key", "p2", "p4", "p6", "w2", "w4", "w6", "c2", "c4", "c6")}}
            for row in examples[:12]])
    _atomic_text(output_dir / "README.md", """# Strict G1/G2/G2-A/G2-E gate comparison

All four per-sample files must have the exact same stable-key sequence; the tool rejects differing sets or order rather than taking an intersection. Common distribution plots show `w=3p`. G2-E coefficients `c=1+0.3w` are plotted separately because they are not probabilities or common residual-branch weights. Retrieval deltas are percentage points versus direct baseline G2-A (tau_g=0.5).
""")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("g1", "g2", "g2a", "g2e"):
        parser.add_argument("--{}-samples".format(prefix), required=True)
        parser.add_argument("--{}-result".format(prefix), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    sample_paths = [getattr(args, "{}_samples".format(prefix)) for prefix in ("g1", "g2", "g2a", "g2e")]
    result_paths = [getattr(args, "{}_result".format(prefix)) for prefix in ("g1", "g2", "g2a", "g2e")]
    print(compare(sample_paths, result_paths, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
