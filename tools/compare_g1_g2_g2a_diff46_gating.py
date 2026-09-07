#!/usr/bin/env python
"""Make a fail-closed G1/G2/G2-A/G2-D1 fixed-sample gate comparison."""

from __future__ import absolute_import

import argparse
import csv
import json
import math
import os
import shutil
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

from tools.compare_g1_g2_tau0p5_gating import (
    ComparisonError, FIXED_FIELDS, SCALES, _extract_tau_gates, _load_candidate_config,
    _market_root, _read_fixed_candidates, _read_fixed_gates, _resolve_image,
    _write_csv, _atomic_text,
)


LABELS = ("G1", "G2 tau=1.0", "G2-A tau=0.5", "G2-D1 tau=0.5")
PREFIXES = ("g1", "g2", "g2a", "diff46")


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _statistics(label, rows):
    values = np.asarray([[float(row["p{}".format(scale)]) for scale in SCALES] for row in rows])
    record = {"version": label, "sample_count": len(rows),
              "dominant_tie_rule": "argmax p2,p4,p6; exact ties select K2 then K4 then K6"}
    for index, scale in enumerate(SCALES):
        record["p{}_mean".format(scale)] = float(values[:, index].mean())
        record["dominant_k{}_ratio".format(scale)] = float(sum(int(row["dominant_k"]) == scale for row in rows)) / len(rows)
    return record


def _metrics(label, path, direct_baseline=None):
    manifest = _read_json(path)
    metrics = manifest.get("metrics", {})
    row = {"version": label, "run_id": manifest.get("run_id", "not_recorded"),
           "checkpoint_sha256": manifest.get("selected_checkpoint", {}).get("sha256", "not_recorded"),
           "selected_epoch": metrics.get("selected_epoch", "not_recorded")}
    for source, target in (("rank1_percent", "Rank-1"), ("rank5_percent", "Rank-5"),
                           ("rank10_percent", "Rank-10"), ("map_percent", "mAP")):
        value = metrics.get(source)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ComparisonError("{} run manifest lacks numeric {}".format(label, source))
        row[target] = float(value)
    row["delta_Rank-1_vs_G2-A"] = "not_applicable" if direct_baseline is None else row["Rank-1"] - direct_baseline["Rank-1"]
    row["delta_mAP_vs_G2-A"] = "not_applicable" if direct_baseline is None else row["mAP"] - direct_baseline["mAP"]
    return row


def _plot(version_rows, figures):
    colors = {"G1": "#4c78a8", "G2 tau=1.0": "#f58518", "G2-A tau=0.5": "#54a24b", "G2-D1 tau=0.5": "#e45756"}
    for kind, xmax, suffix in (("p", 1.0, "probability"), ("w", 3.0, "weight")):
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=180, sharey=True)
        for axis, scale in zip(axes, SCALES):
            for label, rows in version_rows.items():
                axis.hist([float(row["{}{}".format(kind, scale)]) for row in rows],
                          bins=np.linspace(0, xmax, 31), density=True, histtype="step",
                          linewidth=1.7, color=colors[label], label=label)
            axis.set(title="K{} {}".format(scale, kind), xlabel=("p{}" if kind == "p" else "w{}=3p{}").format(scale, scale), xlim=(0, xmax)); axis.grid(alpha=.25)
        axes[0].set_ylabel("density"); axes[-1].legend(fontsize=7)
        figure.suptitle("Exact frozen-sample Dynamic Gating {} distributions".format(suffix))
        figure.tight_layout()
        for ext in ("png", "pdf"): figure.savefig(str(figures / "gating_{}_distributions.{}".format(suffix, ext)), bbox_inches="tight")
        plt.close(figure)


def _examples(candidates, diff_map, market_root, figures, scale):
    selected = [row for row in candidates if int(diff_map[row["stable_sample_key"]]["dominant_k"]) == scale]
    selected = sorted(selected, key=lambda row: (row["selection_hash"], row["stable_sample_key"]))[:6]
    rows = []
    figure, axes = plt.subplots(1, max(1, len(selected)), figsize=(2.5 * max(1, len(selected)), 4.4), dpi=180)
    axes = np.atleast_1d(axes)
    for axis in axes: axis.axis("off")
    for axis, source in zip(axes, selected):
        gate = diff_map[source["stable_sample_key"]]
        axis.imshow(plt.imread(str(_resolve_image(market_root, source))))
        axis.set_title("{}\np=({:.3f},{:.3f},{:.3f})".format(Path(source["relative_path"]).name, float(gate["p2"]), float(gate["p4"]), float(gate["p6"])), fontsize=7)
        rows.append({"selection_rule": "selection_hash ascending; first six G2-D1 dominant class", "dominant_k": gate["dominant_k"], "stable_sample_key": source["stable_sample_key"], "p2": gate["p2"], "p4": gate["p4"], "p6": gate["p6"]})
    figure.suptitle("G2-D1 K{}-dominant examples".format(scale)); figure.tight_layout()
    for ext in ("png", "pdf"): figure.savefig(str(figures / "diff46_k{}_dominant_examples.{}".format(scale, ext)), bbox_inches="tight")
    plt.close(figure)
    return rows


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists(): raise ComparisonError("Refusing to reuse output: {}".format(output))
    candidates = _read_fixed_candidates(args.candidate_manifest)
    keys = [row["stable_sample_key"] for row in candidates]; expected = set(keys)
    gates = {
        "G1": _read_fixed_gates(args.g1_fixed_gates, "G1"),
        "G2 tau=1.0": _read_fixed_gates(args.g2_fixed_gates, "G2"),
        "G2-A tau=0.5": _read_fixed_gates(args.g2a_fixed_gates, "G2-A"),
    }
    if any(set(rows) != expected for rows in gates.values()):
        raise ComparisonError("G1/G2/G2-A samples do not exactly match the frozen candidate manifest")
    output.mkdir(parents=True); figures = output / "figures"; figures.mkdir()
    configuration = _load_candidate_config(args.config_file, "concat_global_local_diff46")
    diff_rows, checkpoint_sha = _extract_tau_gates(configuration, args.checkpoint, candidates,
                                                   _market_root(args.dataset_root),
                                                   output / "diff46_fixed_gating_samples.tsv", args.device)
    gates["G2-D1 tau=0.5"] = {row["stable_sample_key"]: row for row in diff_rows}
    if list(gates["G2-D1 tau=0.5"]) != keys:
        raise ComparisonError("G2-D1 extracted samples do not preserve exact frozen order")
    ordered = {label: [rows[key] for key in keys] for label, rows in gates.items()}
    stat_rows = [_statistics(label, ordered[label]) for label in LABELS]
    _write_csv(output / "gate_statistics.csv", list(stat_rows[0]), stat_rows)
    fields = ["stable_sample_key", "relative_path", "split", "pid", "camid"]
    for prefix in PREFIXES: fields += [prefix + "_p2", prefix + "_p4", prefix + "_p6", prefix + "_w2", prefix + "_w4", prefix + "_w6", prefix + "_dominant_k"]
    per_rows = []
    for candidate in candidates:
        row = {field: candidate[field] for field in fields[:5]}
        for label, prefix in zip(LABELS, PREFIXES):
            source = gates[label][candidate["stable_sample_key"]]
            for scale in SCALES:
                row[prefix + "_p{}".format(scale)] = source["p{}".format(scale)]
                row[prefix + "_w{}".format(scale)] = source["w{}".format(scale)]
            row[prefix + "_dominant_k"] = source["dominant_k"]
        per_rows.append(row)
    _write_csv(output / "per_sample_gating.csv", fields, per_rows)
    g1 = _metrics("G1", args.g1_run_manifest); g2 = _metrics("G2 tau=1.0", args.g2_run_manifest)
    g2a = _metrics("G2-A tau=0.5", args.g2a_run_manifest)
    diff = _metrics("G2-D1 tau=0.5", args.diff46_run_manifest, g2a)
    _write_csv(output / "retrieval_metrics.csv", list(g1), [g1, g2, g2a, diff])
    _plot(ordered, figures)
    selection = _examples(candidates, gates["G2-D1 tau=0.5"], _market_root(args.dataset_root), figures, 4)
    selection += _examples(candidates, gates["G2-D1 tau=0.5"], _market_root(args.dataset_root), figures, 6)
    _write_csv(output / "sample_selection.csv", ["selection_rule", "dominant_k", "stable_sample_key", "p2", "p4", "p6"], selection)
    shutil.copyfile(str(Path(args.resolved_config).resolve()), str(output / "config_resolved.yml"))
    manifest = {"analysis": "G1/G2/G2-A/G2-D1 exact fixed-sample comparison", "dataset": "Market1501", "seed": 42,
                "candidate_manifest": str(Path(args.candidate_manifest).resolve()), "sample_count": len(candidates),
                "sample_order_sha256": __import__("hashlib").sha256("\n".join(keys).encode()).hexdigest(),
                "diff46_checkpoint": {"path": str(Path(args.checkpoint).resolve()), "sha256": checkpoint_sha},
                "input_change_vs_direct_baseline": "[g,z2,z4,z6] -> [g,z2,z4,z6,abs(z4-z6)]", "tau_g": 0.5}
    _atomic_text(output / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_text(output / "README.md", "# G1/G2/G2-A/G2-D1 gate comparison\n\nAll four versions use exactly the same frozen ordered Market1501 images; this tool rejects a key or order mismatch and never silently intersects samples. Histograms use identical bins. The plotted applied weights are `w=3p`; the full retrieval metrics come from each formal selected-checkpoint run manifest. G2-D1 must be interpreted primarily against direct baseline G2-A, since both use `tau_g=0.5`.\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-manifest", "g1-fixed-gates", "g2-fixed-gates", "g2a-fixed-gates", "g1-run-manifest", "g2-run-manifest", "g2a-run-manifest", "diff46-run-manifest", "config-file", "resolved-config", "checkpoint", "dataset-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default=None); args = parser.parse_args(argv)
    print(str(run(args))); return 0


if __name__ == "__main__":
    raise SystemExit(main())
