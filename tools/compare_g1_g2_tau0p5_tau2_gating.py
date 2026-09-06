#!/usr/bin/env python
"""Create the required four-version G1/G2 gate comparison after τg=2.0 training.

This post-training tool consumes formal evidence only.  It never trains or
rewrites a run.  The candidate list and the G1/G2/τg=0.5 values are frozen;
only the actual τg=2.0 checkpoint is inferred on those same images.
"""

from __future__ import absolute_import

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.compare_g1_g2_tau0p5_gating import (
    ComparisonError, SCALES, _atomic_text, _extract_tau_gates,
    _market_root, _metrics, _plot_distributions, _read_fixed_candidates,
    _read_fixed_gates, _read_json, _resolve_image, _statistics, _write_csv,
)
from utils.experiment_recording import sha256_file


LABELS = ("G1", "G2 tau=1.0", "G2 tau=0.5", "G2 tau=2.0")
PREFIXES = {
    "G1": "g1", "G2 tau=1.0": "g2", "G2 tau=0.5": "tau0p5",
    "G2 tau=2.0": "tau2",
}


def _validated_tau2_config(path):
    from config import cfg
    candidate = cfg.clone()
    candidate.merge_from_file(str(path))
    candidate.freeze()
    if (str(candidate.DATASETS.NAMES).lower() != "market1501"
            or int(candidate.SEED) != 42
            or float(candidate.MODEL.MULTI_GRANULARITY_GATING_TAU) != 2.0
            or str(candidate.MODEL.MULTI_GRANULARITY_GATING_INPUT) != "concat_global_local"
            or str(candidate.MODEL.MULTI_GRANULARITY_GATING_NORMALIZATION) != "scaled_softmax"
            or list(candidate.MODEL.MULTI_GRANULARITY_PART_SCALES) != list(SCALES)):
        raise ComparisonError("Candidate config is not the expected G2 τg=2.0 protocol")
    return candidate


def _require_tau(manifest_path, label, expected_tau, require_temperature=True):
    manifest = _read_json(manifest_path, label + " run manifest")
    try:
        actual = float(manifest["gating_temperature"])
    except (KeyError, TypeError, ValueError):
        if require_temperature:
            raise ComparisonError("{} manifest lacks numeric gating_temperature".format(label))
        actual = None
    if actual is not None and actual != float(expected_tau):
        raise ComparisonError("{} manifest gating temperature mismatch".format(label))
    if manifest.get("dataset", "").lower() != "market1501" or manifest.get("seed") != 42:
        raise ComparisonError("{} manifest is not the fixed Market1501 seed=42 protocol".format(label))
    return manifest


def _four_metrics(paths):
    records = [_metrics("G1", paths["g1"]), _metrics("G2 tau=1.0", paths["g2"]),
               _metrics("G2 tau=0.5", paths["tau0p5"])]
    baseline = records[2]
    records.append(_metrics("G2 tau=2.0", paths["tau2"], baseline))
    for record in records:
        # The inherited metric reader uses G2 τ=1.0 as a historical default.
        # This four-way ablation must instead state its direct τ=.5 baseline.
        record.pop("delta_Rank-1_vs_G2", None)
        record.pop("delta_mAP_vs_G2", None)
    fields = list(records[0]) + ["delta_Rank-1_vs_tau0p5", "delta_mAP_vs_tau0p5"]
    for record in records:
        record["delta_Rank-1_vs_tau0p5"] = (
            record["Rank-1"] - baseline["Rank-1"] if record["version"] == "G2 tau=2.0"
            else "not_applicable"
        )
        record["delta_mAP_vs_tau0p5"] = (
            record["mAP"] - baseline["mAP"] if record["version"] == "G2 tau=2.0"
            else "not_applicable"
        )
    return fields, records


def _plot_four_distributions(version_rows, figures):
    colors = {"G1": "#4c78a8", "G2 tau=1.0": "#f58518",
              "G2 tau=0.5": "#54a24b", "G2 tau=2.0": "#e45756"}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=180, sharey=True)
    for axis, scale in zip(axes, SCALES):
        for label in LABELS:
            values = [float(row["p{}".format(scale)]) for row in version_rows[label]]
            axis.hist(values, bins=np.linspace(0.0, 1.0, 31), density=True,
                      histtype="step", linewidth=1.8, color=colors[label], label=label)
        axis.set_title("K{} probability".format(scale)); axis.set_xlabel("p{}".format(scale))
        axis.set_xlim(0.0, 1.0); axis.grid(alpha=0.25)
    axes[0].set_ylabel("Density"); axes[-1].legend(fontsize=8)
    figure.suptitle("Frozen-sample Dynamic Gating probability distributions")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(figures / "gating_probability_distributions.{}".format(suffix)),
                       bbox_inches="tight")
    plt.close(figure)


def _plot_tau2_examples(scale, candidates, gates, market_root, figures):
    selected = [row for row in candidates
                if int(gates["G2 tau=2.0"][row["stable_sample_key"]]["dominant_k"]) == scale]
    selected = sorted(selected, key=lambda row: (row["selection_hash"], row["stable_sample_key"]))[:6]
    fields = ["stable_sample_key", "relative_path", "selection_hash"]
    for label in LABELS:
        prefix = PREFIXES[label]
        fields.extend([prefix + "_p2", prefix + "_p4", prefix + "_p6", prefix + "_dominant_k"])
    records = []
    for source in selected:
        record = {field: source[field] for field in fields[:3]}
        for label in LABELS:
            gate = gates[label][source["stable_sample_key"]]
            prefix = PREFIXES[label]
            for part in SCALES:
                record[prefix + "_p{}".format(part)] = gate["p{}".format(part)]
            record[prefix + "_dominant_k"] = gate["dominant_k"]
        records.append(record)
    _write_csv(figures / "k{}_dominant_samples.csv".format(scale), fields, records)
    figure, axes = plt.subplots(1, max(1, len(selected)), figsize=(2.5 * max(1, len(selected)), 4.8), dpi=180)
    axes = np.atleast_1d(axes)
    if not selected:
        axes[0].text(0.5, 0.5, "No τg=2.0 K{}-dominant fixed samples".format(scale),
                     ha="center", va="center")
        axes[0].axis("off")
    for axis, source in zip(axes, selected):
        image = plt.imread(str(_resolve_image(market_root, source)))
        gate = gates["G2 tau=2.0"][source["stable_sample_key"]]
        axis.imshow(image); axis.axis("off")
        axis.set_title("{}\np=({:.3f},{:.3f},{:.3f})\nK={} | G1/G2/τ.5 K={}/{}/{}".format(
            source["relative_path"].split("/")[-1], float(gate["p2"]), float(gate["p4"]),
            float(gate["p6"]), gate["dominant_k"],
            gates["G1"][source["stable_sample_key"]]["dominant_k"],
            gates["G2 tau=1.0"][source["stable_sample_key"]]["dominant_k"],
            gates["G2 tau=0.5"][source["stable_sample_key"]]["dominant_k"]), fontsize=6.5)
    figure.suptitle("τg=2.0 K{}-dominant samples; fixed selection-hash order".format(scale))
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(figures / "k{}_dominant_samples.{}".format(scale, suffix)), bbox_inches="tight")
    plt.close(figure)
    return len(selected)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ComparisonError("Refusing to reuse comparison output directory: {}".format(output))
    candidates = _read_fixed_candidates(args.candidate_manifest)
    candidate_keys = {row["stable_sample_key"] for row in candidates}
    manifests = {
        "g1": _require_tau(args.g1_run_manifest, "G1", 1.0, require_temperature=False),
        "g2": _require_tau(args.g2_run_manifest, "G2 tau=1.0", 1.0),
        "tau0p5": _require_tau(args.tau0p5_run_manifest, "G2 tau=0.5", 0.5),
        "tau2": _require_tau(args.tau2_run_manifest, "G2 tau=2.0", 2.0),
    }
    gates = {
        "G1": _read_fixed_gates(args.g1_fixed_gates, "G1"),
        "G2 tau=1.0": _read_fixed_gates(args.g2_fixed_gates, "G2 tau=1.0"),
        "G2 tau=0.5": _read_fixed_gates(args.tau0p5_fixed_gates, "G2 tau=0.5"),
    }
    if any(set(values) != candidate_keys for values in gates.values()):
        raise ComparisonError("Existing fixed gates do not exactly match frozen candidates")
    configuration = _validated_tau2_config(args.config_file)
    # A copied source YAML is not evidence that the trained process used it.
    # Refuse post-hoc plots if the finalizer's actual resolved configuration no
    # longer satisfies the same τg=2.0 protocol.
    _validated_tau2_config(args.resolved_config)
    market = _market_root(args.dataset_root)
    output.mkdir(parents=True)
    figures = output / "figures"; figures.mkdir()
    tau2_rows, checkpoint_sha = _extract_tau_gates(
        configuration, args.checkpoint, candidates, market,
        output / "g2_tau2_fixed_gating_samples.tsv", args.device,
    )
    gates["G2 tau=2.0"] = {row["stable_sample_key"]: row for row in tau2_rows}
    if set(gates["G2 tau=2.0"]) != candidate_keys:
        raise ComparisonError("τg=2.0 gates do not exactly match frozen candidates")
    ordered = {label: [gates[label][row["stable_sample_key"]] for row in candidates] for label in LABELS}
    stat_rows = [_statistics(label, ordered[label]) for label in LABELS]
    _write_csv(output / "gate_statistics.csv", list(stat_rows[0]), stat_rows)
    per_fields = ["stable_sample_key", "relative_path", "split", "pid", "camid"]
    for label in LABELS:
        prefix = PREFIXES[label]
        per_fields.extend([prefix + "_p2", prefix + "_p4", prefix + "_p6", prefix + "_dominant_k"])
    per_rows = []
    for source in candidates:
        row = {field: source[field] for field in per_fields[:5]}
        for label in LABELS:
            gate, prefix = gates[label][source["stable_sample_key"]], PREFIXES[label]
            for part in SCALES:
                row[prefix + "_p{}".format(part)] = gate["p{}".format(part)]
            row[prefix + "_dominant_k"] = gate["dominant_k"]
        per_rows.append(row)
    _write_csv(output / "per_sample_gating.csv", per_fields, per_rows)
    fields, metrics = _four_metrics({key: args.__dict__[key + "_run_manifest"] for key in manifests})
    _write_csv(output / "retrieval_metrics.csv", fields, metrics)
    _plot_four_distributions(ordered, figures)
    example_counts = {"K4": _plot_tau2_examples(4, candidates, gates, market, figures),
                      "K6": _plot_tau2_examples(6, candidates, gates, market, figures)}
    shutil.copyfile(str(Path(args.resolved_config).resolve()), str(output / "config_resolved.yml"))
    audit = {
        "analysis": "G1/G2/G2-tau0p5/G2-tau2 fixed-sample gating comparison",
        "dataset": "Market1501", "seed": 42, "sample_count": len(candidates),
        "algorithm_variable": "MODEL.MULTI_GRANULARITY_GATING_TAU: 0.5 -> 2.0",
        "fixed_candidate_manifest": {"path": str(Path(args.candidate_manifest).resolve()),
                                       "sha256": sha256_file(args.candidate_manifest)},
        "tau2_checkpoint": {"path": str(Path(args.checkpoint).resolve()), "sha256": checkpoint_sha},
        "formal_run_manifests": {key: str(Path(args.__dict__[key + "_run_manifest"]).resolve())
                                  for key in manifests},
        "example_counts": example_counts,
        "dominant_tie_rule": stat_rows[0]["dominant_tie_rule"],
    }
    _atomic_text(output / "run_manifest.json", json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_text(output / "README.md", "# G2 门控温度 τg=2.0 对比结果\n\n"
                 "本目录由 `compare_g1_g2_tau0p5_tau2_gating.py` 从四份正式 run manifest、"
                 "冻结候选样本和实际 checkpoint 自动生成。**本版本完整继承 τg=0.5 基线，仅将"
                 "门控温度 τg 从 0.5 改为 2.0，其余设置不变。**\n\n"
                 "交付文件：`retrieval_metrics.csv`（Rank-1/5/10、mAP）、`gate_statistics.csv`"
                 "（p2/p4/p6 均值及 K2/K4/K6 dominant ratio）、`per_sample_gating.csv`、"
                 "四版本概率分布 PNG/PDF 及 K4/K6 主导样本 PNG/PDF/CSV。\n\n"
                 "统计范围为 {} 个冻结 Market1501 query/gallery 样本；概率 p2+p4+p6=1，"
                 "scaled-softmax 融合权重 w=3p。dominant 使用 p 的 argmax；精确并列顺序为"
                 "K2、K4、K6。τg=2.0 K4 示例 {} 个，K6 示例 {} 个；不足或为零时仅输出实际数量。\n".format(
                     len(candidates), example_counts["K4"], example_counts["K6"]))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("candidate-manifest", "g1-fixed-gates", "g2-fixed-gates", "tau0p5-fixed-gates",
                   "g1-run-manifest", "g2-run-manifest", "tau0p5-run-manifest", "tau2-run-manifest",
                   "config-file", "resolved-config", "checkpoint", "dataset-root", "output-dir"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    print(str(run(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
