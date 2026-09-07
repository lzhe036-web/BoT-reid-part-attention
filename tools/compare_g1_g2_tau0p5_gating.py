#!/usr/bin/env python
"""Compare G1, formal G2 and a τg=0.5 G2 candidate on frozen images only.

The tool does not train.  It re-extracts the τg=0.5 gate on the immutable
candidate list already used for G1/G2, rejects incomplete pairing, and writes
all tables/plots from machine-derived samples and formal run manifests.
"""

from __future__ import absolute_import

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from collections import Counter
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

from config import cfg
from data.collate_batch import val_collate_fn
from data.datasets import ImageDataset
from data.transforms import build_transforms
from modeling import build_model
from tools.analyze_dynamic_gating import _state_dict
from utils.experiment_recording import sha256_file
from utils.reproducibility import make_data_loader_generator, seed_worker


SCALES = (2, 4, 6)
FIXED_FIELDS = (
    "stable_sample_key", "split", "relative_path", "pid", "camid",
    "selection_hash", "image_sha256", "p2", "p4", "p6", "w2", "w4",
    "w6", "dominant_k", "checkpoint_sha256",
)


class ComparisonError(RuntimeError):
    pass


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}".format(path.name, uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write_csv(path, fields, rows, delimiter=","):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=delimiter,
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ComparisonError("Cannot read {}: {}".format(label, error))


def _read_fixed_candidates(path):
    fields = ("stable_sample_key", "split", "relative_path", "pid", "camid",
              "selection_hash", "selection_rank", "image_sha256")
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or tuple(rows[0].keys()) != fields:
        raise ComparisonError("Fixed candidate manifest schema is invalid")
    if len({row["stable_sample_key"] for row in rows}) != len(rows):
        raise ComparisonError("Fixed candidate manifest has duplicate keys")
    for split in ("query", "gallery"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            raise ComparisonError("Fixed candidate manifest lacks {} rows".format(split))
        if [row["selection_hash"] for row in subset] != sorted(
                row["selection_hash"] for row in subset):
            raise ComparisonError("Fixed candidate manifest is not selection-hash sorted")
    return rows


def _read_fixed_gates(path, label):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or tuple(rows[0].keys()) != FIXED_FIELDS:
        raise ComparisonError("{} fixed-gate TSV schema is invalid".format(label))
    mapping = {}
    for row in rows:
        key = row["stable_sample_key"]
        if not key or key in mapping:
            raise ComparisonError("{} fixed-gate key is empty or duplicated".format(label))
        p = [float(row["p{}".format(scale)]) for scale in SCALES]
        w = [float(row["w{}".format(scale)]) for scale in SCALES]
        if (not all(math.isfinite(value) and value >= 0.0 for value in p + w)
                or not math.isclose(sum(p), 1.0, rel_tol=1e-6, abs_tol=1e-8)
                or not math.isclose(sum(w), 3.0, rel_tol=1e-6, abs_tol=1e-8)):
            raise ComparisonError("{} has invalid scaled-softmax values".format(label))
        dominant = SCALES[max(range(3), key=lambda index: p[index])]
        if int(row["dominant_k"]) != dominant:
            raise ComparisonError("{} dominant-K does not follow p2/p4/p6".format(label))
        mapping[key] = row
    return mapping


def _market_root(dataset_root):
    root = Path(dataset_root)
    for candidate in (root, root / "market1501", root / "Market-1501-v15.09",
                      root / "Market1501"):
        if (candidate / "query").is_dir() and (candidate / "bounding_box_test").is_dir():
            return candidate
    raise ComparisonError("Market1501 query/gallery directories are absent")


def _resolve_image(market_root, row):
    relative = Path(row["relative_path"])
    choices = (market_root.parent / relative, market_root / relative,
               market_root / Path(*relative.parts[1:]))
    for path in choices:
        if path.is_file():
            if sha256_file(path) != row["image_sha256"]:
                raise ComparisonError("Frozen image SHA256 changed: {}".format(path))
            return path
    raise ComparisonError("Frozen candidate image is absent: {}".format(relative))


def _load_candidate_config(path, expected_gating_input="concat_global_local"):
    configuration = cfg.clone()
    configuration.merge_from_file(str(path))
    configuration.freeze()
    if (str(configuration.DATASETS.NAMES).lower() != "market1501"
            or int(configuration.SEED) != 42
            or str(configuration.MODEL.MULTI_GRANULARITY_GATING_INPUT) != expected_gating_input
            or float(configuration.MODEL.MULTI_GRANULARITY_GATING_TAU) != 0.5
            or str(configuration.MODEL.MULTI_GRANULARITY_GATING_NORMALIZATION) != "scaled_softmax"
            or list(configuration.MODEL.MULTI_GRANULARITY_PART_SCALES) != list(SCALES)):
        raise ComparisonError("Candidate config is not the expected τg=0.5 protocol")
    return configuration


def _extract_tau_gates(configuration, checkpoint_path, candidates, market_root,
                       output_path, device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    state = _state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
    classifier = state.get("classifier.weight")
    if classifier is None or classifier.dim() != 2:
        raise ComparisonError("Selected checkpoint lacks classifier.weight")
    model_cfg = configuration.clone()
    model_cfg.defrost()
    model_cfg.MODEL.PRETRAIN_CHOICE = "none"
    model_cfg.MODEL.PRETRAIN_PATH = ""
    model_cfg.freeze()
    model = build_model(model_cfg, int(classifier.size(0)))
    model.load_state_dict(state, strict=True)
    actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(actual_device)
    model.eval()
    paths = [_resolve_image(market_root, row) for row in candidates]
    entries = [(str(path), int(row["pid"]), int(row["camid"]))
               for path, row in zip(paths, candidates)]
    loader = DataLoader(
        ImageDataset(entries, build_transforms(configuration, is_train=False)),
        batch_size=int(configuration.TEST.IMS_PER_BATCH), shuffle=False,
        num_workers=int(configuration.DATALOADER.NUM_WORKERS),
        collate_fn=val_collate_fn, worker_init_fn=seed_worker,
        generator=make_data_loader_generator(configuration.SEED, "query"),
    )
    rows, offset = [], 0
    with torch.no_grad():
        for images, _pids, _camids in loader:
            descriptors = model(images.to(actual_device))
            if descriptors.dim() != 2 or descriptors.size(1) != 2816:
                raise ComparisonError("Inference descriptor contract changed")
            evidence = model._last_dynamic_gating
            probabilities = evidence["probabilities"].detach().to(dtype=torch.float64, device="cpu")
            weights = evidence["weights"].detach().to(dtype=torch.float64, device="cpu")
            if probabilities.dim() != 2 or probabilities.size(1) != 3 or weights.shape != probabilities.shape:
                raise ComparisonError("Candidate gate tensor is not [B,3]")
            for index in range(probabilities.size(0)):
                p = [float(probabilities[index, item]) for item in range(3)]
                w = [float(weights[index, item]) for item in range(3)]
                if (not all(math.isfinite(value) and value >= 0.0 for value in p + w)
                        or not math.isclose(sum(p), 1.0, rel_tol=1e-6, abs_tol=1e-8)
                        or any(not math.isclose(w[i], 3.0 * p[i], rel_tol=1e-6, abs_tol=1e-8)
                               for i in range(3))):
                    raise ComparisonError("Candidate gate violates scaled-softmax semantics")
                source = candidates[offset + index]
                rows.append({
                    "stable_sample_key": source["stable_sample_key"], "split": source["split"],
                    "relative_path": source["relative_path"], "pid": source["pid"],
                    "camid": source["camid"], "selection_hash": source["selection_hash"],
                    "image_sha256": source["image_sha256"],
                    "p2": p[0], "p4": p[1], "p6": p[2],
                    "w2": w[0], "w4": w[1], "w6": w[2],
                    "dominant_k": SCALES[max(range(3), key=lambda i: p[i])],
                    "checkpoint_sha256": checkpoint_sha,
                })
            offset += probabilities.size(0)
    if offset != len(candidates):
        raise ComparisonError("Candidate fixed-gate extraction count mismatch")
    _write_csv(output_path, FIXED_FIELDS, rows, delimiter="\t")
    return rows, checkpoint_sha


def _statistics(label, rows):
    p = np.asarray([[float(row["p{}".format(scale)]) for scale in SCALES]
                    for row in rows], dtype=np.float64)
    dominant = [int(row["dominant_k"]) for row in rows]
    result = {"version": label, "sample_count": int(len(rows)),
              "dominant_tie_rule": "argmax p2,p4,p6; exact ties select K2 then K4 then K6"}
    for index, scale in enumerate(SCALES):
        result["p{}_mean".format(scale)] = float(np.mean(p[:, index]))
        result["dominant_k{}_ratio".format(scale)] = float(
            sum(value == scale for value in dominant) / len(rows)
        )
    return result


def _metrics(label, manifest_path, relative_to_g2=None):
    manifest = _read_json(manifest_path, label + " run manifest")
    metrics = manifest.get("metrics", {})
    record = {"version": label, "run_id": manifest.get("run_id", "not_recorded"),
              "checkpoint_sha256": manifest.get("selected_checkpoint", {}).get(
                  "sha256", "not_recorded"),
              "selected_epoch": metrics.get("selected_epoch", "not_recorded")}
    for source, target in (("rank1_percent", "Rank-1"), ("rank5_percent", "Rank-5"),
                           ("rank10_percent", "Rank-10"), ("map_percent", "mAP")):
        value = metrics.get(source)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ComparisonError("{} has no numeric {}".format(label, source))
        record[target] = float(value)
    record["delta_Rank-1_vs_G2"] = ("not_applicable" if relative_to_g2 is None
                                    else record["Rank-1"] - relative_to_g2["Rank-1"])
    record["delta_mAP_vs_G2"] = ("not_applicable" if relative_to_g2 is None
                                else record["mAP"] - relative_to_g2["mAP"])
    return record


def _plot_distributions(version_rows, figures):
    colors = {"G1": "#4c78a8", "G2 tau=1.0": "#f58518", "G2 tau=0.5": "#54a24b"}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=180, sharey=True)
    for axis, scale in zip(axes, SCALES):
        for label, rows in version_rows.items():
            values = [float(row["p{}".format(scale)]) for row in rows]
            axis.hist(values, bins=np.linspace(0.0, 1.0, 31), density=True,
                      histtype="step", linewidth=1.8, color=colors[label], label=label)
        axis.set_title("K{} probability".format(scale)); axis.set_xlabel("p{}".format(scale))
        axis.set_xlim(0.0, 1.0); axis.grid(alpha=0.25)
    axes[0].set_ylabel("Density"); axes[-1].legend(fontsize=8)
    figure.suptitle("Frozen-sample Dynamic Gating probability distributions")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(figures / "gating_probability_distributions.{}".format(suffix)), bbox_inches="tight")
    plt.close(figure)


def _plot_examples(target_scale, candidates, values, market_root, figures):
    selected = [row for row in candidates if int(values["G2 tau=0.5"][row["stable_sample_key"]]["dominant_k"]) == target_scale]
    selected = sorted(selected, key=lambda row: (row["selection_hash"], row["stable_sample_key"]))[:6]
    fields = ("stable_sample_key", "relative_path", "tau0p5_p2", "tau0p5_p4", "tau0p5_p6", "tau0p5_dominant_k")
    rows = []
    for row in selected:
        gate = values["G2 tau=0.5"][row["stable_sample_key"]]
        rows.append({"stable_sample_key": row["stable_sample_key"], "relative_path": row["relative_path"],
                     "tau0p5_p2": gate["p2"], "tau0p5_p4": gate["p4"], "tau0p5_p6": gate["p6"],
                     "tau0p5_dominant_k": gate["dominant_k"]})
    _write_csv(figures / "k{}_dominant_samples.csv".format(target_scale), fields, rows)
    figure, axes = plt.subplots(1, max(1, len(selected)), figsize=(2.4 * max(1, len(selected)), 4.0), dpi=180)
    axes = np.atleast_1d(axes)
    if not selected:
        axes[0].text(0.5, 0.5, "No K{}-dominant fixed samples".format(target_scale), ha="center", va="center")
        axes[0].axis("off")
    for axis, row in zip(axes, selected):
        image = plt.imread(str(_resolve_image(market_root, row)))
        tau = values["G2 tau=0.5"][row["stable_sample_key"]]
        g1 = values["G1"][row["stable_sample_key"]]; g2 = values["G2 tau=1.0"][row["stable_sample_key"]]
        axis.imshow(image); axis.axis("off")
        axis.set_title("{}\nτ=.5 p=({:.3f},{:.3f},{:.3f})\nG1/G2 K={}/{}".format(
            row["relative_path"].split("/")[-1], float(tau["p2"]), float(tau["p4"]), float(tau["p6"]),
            g1["dominant_k"], g2["dominant_k"]), fontsize=7)
    figure.suptitle("τg=0.5 K{}-dominant samples; fixed selection-hash order".format(target_scale))
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(figures / "k{}_dominant_samples.{}".format(target_scale, suffix)), bbox_inches="tight")
    plt.close(figure)
    return len(selected)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ComparisonError("Refusing to reuse comparison output directory: {}".format(output))
    candidates = _read_fixed_candidates(args.candidate_manifest)
    candidate_keys = {row["stable_sample_key"] for row in candidates}
    gates = {"G1": _read_fixed_gates(args.g1_fixed_gates, "G1"),
             "G2 tau=1.0": _read_fixed_gates(args.g2_fixed_gates, "G2")}
    if any(set(mapping) != candidate_keys for mapping in gates.values()):
        raise ComparisonError("G1/G2 fixed gates do not exactly match frozen candidates")
    output.mkdir(parents=True)
    figures = output / "figures"; figures.mkdir()
    configuration = _load_candidate_config(args.config_file)
    market = _market_root(args.dataset_root)
    tau_rows, checkpoint_sha = _extract_tau_gates(
        configuration, args.checkpoint, candidates, market,
        output / "g2_tau0p5_fixed_gating_samples.tsv", args.device,
    )
    gates["G2 tau=0.5"] = {row["stable_sample_key"]: row for row in tau_rows}
    if set(gates["G2 tau=0.5"]) != candidate_keys:
        raise ComparisonError("τg=0.5 gates do not exactly match frozen candidates")
    ordered = {label: [mapping[row["stable_sample_key"]] for row in candidates]
               for label, mapping in gates.items()}
    stat_rows = [_statistics(label, rows) for label, rows in ordered.items()]
    _write_csv(output / "gate_statistics.csv", list(stat_rows[0]), stat_rows)
    per_fields = ["stable_sample_key", "relative_path", "split", "pid", "camid"]
    for label in ("G1", "G2 tau=1.0", "G2 tau=0.5"):
        prefix = {"G1": "g1", "G2 tau=1.0": "g2", "G2 tau=0.5": "tau0p5"}[label]
        per_fields.extend([prefix + "_p2", prefix + "_p4", prefix + "_p6", prefix + "_dominant_k"])
    per_rows = []
    for candidate in candidates:
        key = candidate["stable_sample_key"]
        row = {field: candidate[field] for field in ("stable_sample_key", "relative_path", "split", "pid", "camid")}
        for label, prefix in (("G1", "g1"), ("G2 tau=1.0", "g2"), ("G2 tau=0.5", "tau0p5")):
            gate = gates[label][key]
            for scale in SCALES: row[prefix + "_p{}".format(scale)] = gate["p{}".format(scale)]
            row[prefix + "_dominant_k"] = gate["dominant_k"]
        per_rows.append(row)
    _write_csv(output / "per_sample_gating.csv", per_fields, per_rows)
    g1_metrics = _metrics("G1", args.g1_run_manifest)
    g2_metrics = _metrics("G2 tau=1.0", args.g2_run_manifest)
    tau_metrics = _metrics("G2 tau=0.5", args.tau_run_manifest, g2_metrics)
    _write_csv(output / "retrieval_metrics.csv", list(g1_metrics), [g1_metrics, g2_metrics, tau_metrics])
    _plot_distributions(ordered, figures)
    example_counts = {"K4": _plot_examples(4, candidates, gates, market, figures),
                      "K6": _plot_examples(6, candidates, gates, market, figures)}
    shutil.copyfile(str(Path(args.resolved_config).resolve()), str(output / "config_resolved.yml"))
    manifest = {
        "analysis": "G1/G2/G2-tau0p5 fixed-sample gating comparison",
        "baseline_g2_commit": _read_json(args.g2_run_manifest, "G2 manifest").get("commit", "not_recorded"),
        "candidate_training_commit": _read_json(args.tau_run_manifest, "tau manifest").get("commit", "not_recorded"),
        "candidate_gating_temperature": 0.5, "dataset": "Market1501", "seed": 42,
        "candidate_manifest": {"path": str(Path(args.candidate_manifest).resolve()), "sha256": sha256_file(args.candidate_manifest)},
        "selected_checkpoint": {"path": str(Path(args.checkpoint).resolve()), "sha256": checkpoint_sha},
        "sample_count": len(candidates), "dominant_tie_rule": stat_rows[0]["dominant_tie_rule"],
        "example_counts": example_counts,
        "source_files": {"g1_fixed_gates": str(Path(args.g1_fixed_gates).resolve()),
                         "g2_fixed_gates": str(Path(args.g2_fixed_gates).resolve()),
                         "resolved_config": str(Path(args.resolved_config).resolve())},
    }
    _atomic_text(output / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_text(output / "README.md", "# G2 门控温度 τg=0.5 对比\n\n"
                 "本目录由 `compare_g1_g2_tau0p5_gating.py` 从正式 run manifest、"
                 "冻结候选样本和实际 checkpoint 自动生成。G2 的唯一算法变量为 τg：1.0→0.5。\n\n"
                 "统计范围：{} 个固定 Market1501 query/gallery 样本；概率 p2+p4+p6=1，"
                 "scaled-softmax 融合权重 w=3p。dominant 采用 p 的 argmax；精确并列按 K2、K4、K6。\n\n"
                 "K4 示例：{}；K6 示例：{}。若 K6 为 0，图和 CSV 明确记录为零，未以其他样本替代。\n".format(
                     len(candidates), example_counts["K4"], example_counts["K6"]))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("candidate-manifest", "g1-fixed-gates", "g2-fixed-gates", "g1-run-manifest",
                   "g2-run-manifest", "tau-run-manifest", "config-file", "resolved-config",
                   "checkpoint", "dataset-root", "output-dir"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    print(str(run(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
