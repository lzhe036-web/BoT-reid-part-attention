#!/usr/bin/env python
"""Generate a strict full-query G1/G2/E1 gated-residual comparison package.

This tool is post-training only. It extracts all three controllers on one
frozen Market1501 query list, derives E1 residual coefficients from its actual
p/w values, and refuses reused output directories or inconsistent checkpoints.
"""

from __future__ import absolute_import

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
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

from config import cfg
from data.collate_batch import val_collate_fn
from data.datasets import ImageDataset, init_dataset
from data.transforms import build_transforms
from modeling import build_model
from tools.analyze_dynamic_gating import _state_dict
from utils.experiment_recording import sha256_file
from utils.reproducibility import make_data_loader_generator, seed_worker


SCALES = (2, 4, 6)
ALPHA = 0.5
VERSIONS = ("G1", "G2", "E1-static-dynamic-alpha0p5")


class ComparisonError(RuntimeError):
    pass


def _atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}".format(path.name, uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write_csv(path, fields, rows):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _load_config(path, expected_input, e1=False):
    configuration = cfg.clone(); configuration.merge_from_file(str(path)); configuration.freeze()
    if (str(configuration.DATASETS.NAMES).lower() != "market1501" or int(configuration.SEED) != 42
            or not configuration.MODEL.MULTI_GRANULARITY_DYNAMIC_GATING
            or str(configuration.MODEL.MULTI_GRANULARITY_GATING_INPUT) != expected_input
            or float(configuration.MODEL.MULTI_GRANULARITY_GATING_TAU) != 1.0
            or str(configuration.MODEL.MULTI_GRANULARITY_GATING_NORMALIZATION) != "scaled_softmax"
            or list(configuration.MODEL.MULTI_GRANULARITY_PART_SCALES) != list(SCALES)):
        raise ComparisonError("{} does not satisfy the fixed G1/G2 Market seed=42 protocol".format(path))
    if e1 and (configuration.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL is not True
               or float(configuration.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA) != ALPHA):
        raise ComparisonError("E1 config does not enable fixed alpha=0.5 static-dynamic residual")
    return configuration


def _stable_key(image_path, pid, camid, root):
    try:
        relative = Path(image_path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError as error:
        raise ComparisonError("Query image is outside dataset root: {}".format(image_path)) from error
    return "query|{}|{}|{}".format(relative, int(pid), int(camid)), relative


def _full_query_candidates(configuration):
    dataset = init_dataset(configuration.DATASETS.NAMES, root=configuration.DATASETS.ROOT_DIR,
                           verbose=False)
    rows = []
    for image_path, pid, camid in dataset.query:
        key, relative = _stable_key(image_path, pid, camid, configuration.DATASETS.ROOT_DIR)
        path = Path(image_path)
        if not path.is_file():
            raise ComparisonError("Query image is absent: {}".format(path))
        rows.append({"stable_sample_key": key, "relative_path": relative, "split": "query",
                     "pid": int(pid), "camid": int(camid), "image_path": str(path.resolve()),
                     "image_sha256": sha256_file(path)})
    rows.sort(key=lambda row: row["stable_sample_key"])
    if not rows or len({row["stable_sample_key"] for row in rows}) != len(rows):
        raise ComparisonError("Full-query candidate list is empty or contains duplicates")
    return rows, int(dataset.num_train_pids)


def _extract(configuration, checkpoint_path, candidates, num_classes, label, device):
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise ComparisonError("{} checkpoint is absent".format(label))
    checkpoint_sha = sha256_file(checkpoint_path)
    state = _state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
    classifier = state.get("classifier.weight")
    if classifier is None or classifier.dim() != 2 or int(classifier.size(0)) != num_classes:
        raise ComparisonError("{} checkpoint classifier does not match training identities".format(label))
    model_cfg = configuration.clone(); model_cfg.defrost()
    model_cfg.MODEL.PRETRAIN_CHOICE = "none"; model_cfg.MODEL.PRETRAIN_PATH = ""
    model_cfg.freeze()
    model = build_model(model_cfg, num_classes); model.load_state_dict(state, strict=True)
    actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(actual_device); model.eval()
    entries = [(row["image_path"], row["pid"], row["camid"]) for row in candidates]
    loader = DataLoader(
        ImageDataset(entries, build_transforms(configuration, is_train=False)),
        batch_size=int(configuration.TEST.IMS_PER_BATCH), shuffle=False,
        num_workers=int(configuration.DATALOADER.NUM_WORKERS), collate_fn=val_collate_fn,
        worker_init_fn=seed_worker, generator=make_data_loader_generator(configuration.SEED, "query"),
    )
    rows, offset = [], 0
    with torch.no_grad():
        for images, _pids, _camids in loader:
            descriptor = model(images.to(actual_device))
            if descriptor.dim() != 2 or descriptor.size(1) != 2816:
                raise ComparisonError("{} changed the 2816-D inference descriptor".format(label))
            evidence = model._last_dynamic_gating
            if not isinstance(evidence, dict):
                raise ComparisonError("{} emitted no gate evidence".format(label))
            probabilities = evidence["probabilities"].detach().to(device="cpu", dtype=torch.float64)
            weights = evidence["weights"].detach().to(device="cpu", dtype=torch.float64)
            if probabilities.dim() != 2 or probabilities.size(1) != 3 or weights.shape != probabilities.shape:
                raise ComparisonError("{} gate tensors are not [B,3]".format(label))
            for index in range(probabilities.size(0)):
                p = [float(probabilities[index, item]) for item in range(3)]
                w = [float(weights[index, item]) for item in range(3)]
                if (not all(math.isfinite(value) and value >= 0.0 for value in p + w)
                        or not math.isclose(sum(p), 1.0, rel_tol=1e-6, abs_tol=1e-8)
                        or any(not math.isclose(weight, 3.0 * probability, rel_tol=1e-6, abs_tol=1e-8)
                               for probability, weight in zip(p, w))):
                    raise ComparisonError("{} p/w violates scaled-softmax semantics".format(label))
                source = candidates[offset + index]
                record = dict(source); record.pop("image_path")
                record.update({"version": label, "p2": p[0], "p4": p[1], "p6": p[2],
                               "w2": w[0], "w4": w[1], "w6": w[2],
                               "dominant_k": SCALES[max(range(3), key=lambda item: p[item])],
                               "checkpoint_sha256": checkpoint_sha,
                               "alpha": "not_applicable", "residual2": "not_applicable",
                               "residual4": "not_applicable", "residual6": "not_applicable",
                               "c2": "not_applicable", "c4": "not_applicable", "c6": "not_applicable"})
                if label == "E1-static-dynamic-alpha0p5":
                    residual = [ALPHA * value for value in w]
                    coefficients = [1.0 + value for value in residual]
                    record.update({"alpha": ALPHA, "residual2": residual[0], "residual4": residual[1],
                                   "residual6": residual[2], "c2": coefficients[0], "c4": coefficients[1],
                                   "c6": coefficients[2]})
                rows.append(record)
            offset += int(probabilities.size(0))
    if offset != len(candidates):
        raise ComparisonError("{} full-query extraction count mismatch".format(label))
    return rows, checkpoint_sha


def _stats(label, rows):
    output = {"version": label, "sample_scope": "complete Market1501 query set",
              "sample_count": len(rows), "dominant_tie_rule": "argmax(p2,p4,p6); exact ties select K2 then K4 then K6"}
    for scale in SCALES:
        values = [float(row["p{}".format(scale)]) for row in rows]
        output["p{}_mean".format(scale)] = float(np.mean(values))
        output["dominant_k{}_ratio".format(scale)] = float(sum(
            int(row["dominant_k"]) == scale for row in rows) / len(rows))
    return output


def _read_result(path, label):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        checkpoint = payload["selected_checkpoint"]
        record = {"version": label, "result_source": str(Path(path).resolve()),
                  "selected_epoch": metrics.get("selected_epoch", checkpoint.get("epoch", "not_recorded")),
                  "checkpoint_sha256": checkpoint["sha256"]}
        for source, destination in (("rank1_percent", "Rank-1"), ("rank5_percent", "Rank-5"),
                                    ("rank10_percent", "Rank-10"), ("map_percent", "mAP")):
            record[destination] = float(metrics[source])
        return record
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ComparisonError("Cannot read complete formal metrics for {}: {}".format(label, error))


def _plot_probabilities(grouped, figures):
    colors = {"G1": "#4c78a8", "G2": "#f58518", "E1-static-dynamic-alpha0p5": "#54a24b"}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=180, sharey=True)
    for axis, scale in zip(axes, SCALES):
        for label in VERSIONS:
            axis.hist([float(row["p{}".format(scale)]) for row in grouped[label]],
                      bins=np.linspace(0.0, 1.0, 31), density=True, histtype="step",
                      linewidth=1.8, color=colors[label], label=label)
        axis.set_title("K{} gate probability p{}".format(scale, scale)); axis.set_xlim(0, 1); axis.grid(alpha=.25)
    axes[0].set_ylabel("Density"); axes[-1].legend(fontsize=8)
    figure.suptitle("G1/G2/E1 complete-query gate probability distributions")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(figures / ("gate_probability_distributions." + suffix)), bbox_inches="tight")
    plt.close(figure)


def _plot_e1_coefficients(rows, figures):
    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    for scale in SCALES:
        axis.hist([float(row["c{}".format(scale)]) for row in rows], bins=30, density=True,
                  histtype="step", linewidth=1.8, label="c{}=1+0.5w{}".format(scale, scale))
    axis.set_title("E1 final local coefficients (not probabilities)"); axis.set_xlabel("c"); axis.set_ylabel("Density")
    axis.grid(alpha=.25); axis.legend(); figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(figures / ("e1_final_local_coefficient_distribution." + suffix)), bbox_inches="tight")
    plt.close(figure)


def _sample_figures(candidates, grouped, figures):
    maps = {label: {row["stable_sample_key"]: row for row in rows} for label, rows in grouped.items()}
    counts = {}
    for scale in (4, 6):
        selected = [row for row in candidates if int(maps["E1-static-dynamic-alpha0p5"][row["stable_sample_key"]]["dominant_k"]) == scale]
        selected = sorted(selected, key=lambda row: row["stable_sample_key"])[:6]
        fields = ["stable_sample_key", "relative_path", "pid", "camid"]
        for label, prefix in (("G1", "g1"), ("G2", "g2"), ("E1-static-dynamic-alpha0p5", "e1")):
            fields.extend([prefix + "_p2", prefix + "_p4", prefix + "_p6", prefix + "_dominant_k"])
        result = []
        for item in selected:
            line = {field: item[field] for field in fields[:4]}
            for label, prefix in (("G1", "g1"), ("G2", "g2"), ("E1-static-dynamic-alpha0p5", "e1")):
                source = maps[label][item["stable_sample_key"]]
                for part in SCALES: line[prefix + "_p{}".format(part)] = source["p{}".format(part)]
                line[prefix + "_dominant_k"] = source["dominant_k"]
            result.append(line)
        _write_csv(figures / "k{}_dominant_samples.csv".format(scale), fields, result)
        figure, axes = plt.subplots(1, max(1, len(selected)), figsize=(2.5 * max(1, len(selected)), 4.9), dpi=180)
        axes = np.atleast_1d(axes)
        if not selected:
            axes[0].text(.5, .5, "No E1 K{}-dominant query samples".format(scale), ha="center", va="center")
            axes[0].axis("off")
        for axis, item in zip(axes, selected):
            axis.imshow(plt.imread(item["image_path"])); axis.axis("off")
            row = maps["E1-static-dynamic-alpha0p5"][item["stable_sample_key"]]
            axis.set_title("{}\np=({:.3f},{:.3f},{:.3f}); K={}\nG1/G2 K={}/{}".format(
                Path(item["relative_path"]).name, float(row["p2"]), float(row["p4"]), float(row["p6"]),
                row["dominant_k"], maps["G1"][item["stable_sample_key"]]["dominant_k"],
                maps["G2"][item["stable_sample_key"]]["dominant_k"]), fontsize=6.5)
        figure.suptitle("E1 K{}-dominant examples; stable-key order".format(scale)); figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(str(figures / ("k{}_dominant_samples.".format(scale) + suffix)), bbox_inches="tight")
        plt.close(figure); counts["K{}".format(scale)] = len(selected)
    return counts


def _git_diff():
    try:
        return subprocess.check_output([
            "git", "-C", str(REPO_ROOT), "diff", "--no-ext-diff",
            "5a9a2a33e19f3d9dfc3afaf356dc8b9ee3f0f737", "HEAD", "--",
            "modeling/baseline.py", "modeling/__init__.py", "config/defaults.py",
        ], stderr=subprocess.PIPE).decode("utf-8", errors="replace")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ComparisonError("Cannot generate code difference evidence") from error


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ComparisonError("Refusing to reuse result directory: {}".format(output))
    g1_config = _load_config(args.g1_config, "global")
    g2_config = _load_config(args.g2_config, "concat_global_local")
    e1_config = _load_config(args.e1_config, "concat_global_local", e1=True)
    # The source YAML declares the intended treatment.  The persisted resolved
    # YAML is the evidence that the training/evaluation process actually saw
    # that treatment, so do not merely copy it into the package unchecked.
    _load_config(args.e1_resolved_config, "concat_global_local", e1=True)
    candidates, classes = _full_query_candidates(e1_config)
    output.mkdir(parents=True); figures = output / "figures"; figures.mkdir()
    grouped, checkpoint_sha = {}, {}
    for label, configuration, checkpoint in (("G1", g1_config, args.g1_checkpoint),
                                               ("G2", g2_config, args.g2_checkpoint),
                                               ("E1-static-dynamic-alpha0p5", e1_config, args.e1_checkpoint)):
        rows, digest = _extract(configuration, checkpoint, candidates, classes, label, args.device)
        grouped[label], checkpoint_sha[label] = rows, digest
    fields = ("version", "stable_sample_key", "relative_path", "split", "pid", "camid", "image_sha256",
              "p2", "p4", "p6", "w2", "w4", "w6", "alpha", "residual2", "residual4", "residual6",
              "c2", "c4", "c6", "dominant_k", "checkpoint_sha256")
    _write_csv(output / "per_sample_gating.csv", fields, [row for label in VERSIONS for row in grouped[label]])
    _write_csv(output / "sample_selection.csv", ("stable_sample_key", "relative_path", "split", "pid", "camid", "image_sha256"),
               [{key: row[key] for key in ("stable_sample_key", "relative_path", "split", "pid", "camid", "image_sha256")} for row in candidates])
    statistics = [_stats(label, grouped[label]) for label in VERSIONS]
    _write_csv(output / "gate_statistics.csv", list(statistics[0]), statistics)
    metrics = [_read_result(args.g1_result, "G1"), _read_result(args.g2_result, "G2"),
               _read_result(args.e1_result, "E1-static-dynamic-alpha0p5")]
    g2_metrics = metrics[1]
    for record in metrics:
        record["delta_Rank-1_vs_G2"] = record["Rank-1"] - g2_metrics["Rank-1"] if record["version"].startswith("E1") else "not_applicable"
        record["delta_mAP_vs_G2"] = record["mAP"] - g2_metrics["mAP"] if record["version"].startswith("E1") else "not_applicable"
    _write_csv(output / "retrieval_metrics.csv", list(metrics[0]) + ["delta_Rank-1_vs_G2", "delta_mAP_vs_G2"], metrics)
    _plot_probabilities(grouped, figures); _plot_e1_coefficients(grouped["E1-static-dynamic-alpha0p5"], figures)
    counts = _sample_figures(candidates, grouped, figures)
    shutil.copyfile(str(Path(args.e1_resolved_config).resolve()), str(output / "config_resolved.yml"))
    _atomic_text(output / "code_config_diff.md", "# 相对原始 G2 的代码与配置差异\n\n"
                 "唯一算法改动：静态拼接加门控残差，固定 alpha=0.5；tau_g 保持 1.0。\n\n```diff\n" + _git_diff() + "\n```\n")
    e1_run = json.loads(Path(args.e1_run_manifest).read_text(encoding="utf-8"))
    if e1_run.get("branch") != "codex/g2-e1-static-dynamic-alpha0p5":
        raise ComparisonError("E1 registered run manifest has the wrong branch")
    manifest = {"analysis": "G1/G2/E1 complete Market1501 query gating comparison", "dataset": "Market1501", "seed": 42,
                "sample_scope": "complete query", "sample_count": len(candidates), "alpha": ALPHA, "tau_g": 1.0,
                "fusion_mode": "static_concat_plus_gated_residual", "formula": "concat(g,(1+0.5w2)z2,(1+0.5w4)z4,(1+0.5w6)z6)",
                "checkpoint_sha256": checkpoint_sha, "example_counts": counts,
                "baseline_g2_commit": e1_run.get("parent_commit", "not_recorded"),
                "e1_training_code_commit": e1_run.get("commit", "not_recorded"),
                "e1_training_command": e1_run.get("command", "not_recorded"),
                "e1_environment_evidence": e1_run.get("environment", "not_recorded"),
                "e1_run_manifest": str(Path(args.e1_run_manifest).resolve()),
                "result_sources": {"g1": str(Path(args.g1_result).resolve()), "g2": str(Path(args.g2_result).resolve()), "e1": str(Path(args.e1_result).resolve())}}
    _atomic_text(output / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_text(output / "README.md", "# E1 静态拼接 + 门控残差（alpha=0.5）\n\n"
                 "本目录由程序从完整正式评估指标、三个实际 checkpoint 和同一 Market1501 完整 query 集生成。"
                 "E1 相对原始 G2 的唯一算法变化是：保留 2816 维静态拼接并增加固定 alpha=0.5 的局部门控残差；tau_g=1.0。\n\n"
                 "`p` 是概率，p2+p4+p6=1；`w=3p` 是原始动态权重；`residual=0.5w`；`c=1+0.5w` 是 E1 最终局部系数，**不是概率**。\n\n"
                 "K4 示例 {} 个，K6 示例 {} 个；不足时仅输出实际数量。\n".format(counts["K4"], counts["K6"]))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("g1-config", "g1-checkpoint", "g1-result", "g2-config", "g2-checkpoint", "g2-result",
                   "e1-config", "e1-resolved-config", "e1-checkpoint", "e1-result", "e1-run-manifest", "output-dir"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv); print(run(args)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
