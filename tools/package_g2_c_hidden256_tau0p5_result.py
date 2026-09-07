#!/usr/bin/env python
"""Package machine-generated single-run G2-C evidence without retraining.

The package deliberately reports the deterministic 256-image gate sample as a
gate-observation subset and the validation-history selection as full retrieval
evaluation.  It never fills a metric, image label, or gate value by hand.
"""

from __future__ import absolute_import

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCALES = (2, 4, 6)
PACKAGE_NAME = "g2_c_hidden256_tau0p5_result_package"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _parse_stable_key(key):
    fields = str(key).split("|")
    if len(fields) != 5 or fields[0] not in ("query", "gallery"):
        raise ValueError("Invalid stable sample key: {!r}".format(key))
    return fields[0], fields[1], int(fields[2]), int(fields[3])


def _statistics(rows):
    count = len(rows)
    if count <= 0:
        raise ValueError("No gate samples to package")
    result = []
    dominant = [int(row["dominant_k"]) for row in rows]
    for scale in SCALES:
        probability = np.asarray([float(row["p{}".format(scale)]) for row in rows])
        weight = np.asarray([float(row["w{}".format(scale)]) for row in rows])
        result.append({
            "scale": "K{}".format(scale), "sample_count": count,
            "p_mean": float(probability.mean()), "p_std": float(probability.std(ddof=0)),
            "w_mean": float(weight.mean()), "w_std": float(weight.std(ddof=0)),
            "dominant_ratio": float(sum(item == scale for item in dominant)) / float(count),
            "dominant_rule": "argmax(p2,p4,p6); first-index tie behavior of torch.argmax",
            "sample_protocol": "deterministic sha256(stable_sample_key) ascending subset",
        })
    return result


def _plot_distributions(rows, destination):
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=180)
    for index, scale in enumerate(SCALES):
        p = np.asarray([float(row["p{}".format(scale)]) for row in rows])
        w = np.asarray([float(row["w{}".format(scale)]) for row in rows])
        axes[0].hist(p, bins=np.linspace(0.0, 1.0, 31), histtype="step", linewidth=1.7,
                     label="p{}".format(scale))
        axes[1].hist(w, bins=np.linspace(0.0, 3.0, 31), histtype="step", linewidth=1.7,
                     label="w{}".format(scale))
    axes[0].set(xlabel="Probability p", ylabel="Sample count", xlim=(0.0, 1.0),
                title="G2-C gate probability distribution")
    axes[1].set(xlabel="Applied weight w=3p", ylabel="Sample count", xlim=(0.0, 3.0),
                title="G2-C scaled gate-weight distribution")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(str(destination), bbox_inches="tight")
    figure.savefig(str(destination.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(figure)


def _plot_examples(rows, dataset_root, scale, destination):
    selected = sorted(
        (row for row in rows if int(row["dominant_k"]) == scale),
        key=lambda row: row["stable_sample_key"],
    )[:6]
    figure, axes = plt.subplots(1, max(1, len(selected)), figsize=(2.3 * max(1, len(selected)), 4.1), dpi=180)
    axes = np.atleast_1d(axes)
    for axis in axes:
        axis.axis("off")
    for axis, row in zip(axes, selected):
        _split, relative_path, _pid, _camid = _parse_stable_key(row["stable_sample_key"])
        image_path = dataset_root / relative_path
        if image_path.is_file():
            with Image.open(str(image_path)) as image:
                axis.imshow(image.convert("RGB"))
        else:
            axis.text(0.5, 0.5, "image unavailable", ha="center", va="center")
        axis.set_title(
            "{}\np=({:.3f},{:.3f},{:.3f})".format(
                row["stable_sample_key"].split("|")[1].rsplit("/", 1)[-1],
                float(row["p2"]), float(row["p4"]), float(row["p6"]),
            ), fontsize=7,
        )
    figure.suptitle("G2-C deterministic K{}-dominant examples (all available up to 6)".format(scale))
    figure.tight_layout()
    figure.savefig(str(destination), bbox_inches="tight")
    figure.savefig(str(destination.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(figure)
    return selected


def _git_diff(commit):
    command = ["git", "-C", str(REPO_ROOT), "diff", "{}^".format(commit), commit, "--"]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return completed.stdout.decode("utf-8", errors="replace")


def package(output_dir):
    output_dir = Path(output_dir).resolve()
    result_path = output_dir / "g2_c_hidden256_tau0p5_formal_result.json"
    result = _read_json(result_path)
    evidence = result.get("evidence", {})
    analysis_manifest = _read_json(evidence["analysis_manifest"])
    samples_path = Path(analysis_manifest["files"]["test_gate_samples_tsv"]["path"])
    with samples_path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))
    if not source_rows:
        raise ValueError("Formal G2-C analysis has no sample-level gate evidence")
    if any(row.get("checkpoint_sha256") != result["selected_checkpoint"]["sha256"] for row in source_rows):
        raise ValueError("Per-sample gate evidence is bound to another checkpoint")

    package_dir = output_dir / PACKAGE_NAME
    if package_dir.exists():
        raise FileExistsError("Refusing to overwrite existing G2-C package: {}".format(package_dir))
    package_dir.mkdir()
    figures = package_dir / "figures"
    figures.mkdir()
    config_path = Path(evidence["config"])
    resolved_path = output_dir / "config_resolved.yml"
    for source, target in ((config_path, package_dir / "config_source.yml"), (resolved_path, package_dir / "config_resolved.yml")):
        if not source.is_file():
            raise FileNotFoundError("Required formal evidence is absent: {}".format(source))
        shutil.copyfile(str(source), str(target))

    package_rows = []
    for row in source_rows:
        split, relative_path, pid, camid = _parse_stable_key(row["stable_sample_key"])
        package_rows.append(dict(row, version="G2-C-hidden256-tau0p5", relative_image_path=relative_path,
                                 split=split, pid=pid, camid=camid))
    per_sample_fields = ["version", "stable_sample_key", "relative_image_path", "split", "pid", "camid",
                         "p2", "p4", "p6", "w2", "w4", "w6", "entropy", "dominant_k", "checkpoint_sha256"]
    _write_csv(package_dir / "per_sample_gating.csv", per_sample_fields, package_rows)
    _write_csv(package_dir / "sample_manifest.csv", per_sample_fields, package_rows)
    statistics = _statistics(package_rows)
    _write_csv(package_dir / "gate_statistics.csv", list(statistics[0]), statistics)
    _plot_distributions(package_rows, figures / "g2_c_gate_distributions.png")

    dataset_root = Path(_read_json(evidence["analysis_manifest"]).get("config_path", "")).parent
    # The resolved config is the authoritative dataset-root carrier.  Avoid
    # treating a missing image as a replacement sample.
    from utils.config_serialization import deserialize_cfg_node_yaml
    resolved = deserialize_cfg_node_yaml(resolved_path.read_text(encoding="utf-8"))
    dataset_root = Path(str(resolved["DATASETS"]["ROOT_DIR"]))
    selected = []
    for scale in (4, 6):
        selected.extend(_plot_examples(package_rows, dataset_root, scale, figures / "k{}_dominant_examples.png".format(scale)))
    selection_rows = [{"selection_rule": "stable_sample_key ascending; first six within dominant class",
                       "dominant_k": row["dominant_k"], "stable_sample_key": row["stable_sample_key"],
                       "p2": row["p2"], "p4": row["p4"], "p6": row["p6"]} for row in selected]
    _write_csv(package_dir / "sample_selection.csv", list(selection_rows[0]) if selection_rows else ["selection_rule", "dominant_k", "stable_sample_key", "p2", "p4", "p6"], selection_rows)

    metrics = result["metrics"]
    metric_row = {
        "experiment": "G2-C-hidden256-tau0p5", "checkpoint_epoch": result["selected_checkpoint"]["epoch"],
        "checkpoint_sha256": result["selected_checkpoint"]["sha256"],
        "rank1_percent": metrics["rank1_percent"], "rank5_percent": metrics["rank5_percent"],
        "rank10_percent": metrics["rank10_percent"], "map_percent": metrics["map_percent"],
        "metric_source": "machine-selected validation_history.jsonl record",
    }
    _write_csv(package_dir / "retrieval_metrics.csv", list(metric_row), [metric_row])
    _atomic_text(package_dir / "code_diff_from_original_g2.patch", _git_diff(result["commit"]))
    manifest = {
        "experiment": "G2-C-hidden256-tau0p5",
        "baseline_commit": "5a9a2a33e19f3d9dfc3afaf356dc8b9ee3f0f737",
        "training_commit": result["commit"], "branch": result["branch"], "dataset": "market1501",
        "seed": 42, "controller_architecture": "mlp", "controller_hidden_dim": 256,
        "gating_input": "concat([g,z2,z4,z6])", "gating_temperature": 0.5,
        "selected_checkpoint": result["selected_checkpoint"],
        "gate_sample_protocol": analysis_manifest["test_weight_protocol"],
        "gate_sample_count": len(package_rows), "retrieval_metric_protocol": metric_row["metric_source"],
        "controller_block_proxy": analysis_manifest["controller_block_plot_semantics"],
    }
    _atomic_text(package_dir / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    readme = """# G2-C formal result package

This package is generated only after a formal G2-C finalizer selects one checkpoint. It preserves the two declared differences from original G2: `Linear(2816,3) -> Linear(2816,256) -> ReLU -> Linear(256,3)` and `tau_g: 1.0 -> 0.5`. Seed is fixed at 42.

`retrieval_metrics.csv` comes from the selected full validation record. `per_sample_gating.csv`, `gate_statistics.csv`, and figures use the deterministic bounded analysis subset recorded in `run_manifest.json`; it is not silently interchangeable with a full-query analysis. MLP input-block coefficient norms are marked `not_applicable` because no direct 2816-to-3 coefficient matrix exists.
"""
    _atomic_text(package_dir / "README.md", readme)
    files = sorted(path for path in package_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    _atomic_text(package_dir / "SHA256SUMS.txt", "".join("{}  {}\n".format(_sha256(path), path.relative_to(package_dir).as_posix()) for path in files))
    return package_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(str(package(args.output_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
