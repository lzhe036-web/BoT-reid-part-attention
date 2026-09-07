#!/usr/bin/env python
"""Build a self-contained, machine-derived G2-D1 delivery package."""

from __future__ import absolute_import

import argparse
import csv
import hashlib
import json
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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_serialization import deserialize_cfg_node_yaml
PACKAGE_NAME = "g2_d1_result_package"
SCALES = (2, 4, 6)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}".format(path.name, uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write(path, fields, rows, delimiter=","):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=delimiter,
                            lineterminator="\n")
    writer.writeheader(); writer.writerows(rows); _atomic_text(path, buffer.getvalue())


def _json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _key(row):
    values = str(row["stable_sample_key"]).split("|")
    if len(values) != 4 or values[0] not in ("query", "gallery"):
        raise ValueError("Invalid stable_sample_key: {!r}".format(row["stable_sample_key"]))
    return values[0], values[1], int(values[2]), int(values[3])


def _statistics(rows):
    output, count = [], len(rows)
    if count <= 0:
        raise ValueError("No selected-checkpoint gate samples")
    for scale in SCALES:
        p = np.asarray([float(row["p{}".format(scale)]) for row in rows])
        w = np.asarray([float(row["w{}".format(scale)]) for row in rows])
        output.append({
            "scale": "K{}".format(scale), "sample_count": count,
            "p_mean": float(p.mean()), "p_std": float(p.std(ddof=0)),
            "w_mean": float(w.mean()), "w_std": float(w.std(ddof=0)),
            "dominant_ratio": float(sum(int(row["dominant_k"]) == scale for row in rows)) / count,
            "dominant_tie_rule": "torch.argmax(p2,p4,p6): ties choose K2, then K4, then K6",
            "sample_scope": "deterministic sha256(stable_sample_key) ascending bounded query+gallery subset",
        })
    return output


def _plot_distribution(rows, path):
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), dpi=180)
    for scale in SCALES:
        axes[0].hist([float(row["p{}".format(scale)]) for row in rows],
                     bins=np.linspace(0, 1, 31), histtype="step", linewidth=1.7,
                     label="p{}".format(scale))
        axes[1].hist([float(row["w{}".format(scale)]) for row in rows],
                     bins=np.linspace(0, 3, 31), histtype="step", linewidth=1.7,
                     label="w{}".format(scale))
    axes[0].set(title="G2-D1 probability distribution", xlabel="p", ylabel="sample count", xlim=(0, 1))
    axes[1].set(title="G2-D1 applied weight distribution", xlabel="w=3p", ylabel="sample count", xlim=(0, 3))
    for axis in axes: axis.grid(alpha=.25); axis.legend()
    figure.tight_layout(); figure.savefig(str(path), bbox_inches="tight")
    figure.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight"); plt.close(figure)


def _plot_examples(rows, dataset_root, scale, path, images_dir):
    selected = sorted((row for row in rows if int(row["dominant_k"]) == scale),
                      key=lambda row: row["stable_sample_key"])[:6]
    figure, axes = plt.subplots(1, max(1, len(selected)), figsize=(2.4 * max(1, len(selected)), 4.2), dpi=180)
    axes = np.atleast_1d(axes)
    for axis in axes: axis.axis("off")
    for index, (axis, row) in enumerate(zip(axes, selected)):
        _split, relative, _pid, _camid = _key(row)
        source = dataset_root / relative
        if source.is_file():
            target = images_dir / "K{}_{}.jpg".format(scale, index + 1)
            shutil.copyfile(str(source), str(target))
            with Image.open(str(source)) as image: axis.imshow(image.convert("RGB"))
        else:
            axis.text(.5, .5, "image unavailable", ha="center", va="center")
        axis.set_title("{}\np=({:.3f},{:.3f},{:.3f})".format(
            Path(relative).name, float(row["p2"]), float(row["p4"]), float(row["p6"])), fontsize=7)
    figure.suptitle("G2-D1 K{}-dominant examples; stable-key order".format(scale))
    figure.tight_layout(); figure.savefig(str(path), bbox_inches="tight")
    figure.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight"); plt.close(figure)
    return selected


def _diff(base, commit):
    completed = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", base, commit, "--"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return completed.stdout.decode("utf-8", errors="replace")


def package(output_dir):
    output_dir = Path(output_dir).resolve()
    result_path = output_dir / "g2_d1_formal_result.json"
    result = _json(result_path)
    reproducibility_path = output_dir / "reproducibility.json"
    if not reproducibility_path.is_file():
        raise FileNotFoundError("Formal reproducibility evidence is absent: {}".format(reproducibility_path))
    reproducibility = _json(reproducibility_path)
    evidence = result.get("evidence", {})
    analysis = _json(evidence["analysis_manifest"])
    samples_path = Path(analysis["files"]["test_gate_samples_tsv"]["path"])
    with samples_path.open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    selected_sha = result["selected_checkpoint"]["sha256"]
    if not raw or any(row.get("checkpoint_sha256") != selected_sha for row in raw):
        raise ValueError("Gate samples are absent or bound to another checkpoint")
    package_dir = output_dir / PACKAGE_NAME
    if package_dir.exists():
        raise FileExistsError("Refusing to overwrite delivery package: {}".format(package_dir))
    figures, images = package_dir / "figures", package_dir / "sample_images"
    figures.mkdir(parents=True); images.mkdir()
    source_config, resolved_config = Path(evidence["config"]), output_dir / "config_resolved.yml"
    for source, target in ((source_config, package_dir / "config_source.yml"),
                           (resolved_config, package_dir / "config_resolved.yml")):
        if not source.is_file(): raise FileNotFoundError("Required formal evidence: {}".format(source))
        shutil.copyfile(str(source), str(target))
    script_dir = package_dir / "scripts"
    script_dir.mkdir()
    for source in (
            REPO_ROOT / "tools" / "package_g2_diff46_result.py",
            REPO_ROOT / "tools" / "compare_g1_g2_g2a_diff46_gating.py",
            REPO_ROOT / "tools" / "analyze_g2_global_local_gating.py",
            REPO_ROOT / "scripts" / "export_g2_diff46_result_autodl.sh"):
        if not source.is_file():
            raise FileNotFoundError("Required reproducibility script: {}".format(source))
        shutil.copyfile(str(source), str(script_dir / source.name))
    packaged = []
    for row in raw:
        split, relative, pid, camid = _key(row)
        packaged.append({
            "version": "G2-D1", "stable_sample_key": row["stable_sample_key"],
            "relative_image_path": relative, "split": split, "pid": pid, "camid": camid,
            "p2": row["p2"], "p4": row["p4"], "p6": row["p6"],
            "w2": row["w2"], "w4": row["w4"], "w6": row["w6"],
            "entropy": row["entropy"], "dominant_k": row["dominant_k"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        })
    sample_fields = ["version", "stable_sample_key", "relative_image_path", "split", "pid", "camid",
                     "p2", "p4", "p6", "w2", "w4", "w6", "entropy", "dominant_k", "checkpoint_sha256"]
    _write(package_dir / "per_sample_gating.tsv", sample_fields, packaged, "\t")
    _write(package_dir / "sample_manifest.tsv", sample_fields, packaged, "\t")
    statistics = _statistics(packaged); _write(package_dir / "gate_statistics.csv", list(statistics[0]), statistics)
    _plot_distribution(packaged, figures / "g2_d1_gate_distributions.png")
    resolved = deserialize_cfg_node_yaml(resolved_config.read_text(encoding="utf-8"))
    dataset_root = Path(str(resolved["DATASETS"]["ROOT_DIR"]))
    chosen = []
    for scale in (4, 6):
        chosen.extend(_plot_examples(packaged, dataset_root, scale,
                                     figures / "k{}_dominant_examples.png".format(scale), images))
    _write(package_dir / "sample_selection.csv",
           ["selection_rule", "dominant_k", "stable_sample_key", "p2", "p4", "p6"],
           [{"selection_rule": "stable_sample_key ascending; first six within dominant class",
             "dominant_k": row["dominant_k"], "stable_sample_key": row["stable_sample_key"],
             "p2": row["p2"], "p4": row["p4"], "p6": row["p6"]} for row in chosen])
    metrics = result["metrics"]
    _write(package_dir / "retrieval_metrics.csv",
           ["experiment", "checkpoint_epoch", "checkpoint_sha256", "rank1_percent", "rank5_percent", "rank10_percent", "map_percent", "metric_source"],
           [{"experiment": "G2-D1", "checkpoint_epoch": result["selected_checkpoint"]["epoch"],
             "checkpoint_sha256": selected_sha, "rank1_percent": metrics["rank1_percent"],
             "rank5_percent": metrics["rank5_percent"], "rank10_percent": metrics["rank10_percent"],
             "map_percent": metrics["map_percent"], "metric_source": "machine-selected validation_history.jsonl record"}])
    _atomic_text(package_dir / "code_diff_from_g2a_tau0p5.patch", _diff(
        "d724a6536e4a819c5d2932412e90b7dea224041b", result["commit"]))
    manifest = {
        "experiment": "G2-D1", "baseline_branch": "codex/g2-global-local-gating-tau0p5",
        "baseline_commit": "d724a6536e4a819c5d2932412e90b7dea224041b", "training_commit": result["commit"],
        "branch": result["branch"], "dataset": "market1501", "seed": 42,
        "gating_input": result["gating_input"], "gating_input_mode": result["gating_input_mode"],
        "controller_input_dim": result["controller_input_dim"], "controller_parameter_count": result["controller_parameter_count"],
        "retrieval_feature_dim": result["retrieval_feature_dim"], "delta46_definition": result["delta46_definition"],
        "gating_temperature": result["gating_temperature"], "selected_checkpoint": result["selected_checkpoint"],
        "source_config_sha256": _sha(source_config), "resolved_config_sha256": _sha(resolved_config),
        "reproducibility_path": str(reproducibility_path), "reproducibility_sha256": _sha(reproducibility_path),
        "run_command": reproducibility.get("command", "not_recorded"),
        "environment": reproducibility.get("environment", "not_recorded"),
        "gate_sample_protocol": analysis["test_weight_protocol"],
        "gate_sample_count": len(packaged), "full_retrieval_metric_protocol": "selected full validation record",
    }
    _atomic_text(package_dir / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_text(package_dir / "README.md", """# G2-D1 formal result package

This package is generated only after a formal, seed-42 G2-D1 run selects a checkpoint from the full validation history. Relative to direct baseline G2-A, it changes only the controller input from `[g,z2,z4,z6]` to `[g,z2,z4,z6,abs(z4-z6)]`, with `tau_g=0.5` unchanged. The controller is linear `3072 -> 3`; `abs(z4-z6)` receives no extra gate and does not enter the 2816-dimensional retrieval descriptor.

`retrieval_metrics.csv` is full formal validation. Gate statistics, distributions and examples use the bounded deterministic sample set explicitly recorded in `run_manifest.json`; they are not a substitute for full retrieval evaluation. `w=3p`, and dominant scale is `torch.argmax(p2,p4,p6)`.
""")
    files = sorted(path for path in package_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    _atomic_text(package_dir / "SHA256SUMS", "".join("{}  {}\n".format(_sha(path), path.relative_to(package_dir).as_posix()) for path in files))
    return package_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv); print(str(package(args.output_dir))); return 0


if __name__ == "__main__":
    raise SystemExit(main())
