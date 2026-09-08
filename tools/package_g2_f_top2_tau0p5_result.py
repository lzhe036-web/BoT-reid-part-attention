#!/usr/bin/env python
"""Build a self-contained, hash-checked G2-F hard Top-2 evidence package."""

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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.datasets import init_dataset
from tools.analyze_dynamic_gating import _stable_key
from utils.config_serialization import deserialize_cfg_node_yaml


PACKAGE_NAME = "g2_f_top2_tau0p5_result_package"
RESULT_NAME = "g2_f_top2_tau0p5_formal_result.json"
BASE_COMMIT = "d724a6536e4a819c5d2932412e90b7dea224041b"
SCALES = (2, 4, 6)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}".format(path.name, uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write(path, fields, rows, delimiter=","):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=delimiter,
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    _atomic_text(path, buffer.getvalue())


def _require(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError("Required {} is missing or empty: {}".format(label, path))
    return path


def _read_json(path, label):
    try:
        return json.loads(_require(path, label).read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError("Invalid {}: {}".format(label, error))


def _rows(path):
    with _require(path, "per-sample gate table").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("G2-F per-sample gate table is empty")
    return rows


def _validate_rows(rows, checkpoint_sha):
    required = {
        "stable_sample_key", "dataset_split", "pid", "camid", "p_dense2", "p_dense4", "p_dense6",
        "p_top2_2", "p_top2_4", "p_top2_6", "w2", "w4", "w6", "mask2", "mask4", "mask6",
        "top2_combination", "disabled_scale", "dominant_k", "selection_boundary_tie", "checkpoint_sha256",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("G2-F per-sample gate table has an invalid schema")
    seen = set()
    for row in rows:
        if row["stable_sample_key"] in seen:
            raise ValueError("G2-F gate table repeats a stable sample key")
        seen.add(row["stable_sample_key"])
        if row["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("G2-F sample table is bound to another checkpoint")
        dense = [float(row["p_dense{}".format(scale)]) for scale in SCALES]
        sparse = [float(row["p_top2_{}".format(scale)]) for scale in SCALES]
        weights = [float(row["w{}".format(scale)]) for scale in SCALES]
        mask = [int(row["mask{}".format(scale)]) for scale in SCALES]
        if not all(math.isfinite(value) and value >= 0.0 for value in dense + sparse + weights):
            raise ValueError("G2-F sample table contains invalid gate values")
        if not math.isclose(sum(dense), 1.0, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("G2-F dense probabilities do not sum to one")
        if not math.isclose(sum(sparse), 1.0, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("G2-F Top-2 probabilities do not sum to one")
        if not math.isclose(sum(weights), 3.0, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("G2-F weights do not sum to three")
        if mask.count(1) != 2 or any(value not in (0, 1) for value in mask):
            raise ValueError("G2-F sample does not contain exactly two selected scales")
        if any(sparse[index] != 0.0 for index, enabled in enumerate(mask) if not enabled):
            raise ValueError("G2-F unselected sparse probability is not exactly zero")
        if any(not math.isclose(weight, 3.0 * probability, rel_tol=1e-6, abs_tol=1e-7)
               for weight, probability in zip(weights, sparse)):
            raise ValueError("G2-F sample violates w=3*p_top2")
        selected = [scale for index, scale in enumerate(SCALES) if mask[index]]
        expected_combo = "+".join("K{}".format(scale) for scale in selected)
        disabled = [scale for scale in SCALES if scale not in selected]
        if row["top2_combination"] != expected_combo or row["disabled_scale"] != "K{}".format(disabled[0]):
            raise ValueError("G2-F Top-2 combination metadata is inconsistent")


def _image_mapping(configuration, rows):
    dataset_root = configuration["DATASETS"]["ROOT_DIR"]
    dataset = init_dataset(configuration["DATASETS"]["NAMES"], root=dataset_root, verbose=False)
    targets = {row["stable_sample_key"] for row in rows}
    mapping = {}
    for split, entries in (("query", dataset.query), ("gallery", dataset.gallery)):
        for image_path, pid, camid in entries:
            key = _stable_key(split, image_path, pid, camid, dataset_root)
            if key not in targets:
                continue
            if key in mapping:
                raise ValueError("Stable hash maps to more than one dataset image")
            relative = Path(image_path).resolve().relative_to(Path(dataset_root).resolve()).as_posix()
            mapping[key] = (split, relative, int(pid), int(camid), Path(image_path))
    missing = targets - set(mapping)
    if missing:
        raise ValueError("Cannot recover {} G2-F sample image path(s)".format(len(missing)))
    return mapping


def _plot_examples(rows, mapping, scale, output_base, images_dir):
    selected = sorted((row for row in rows if int(row["dominant_k"]) == scale),
                      key=lambda row: row["stable_sample_key"])[:6]
    figure, axes = plt.subplots(1, max(1, len(selected)), figsize=(2.1 * max(1, len(selected)), 4.0), dpi=180)
    axes = [axes] if len(selected) <= 1 else list(axes)
    if not selected:
        axes[0].axis("off")
        axes[0].text(.5, .5, "No K{} dominant samples\n(observed zero)".format(scale),
                     ha="center", va="center")
    for position, row in enumerate(selected):
        _split, relative, _pid, _camid, source = mapping[row["stable_sample_key"]]
        target = images_dir / "K{}_{}_{}".format(scale, position + 1, Path(relative).name)
        shutil.copyfile(str(source), str(target))
        with Image.open(str(source)) as image:
            axes[position].imshow(image.convert("RGB"))
        axes[position].axis("off")
        axes[position].set_title(
            "{}\n{}\np=({:.3f},{:.3f},{:.3f})\n{}; off {}".format(
                Path(relative).name, row["top2_combination"],
                float(row["p_top2_2"]), float(row["p_top2_4"]), float(row["p_top2_6"]),
                "K{}".format(row["dominant_k"]), row["disabled_scale"]), fontsize=6)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(output_base) + "." + suffix, bbox_inches="tight")
    plt.close(figure)
    return selected


def _diff(commit):
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "diff", "--binary",
                                    "{}..{}".format(BASE_COMMIT, commit)],
                                   stderr=subprocess.STDOUT).decode("utf-8", errors="replace")


def package(output_dir, package_dir=None):
    output_dir = Path(output_dir).resolve()
    result = _read_json(output_dir / RESULT_NAME, "G2-F formal result")
    selected = result.get("selected_checkpoint", {})
    checkpoint_sha = selected.get("sha256")
    if result.get("branch") != "codex/g2-f-top2-tau0p5" or result.get("seed") != 42:
        raise ValueError("G2-F formal result identity is invalid")
    if result.get("gate_sparsification") != "topk" or result.get("gate_topk") != 2 \
            or result.get("gate_tie_break") != "scale_order" or result.get("weight_scale") != 3.0:
        raise ValueError("G2-F hard Top-2 result identity is invalid")
    analysis = _read_json(result["evidence"]["analysis_manifest"], "G2-F analysis manifest")
    if _sha256(result["evidence"]["analysis_manifest"]) != result["evidence"]["analysis_manifest_sha256"]:
        raise ValueError("G2-F analysis manifest SHA256 mismatch")
    raw = _rows(analysis["files"]["per_sample_gating_tsv"]["path"])
    _validate_rows(raw, checkpoint_sha)
    resolved = _require(output_dir / "config_resolved.yml", "resolved config")
    configuration = deserialize_cfg_node_yaml(resolved.read_text(encoding="utf-8"))
    if configuration["MODEL"]["MULTI_GRANULARITY_GATING_SPARSIFICATION"] != "topk":
        raise ValueError("Resolved config does not identify G2-F Top-2")
    mapping = _image_mapping(configuration, raw)
    package_dir = Path(package_dir).resolve() if package_dir else output_dir / PACKAGE_NAME
    if package_dir.exists():
        raise FileExistsError("Refusing to overwrite package directory: {}".format(package_dir))
    package_dir.mkdir(parents=True)
    figures, images, scripts = package_dir / "figures", package_dir / "sample_images", package_dir / "scripts"
    figures.mkdir(); images.mkdir(); scripts.mkdir()
    source_config = _require(result["evidence"]["config"], "source config")
    formal = _require(output_dir / RESULT_NAME, "formal result")
    reproducibility = _require(output_dir / "reproducibility.json", "reproducibility")
    for source, target in ((source_config, package_dir / "config_source.yml"),
                           (resolved, package_dir / "config_resolved.yml"),
                           (formal, package_dir / "formal_result.json"),
                           (Path(result["evidence"]["analysis_manifest"]), package_dir / "analysis_manifest.json"),
                           (reproducibility, package_dir / "reproducibility.json")):
        shutil.copyfile(str(source), str(target))
    for source in (REPO_ROOT / "tools" / "analyze_g2_f_top2_tau0p5.py",
                   REPO_ROOT / "tools" / "package_g2_f_top2_tau0p5_result.py",
                   REPO_ROOT / "tools" / "compare_g1_g2_g2a_g2f_gating.py",
                   REPO_ROOT / "scripts" / "export_g2_f_top2_tau0p5_result_autodl.sh"):
        shutil.copyfile(str(_require(source, "reproducibility script")), str(scripts / source.name))
    enriched = []
    for row in raw:
        split, relative, pid, camid, _source = mapping[row["stable_sample_key"]]
        item = dict(row)
        item.update({"version": "G2-F-Top2-tau0p5", "run_id": result["commit"][:12],
                     "split": split, "relative_image_path": relative, "pid": pid, "camid": camid})
        enriched.append(item)
    fields = ["version", "run_id", "stable_sample_key", "relative_image_path", "split", "pid", "camid"] + list(PER_SAMPLE_FIELDS[4:])
    _write(package_dir / "per_sample_gating.tsv", fields, enriched, delimiter="\t")
    _write(package_dir / "sample_manifest.tsv", fields, enriched, delimiter="\t")
    _write(package_dir / "sample_path_mapping.tsv",
           ["stable_sample_key", "relative_image_path", "split", "pid", "camid", "mapping_status"],
           [{"stable_sample_key": row["stable_sample_key"], "relative_image_path": row["relative_image_path"],
             "split": row["split"], "pid": row["pid"], "camid": row["camid"],
             "mapping_status": "reconstructed_from_configured_dataset_and_hashed_key"} for row in enriched], delimiter="\t")
    for name in ("gate_statistics_csv", "top2_combination_statistics_csv"):
        source = _require(analysis["files"][name]["path"], name)
        shutil.copyfile(str(source), str(package_dir / ("gate_statistics.csv" if name.startswith("gate_") else "top2_combination_statistics.csv")))
    for stem, source_key in (("g2_f_top2_weight_distribution", "g2_f_top2_weight_distribution"),
                             ("g2_f_top2_combination_frequency", "g2_f_top2_combination_frequency")):
        for suffix in ("png", "pdf"):
            source = _require(analysis["files"][source_key + "_" + suffix]["path"], source_key + suffix)
            shutil.copyfile(str(source), str(figures / (stem + "." + suffix)))
    selected_examples = []
    for scale in (4, 6):
        selected_examples.extend(_plot_examples(enriched, mapping, scale,
                                                figures / "k{}_dominant_examples".format(scale), images))
    _write(package_dir / "sample_selection.csv",
           ["selection_rule", "dominant_k", "stable_sample_key", "top2_combination", "disabled_scale",
            "p_top2_2", "p_top2_4", "p_top2_6", "w2", "w4", "w6"],
           [{"selection_rule": "stable_sample_key ascending; first six within actual dominant class",
             **{key: row[key] for key in ("dominant_k", "stable_sample_key", "top2_combination", "disabled_scale",
                                           "p_top2_2", "p_top2_4", "p_top2_6", "w2", "w4", "w6")}}
            for row in selected_examples])
    metrics = result["metrics"]
    _write(package_dir / "retrieval_metrics.csv",
           ["experiment", "dataset", "checkpoint_epoch", "checkpoint_sha256", "rank1_percent", "rank5_percent", "rank10_percent", "map_percent", "metric_source"],
           [{"experiment": "G2-F-Top2-tau0p5", "dataset": "market1501",
             "checkpoint_epoch": selected["epoch"], "checkpoint_sha256": checkpoint_sha,
             **metrics, "metric_source": "machine-selected full validation_history.jsonl record"}])
    _atomic_text(package_dir / "code_diff_from_g2a_tau0p5.patch", _diff(result["commit"]))
    reproducibility_record = _read_json(reproducibility, "reproducibility")
    manifest = {"experiment": "G2-F-Top2-tau0p5", "baseline_branch": "codex/g2-global-local-gating-tau0p5",
                "baseline_commit": BASE_COMMIT, "training_commit": result["commit"], "branch": result["branch"],
                "dataset": "market1501", "seed": 42, "gating_input": result["gating_input"],
                "gating_input_dim": 2816, "controller": "Linear(2816,3)", "gating_temperature": .5,
                "gating_normalization": "scaled_softmax", "gate_sparsification": "topk", "gate_topk": 2,
                "gate_tie_break": "scale_order", "initial_zero_controller_selection": "K2+K4", "weight_scale": 3.0,
                "descriptor_dim": 2816, "selected_checkpoint": selected, "gate_sample_count": len(enriched),
                "selection_boundary_tie_ratio": result["top2_selection_boundary_tie_ratio"],
                "source_config_sha256": _sha256(source_config), "resolved_config_sha256": _sha256(resolved),
                "run_command": reproducibility_record.get("command", "not_recorded"),
                "environment": reproducibility_record.get("environment", "not_recorded")}
    _atomic_text(package_dir / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_text(package_dir / "README.md", """# G2-F-Top2-tau0p5 formal evidence

This package contains evidence from one selected formal Market1501 checkpoint. It inherits G2-A (`tau_g=0.5`) and changes only dense scaled-softmax gating to hard Top-2: `p_dense=softmax(logits/0.5)`, select two scales by stable K2/K4/K6 order, `p_top2=softmax(masked_logits)`, and `w=3*p_top2`. Thus each sample keeps exactly two local 256-dimensional blocks and zero-fills the disabled block. The scale factor stays 3, not 2.

`p_dense` is recorded only as pre-sparsification evidence. `p_top2` and `w` are the actual fusion values. Initial zero logits select K2+K4 due to the declared stable tie rule. Retrieval metrics are full machine-selected validation metrics; deterministic selected samples serve only gate statistics and figures. K4/K6 examples are the first six actual dominant rows in stable-key order; an observed zero is never substituted.
""")
    files = sorted(path for path in package_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    _atomic_text(package_dir / "SHA256SUMS", "".join("{}  {}\n".format(
        _sha256(path), path.relative_to(package_dir).as_posix()) for path in files))
    return package_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-dir", default=None)
    args = parser.parse_args(argv)
    print(str(package(args.output_dir, args.package_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
