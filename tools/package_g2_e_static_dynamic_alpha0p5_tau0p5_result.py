#!/usr/bin/env python
"""Build a self-contained, hash-checked G2-E formal result package."""

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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.datasets import init_dataset
from tools.analyze_dynamic_gating import _stable_key
from utils.config_serialization import deserialize_cfg_node_yaml
from utils.g2_e_alpha_delivery import require_registered, copy_registry_evidence, validate_raw


PACKAGE_NAME = "g2_e_static_dynamic_alpha0p5_tau0p5_result_package"
RESULT_NAME = "g2_e_static_dynamic_alpha0p5_tau0p5_formal_result.json"
BASE_COMMIT = "63761021a40693694f037d850066deb2237a5c41"
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
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write(path, fields, rows, delimiter=","):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=delimiter,
                            lineterminator="\n")
    writer.writeheader()
    # Analyzer rows preserve provenance-only columns such as ``dataset_split``
    # and ``entropy``.  Delivery tables have their own explicit schema (for
    # example, the canonical ``split`` reconstructed from the stable key), so
    # project each row onto that schema before serializing.  Indexing rather
    # than ``get`` deliberately remains fail-closed for a missing required
    # delivery field while harmless source-only columns cannot break export.
    writer.writerows({field: row[field] for field in fields} for row in rows)
    _atomic_text(path, buffer.getvalue())


def _json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_samples(rows, configuration):
    """Recover file paths only by reproducing the original SHA256 key function."""
    keys = [str(row.get("stable_sample_key", "")) for row in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError("Per-sample evidence is empty or has duplicate stable keys")
    if any(len(key) != 64 or any(char not in "0123456789abcdef" for char in key)
           for key in keys):
        raise ValueError("G2-E package requires SHA256 stable sample keys")
    targets = {row["stable_sample_key"]: row for row in rows}
    dataset_root = Path(str(configuration["DATASETS"]["ROOT_DIR"])).resolve()
    dataset = init_dataset(configuration["DATASETS"]["NAMES"], root=str(dataset_root),
                           verbose=False)
    mapping = {}
    for split, entries in (("query", dataset.query), ("gallery", dataset.gallery)):
        for image_path, pid, camid in entries:
            key = _stable_key(split, image_path, pid, camid, str(dataset_root))
            if key not in targets:
                continue
            if key in mapping:
                raise ValueError("Stable key maps to multiple images: {}".format(key))
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError("Mapped selected image is absent: {}".format(path))
            evidence = targets[key]
            if (evidence["dataset_split"] != split or int(evidence["pid"]) != int(pid)
                    or int(evidence["camid"]) != int(camid)):
                raise ValueError("Stable key metadata mismatch: {}".format(key))
            mapping[key] = (split, path.resolve().relative_to(dataset_root).as_posix(),
                            int(pid), int(camid))
    missing = sorted(set(targets).difference(mapping))
    if missing:
        raise ValueError("Cannot map all selected samples; first={}".format(missing[0]))
    return mapping, dataset_root


def _plot_distribution(rows, output_base):
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=180)
    for scale in SCALES:
        axes[0].hist([float(row["w{}".format(scale)]) for row in rows],
                     bins=np.linspace(0, 3, 31), histtype="step", linewidth=1.7,
                     label="w{}=3p{}".format(scale, scale))
        axes[1].hist([float(row["c{}".format(scale)]) for row in rows],
                     bins=30, histtype="step", linewidth=1.7,
                     label="c{}=1+0.5w{}".format(scale, scale))
    axes[0].set(title="G2-E residual-branch weights", xlabel="w=3p", ylabel="sample count",
                xlim=(0, 3))
    axes[1].set(title="G2-E final local coefficients", xlabel="c=1+0.5w",
                ylabel="sample count")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(Path(str(output_base) + "." + suffix)), bbox_inches="tight")
    plt.close(figure)


def _plot_examples(rows, dataset_root, scale, output_base, images_dir):
    selected = sorted((row for row in rows if int(row["dominant_k"]) == scale),
                      key=lambda row: row["stable_sample_key"])[:6]
    figure, axes = plt.subplots(1, max(1, len(selected)),
                                figsize=(2.5 * max(1, len(selected)), 4.5), dpi=180)
    axes = np.atleast_1d(axes)
    for axis in axes:
        axis.axis("off")
    if not selected:
        axes[0].text(.5, .5, "0 K{}-dominant samples observed".format(scale),
                     ha="center", va="center")
    for index, (axis, row) in enumerate(zip(axes, selected)):
        source = dataset_root / row["relative_image_path"]
        if not source.is_file():
            raise FileNotFoundError("Selected sample vanished: {}".format(source))
        target = images_dir / "K{}_{}_{}".format(scale, index + 1, source.name)
        shutil.copyfile(str(source), str(target))
        with Image.open(str(source)) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title("{}\np=({:.5f},{:.5f},{:.5f})\nc=({:.5f},{:.5f},{:.5f})".format(
            source.name, float(row["p2"]), float(row["p4"]), float(row["p6"]),
            float(row["c2"]), float(row["c4"]), float(row["c6"])), fontsize=6.5)
    figure.suptitle("G2-E K{}-dominant samples; stable-key ascending".format(scale))
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(Path(str(output_base) + "." + suffix)), bbox_inches="tight")
    plt.close(figure)
    return selected


def _diff(commit):
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", BASE_COMMIT, commit, "--"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return completed.stdout.decode("utf-8", errors="replace")


def _require(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError("Required {} is absent or empty: {}".format(label, path))
    return path


def package(output_dir, package_dir=None, records_root=None, experiments_path=None):
    output_dir = Path(output_dir).resolve()
    result_path = _require(output_dir / RESULT_NAME, "formal result")
    result = _json(result_path)
    if (result.get("fusion_mode") != "static_concat_plus_gated_residual"
            or float(result.get("static_dynamic_alpha", -1)) != 0.5
            or float(result.get("gating_temperature", -1)) != 0.5):
        raise ValueError("Formal result is not G2-E alpha=.5 tau_g=.5")
    records_root = Path(records_root or REPO_ROOT / "experiment_records").resolve()
    experiments_path = Path(experiments_path or REPO_ROOT / "EXPERIMENTS.md").resolve()
    run_dir = require_registered(result, records_root)
    validate_raw(result)
    evidence = result.get("evidence", {})
    source_config = _require(evidence.get("config", ""), "source config")
    resolved_config = _require(output_dir / "config_resolved.yml", "resolved config")
    reproducibility = _require(output_dir / "reproducibility.json", "reproducibility")
    reproducibility_record = _json(reproducibility)
    analysis_manifest = _require(evidence.get("analysis_manifest", ""), "analysis manifest")
    if _sha256(analysis_manifest) != evidence.get("analysis_manifest_sha256"):
        raise ValueError("Analysis manifest SHA256 does not match final result")
    analysis = _json(analysis_manifest)
    per_sample_source = _require(analysis["files"]["per_sample_gating_tsv"]["path"],
                                 "per-sample gating")
    if _sha256(per_sample_source) != analysis["files"]["per_sample_gating_tsv"]["sha256"]:
        raise ValueError("Per-sample gating SHA256 does not match analysis manifest")
    with per_sample_source.open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    selected_sha = result["selected_checkpoint"]["sha256"]
    if any(row.get("checkpoint_sha256") != selected_sha for row in raw):
        raise ValueError("Per-sample evidence is not bound to selected checkpoint")
    for row in raw:
        probabilities = [float(row["p{}".format(scale)]) for scale in SCALES]
        weights = [float(row["w{}".format(scale)]) for scale in SCALES]
        coefficients = [float(row["c{}".format(scale)]) for scale in SCALES]
        if (not all(math.isfinite(value) and value >= 0.0
                    for value in probabilities + weights + coefficients)
                or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-8)
                or any(not math.isclose(weight, 3.0 * probability, rel_tol=1e-6,
                                         abs_tol=1e-8)
                       for weight, probability in zip(weights, probabilities))
                or any(not math.isclose(coefficient, 1.0 + 0.5 * weight,
                                         rel_tol=1e-6, abs_tol=1e-8)
                       for coefficient, weight in zip(coefficients, weights))
                or not math.isclose(float(row["alpha"]), 0.5,
                                    rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError("Per-sample p/w/c evidence violates G2-E semantics")
    resolved = deserialize_cfg_node_yaml(resolved_config.read_text(encoding="utf-8"))
    if _sha256(source_config) != evidence["config_sha256"]:
        raise ValueError("Source config changed since formal registration")
    from utils.g2_e_checkpoint_identity import validate
    validate(result["selected_checkpoint"]["path"], resolved)
    mapping, dataset_root = _resolve_samples(raw, resolved)
    package_dir = output_dir / PACKAGE_NAME if package_dir is None else Path(package_dir).resolve()
    if package_dir.parent != output_dir:
        raise ValueError("Package directory must be a direct OUTPUT_DIR child")
    if package_dir.exists():
        raise FileExistsError("Refusing to overwrite package: {}".format(package_dir))
    figures, images, scripts = (package_dir / "figures", package_dir / "sample_images",
                                package_dir / "scripts")
    figures.mkdir(parents=True)
    images.mkdir()
    scripts.mkdir()
    for source, target in ((source_config, package_dir / "config_source.yml"),
                           (resolved_config, package_dir / "config_resolved.yml"),
                           (result_path, package_dir / "formal_result.json"),
                           (analysis_manifest, package_dir / "analysis_manifest.json"),
                           (reproducibility, package_dir / "reproducibility.json")):
        shutil.copyfile(str(source), str(target))
    for source in (
            REPO_ROOT / "tools" / "package_g2_e_static_dynamic_alpha0p5_tau0p5_result.py",
            REPO_ROOT / "tools" / "analyze_g2_e_static_dynamic_alpha0p5_tau0p5.py",
            REPO_ROOT / "scripts" / "export_g2_e_static_dynamic_alpha0p5_tau0p5_result_autodl.sh",
    ):
        _require(source, "reproducibility script")
        shutil.copyfile(str(source), str(scripts / source.name))
    rows = []
    for row in raw:
        split, relative, pid, camid = mapping[row["stable_sample_key"]]
        enriched = dict(row)
        enriched.update({"version": "G2-E-alpha0p5-tau0p5", "split": split,
                         "relative_image_path": relative, "pid": pid, "camid": camid,
                         "run_id": run_dir.name})
        rows.append(enriched)
    fields = ["version", "run_id", "stable_sample_key", "relative_image_path", "split", "pid", "camid",
              "p2", "p4", "p6", "w2", "w4", "w6", "residual2", "residual4", "residual6",
              "c2", "c4", "c6", "alpha", "dominant_k", "checkpoint_sha256"]
    _write(package_dir / "per_sample_gating.tsv", fields, rows, delimiter="\t")
    _write(package_dir / "sample_manifest.tsv", fields, rows, delimiter="\t")
    _write(package_dir / "sample_path_mapping.tsv",
           ["stable_sample_key", "relative_image_path", "split", "pid", "camid", "mapping_status"],
           [{"stable_sample_key": row["stable_sample_key"],
             "relative_image_path": row["relative_image_path"], "split": row["split"],
             "pid": row["pid"], "camid": row["camid"],
             "mapping_status": "reconstructed_from_configured_dataset_and_hashed_key"}
            for row in rows], delimiter="\t")
    statistics = _require(analysis["files"]["gate_statistics_csv"]["path"], "gate statistics")
    shutil.copyfile(str(statistics), str(package_dir / "gate_statistics.csv"))
    _plot_distribution(rows, figures / "g2_e_gate_distributions")
    selected = []
    for scale in (4, 6):
        selected.extend(_plot_examples(rows, dataset_root, scale,
                                       figures / "k{}_dominant_examples".format(scale), images))
    _write(package_dir / "sample_selection.csv",
           ["selection_rule", "dominant_k", "stable_sample_key", "p2", "p4", "p6", "w2", "w4", "w6", "c2", "c4", "c6"],
           [{"selection_rule": "stable_sample_key ascending; first six within dominant class",
             **{key: row[key] for key in ("dominant_k", "stable_sample_key", "p2", "p4", "p6", "w2", "w4", "w6", "c2", "c4", "c6")}}
            for row in selected])
    from tools.plot_g2_e_alpha_results import plot
    plot(package_dir, figures)
    shutil.copyfile(str(Path(str(result["selected_checkpoint"]["path"])+".metadata.json")),
                    str(package_dir/"selected_checkpoint.metadata.json"))
    for script in ("finish_g2_e_alpha_experiment.py", "recover_g2_e_static_dynamic_alpha0p5_tau0p5_experiment.py"):
        shutil.copyfile(str(REPO_ROOT/"tools"/script), str(scripts/script))
    metrics = result["metrics"]
    _write(package_dir / "retrieval_metrics.csv",
           ["experiment", "dataset", "checkpoint_epoch", "checkpoint_sha256", "rank1_percent", "rank5_percent", "rank10_percent", "map_percent", "metric_source"],
           [{"experiment": "G2-E-alpha0p5-tau0p5", "dataset": "market1501",
             "checkpoint_epoch": result["selected_checkpoint"]["epoch"],
             "checkpoint_sha256": selected_sha, **metrics,
             "metric_source": "machine-selected full validation_history.jsonl record"}])
    _atomic_text(package_dir / "code_diff_from_alpha0p3.patch", _diff(result["commit"]))
    run_manifest = {
        "experiment": "G2-E-alpha0p5-tau0p5", "baseline_branch": "codex/g2-e-static-dynamic-alpha0p3-tau0p5",
        "baseline_commit": BASE_COMMIT, "training_commit": result["commit"], "branch": result["branch"],
        "dataset": "market1501", "seed": 42, "gating_input": result["gating_input"],
        "gating_input_dim": result["gating_input_dim"], "controller": result["controller"],
        "controller_parameter_count": result["controller_parameter_count"],
        "gating_temperature": .5, "gating_normalization": "scaled_softmax",
        "fusion_mode": result["fusion_mode"], "static_dynamic_alpha": .5,
        "fusion_formula": result["fusion_formula"], "descriptor_dim": 2816,
        "selected_checkpoint": result["selected_checkpoint"],
        "source_config_sha256": _sha256(source_config), "resolved_config_sha256": _sha256(resolved_config),
        "reproducibility_sha256": _sha256(reproducibility), "gate_sample_count": len(rows),
        "gate_sample_scope": analysis.get("test_weight_protocol", "recorded in analysis manifest"),
        "run_command": reproducibility_record.get("command", "not_recorded"),
        "environment": reproducibility_record.get("environment", "not_recorded"),
    }
    _atomic_text(package_dir / "run_manifest.json", json.dumps(run_manifest, ensure_ascii=False,
                                                                indent=2, sort_keys=True) + "\n")
    _atomic_text(package_dir / "README.md", """# G2-E-alpha0p5-tau0p5 formal evidence

This package was generated from one selected formal Market1501 checkpoint. It inherits G2-E alpha=0.3 (`tau_g=0.5`) and changes only alpha in the existing fusion: static `concat(g,z2,z4,z6)` plus residual `0.5 * concat(0,w2*z2,w4*z4,w6*z6)`. Thus the final local coefficients are `c_k=1+0.5*w_k`; `p` is a probability, `w=3p` is the residual-branch weight, and `c` is not a probability.

Retrieval metrics are full machine-selected validation metrics. Per-sample and figure statistics use the deterministic bounded selected-checkpoint sample set recorded in `run_manifest.json`, not a substitute for the full retrieval evaluator. K4/K6 examples are the first six matching rows in stable-key order; a zero-class figure is an observed zero, not a substitution.
""")
    copy_registry_evidence(run_dir, records_root, experiments_path, package_dir)
    for filename in ("plot_g2_e_alpha_results.py", "compare_g2_e_alpha_sweep.py"):
        shutil.copyfile(str(REPO_ROOT/"tools"/filename), str(scripts/filename))
    files = sorted(path for path in package_dir.rglob("*")
                   if path.is_file() and path.name != "SHA256SUMS")
    _atomic_text(package_dir / "SHA256SUMS", "".join(
        "{}  {}\n".format(_sha256(path), path.relative_to(package_dir).as_posix())
        for path in files
    ))
    return package_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-dir", default=None)
    parser.add_argument("--records-root", default=None)
    parser.add_argument("--experiments-path", default=None)
    args = parser.parse_args(argv)
    print(str(package(args.output_dir, args.package_dir, args.records_root, args.experiments_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
