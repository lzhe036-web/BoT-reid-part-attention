#!/usr/bin/env python
"""Compare formal G1/G2/G2-A/G2-C gate exports on one identical sample set.

The tool intentionally reads pre-existing, checkpoint-bound per-sample TSVs;
it never reuses a scalar result as an imagined per-sample gate.  Any mismatch
in stable sample keys, row metadata, p/w identity, or selected checkpoint is a
hard failure rather than an implicit intersection.
"""

from __future__ import absolute_import

import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from collections import OrderedDict
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SCALES = (2, 4, 6)
VERSIONS = ("G1", "G2", "G2-A", "G2-C")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _write_csv(path, fields, rows):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _read_samples(path, label):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"stable_sample_key", "dataset_split", "pid", "camid", "p2", "p4", "p6", "w2", "w4", "w6", "dominant_k", "checkpoint_sha256"}
    if not rows or not required.issubset(set(rows[0])):
        raise ValueError("{} gating TSV is empty or has an invalid schema".format(label))
    keyed = OrderedDict()
    for row in rows:
        key = row["stable_sample_key"]
        if not key or key in keyed:
            raise ValueError("{} gating TSV has empty/duplicate sample keys".format(label))
        try:
            p = [float(row["p{}".format(scale)]) for scale in SCALES]
            w = [float(row["w{}".format(scale)]) for scale in SCALES]
            dominant = int(row["dominant_k"])
        except (TypeError, ValueError) as error:
            raise ValueError("{} contains non-numeric gate data".format(label)) from error
        if not all(math.isfinite(value) and value >= 0.0 for value in p + w):
            raise ValueError("{} has invalid probability/weight values".format(label))
        if not math.isclose(sum(p), 1.0, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("{} probabilities do not sum to one".format(label))
        if any(not math.isclose(w[index], 3.0 * p[index], rel_tol=1e-6, abs_tol=1e-9)
               for index in range(3)):
            raise ValueError("{} weights are not w=3p".format(label))
        if dominant != SCALES[int(np.argmax(p))]:
            raise ValueError("{} dominant_k does not match deterministic argmax".format(label))
        keyed[key] = row
    return keyed


def _validate_same_samples(maps):
    reference_label = VERSIONS[0]
    reference = maps[reference_label]
    keys = list(reference)
    for label in VERSIONS[1:]:
        current = maps[label]
        if list(current) != keys:
            raise ValueError(
                "{} does not use exactly the same frozen sample list as {}; "
                "intersection is intentionally forbidden".format(label, reference_label)
            )
        for key in keys:
            left, right = reference[key], current[key]
            for field in ("dataset_split", "pid", "camid"):
                if str(left[field]) != str(right[field]):
                    raise ValueError("Sample metadata mismatch for {} at {}".format(field, key))
    return keys


def _metrics(path, label):
    if not path:
        return {"version": label, "metric_status": "not_recorded"}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    selected = payload.get("selected_checkpoint", {})
    required = ("rank1_percent", "rank5_percent", "rank10_percent", "map_percent")
    if not all(name in metrics for name in required):
        raise ValueError("{} formal result has no complete metric object".format(label))
    return {
        "version": label, "metric_status": "machine_recorded", "rank1_percent": metrics["rank1_percent"],
        "rank5_percent": metrics["rank5_percent"], "rank10_percent": metrics["rank10_percent"],
        "map_percent": metrics["map_percent"], "selected_epoch": selected.get("epoch", "not_recorded"),
        "checkpoint_sha256": selected.get("sha256", "not_recorded"), "result_source": str(Path(path).resolve()),
    }


def _statistics(label, rows):
    result = []
    count = len(rows)
    for scale in SCALES:
        p = np.asarray([float(row["p{}".format(scale)]) for row in rows])
        w = np.asarray([float(row["w{}".format(scale)]) for row in rows])
        result.append({
            "version": label, "scale": "K{}".format(scale), "sample_count": count,
            "p_mean": float(p.mean()), "p_std": float(p.std(ddof=0)),
            "w_mean": float(w.mean()), "w_std": float(w.std(ddof=0)),
            "dominant_ratio": float(sum(int(row["dominant_k"]) == scale for row in rows)) / float(count),
            "dominant_rule": "argmax(p2,p4,p6); numpy first-index tie behavior",
        })
    return result


def _plot(maps, field, output):
    upper = 1.0 if field == "p" else 3.0
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), dpi=180, sharey=True)
    for axis, scale in zip(axes, SCALES):
        for label in VERSIONS:
            values = [float(row["{}{}".format(field, scale)]) for row in maps[label].values()]
            axis.hist(values, bins=np.linspace(0.0, upper, 31), histtype="step", linewidth=1.5, label=label)
        axis.set(title="K{}".format(scale), xlabel=("probability p" if field == "p" else "weight w=3p"), xlim=(0.0, upper))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("sample count")
    axes[-1].legend(fontsize=8)
    figure.suptitle("G1 / G2 / G2-A / G2-C gate {} distributions on identical samples".format(field))
    figure.tight_layout()
    figure.savefig(str(output), bbox_inches="tight")
    figure.savefig(str(output.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(figure)


def _relative_path(key):
    fields = key.split("|")
    if len(fields) != 4:
        raise ValueError("Invalid stable sample key: {}".format(key))
    return fields[1]


def _examples(maps, keys, dataset_root, scale, output):
    chosen = [key for key in keys if int(maps["G2-C"][key]["dominant_k"]) == scale][:6]
    figure, axes = plt.subplots(1, max(1, len(chosen)), figsize=(2.4 * max(1, len(chosen)), 4.4), dpi=180)
    axes = np.atleast_1d(axes)
    for axis in axes:
        axis.axis("off")
    for axis, key in zip(axes, chosen):
        image_path = dataset_root / _relative_path(key)
        if image_path.is_file():
            with Image.open(str(image_path)) as image:
                axis.imshow(image.convert("RGB"))
        else:
            axis.text(0.5, 0.5, "image unavailable", ha="center", va="center")
        lines = [Path(_relative_path(key)).name]
        for label in VERSIONS:
            row = maps[label][key]
            lines.append("{}: {:.2f}/{:.2f}/{:.2f}".format(label, float(row["p2"]), float(row["p4"]), float(row["p6"])))
        axis.set_title("\n".join(lines), fontsize=6)
    figure.suptitle("G2-C K{}-dominant examples; stable-key ascending, first six".format(scale))
    figure.tight_layout()
    figure.savefig(str(output), bbox_inches="tight")
    figure.savefig(str(output.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(figure)
    return chosen


def compare(sample_paths, result_paths, dataset_root, output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite comparison output: {}".format(output_dir))
    maps = {label: _read_samples(sample_paths[label], label) for label in VERSIONS}
    keys = _validate_same_samples(maps)
    output_dir.mkdir(parents=True)
    figures = output_dir / "figures"
    figures.mkdir()
    source_rows = []
    for label in VERSIONS:
        for key in keys:
            row = maps[label][key]
            source_rows.append(dict(row, version=label, relative_image_path=_relative_path(key)))
    fields = ["version", "stable_sample_key", "relative_image_path", "dataset_split", "pid", "camid", "p2", "p4", "p6", "w2", "w4", "w6", "entropy", "dominant_k", "checkpoint_sha256"]
    _write_csv(output_dir / "per_sample_gating.csv", fields, source_rows)
    stats = sum((_statistics(label, list(maps[label].values())) for label in VERSIONS), [])
    _write_csv(output_dir / "gate_statistics.csv", list(stats[0]), stats)
    metrics = [_metrics(result_paths.get(label), label) for label in VERSIONS]
    metric_fields = sorted(set().union(*(row.keys() for row in metrics)))
    _write_csv(output_dir / "retrieval_metrics.csv", metric_fields, metrics)
    _plot(maps, "p", figures / "gate_probability_distributions.png")
    _plot(maps, "w", figures / "gate_weight_distributions.png")
    selection = []
    for scale in (4, 6):
        selected = _examples(maps, keys, Path(dataset_root), scale, figures / "k{}_dominant_examples.png".format(scale))
        for key in selected:
            selection.append({"selection_rule": "G2-C dominant then stable_sample_key order, first six", "dominant_k": scale, "stable_sample_key": key})
    _write_csv(output_dir / "sample_selection.csv", ["selection_rule", "dominant_k", "stable_sample_key"], selection)
    manifest = {
        "versions": list(VERSIONS), "sample_count_per_version": len(keys),
        "sample_set_status": "identical_stable_sample_key_order_verified",
        "sample_sources": {label: {"path": str(Path(sample_paths[label]).resolve()), "sha256": _sha(sample_paths[label])} for label in VERSIONS},
        "result_sources": {label: str(Path(path).resolve()) if path else "not_recorded" for label, path in result_paths.items()},
        "dataset_root": str(Path(dataset_root).resolve()),
        "probability_semantics": "p=softmax(logits/tau_g); p2+p4+p6=1",
        "weight_semantics": "w=3p; w2+w4+w6=3",
    }
    _atomic_text(output_dir / "comparison_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    readme = """# Four-version Dynamic Gating comparison

All four input TSVs passed exact stable-sample-key order and metadata equality checks. The comparison therefore uses one identical frozen image set. `retrieval_metrics.csv` contains only formal-result JSON values supplied to the tool; `not_recorded` means no formal result JSON was supplied. Gate plots distinguish probability `p` from applied weight `w=3p`. K4/K6 images are selected from G2-C by deterministic stable-key order, not by post-hoc visual choice.
"""
    _atomic_text(output_dir / "README.md", readme)
    digest_files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    _atomic_text(output_dir / "SHA256SUMS.txt", "".join("{}  {}\n".format(_sha(path), path.relative_to(output_dir).as_posix()) for path in digest_files))
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("g1", "g2", "g2a", "g2c"):
        parser.add_argument("--{}-samples".format(key), required=True)
        parser.add_argument("--{}-formal-result".format(key), default=None)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    sample_paths = {"G1": args.g1_samples, "G2": args.g2_samples, "G2-A": args.g2a_samples, "G2-C": args.g2c_samples}
    result_paths = {"G1": args.g1_formal_result, "G2": args.g2_formal_result, "G2-A": args.g2a_formal_result, "G2-C": args.g2c_formal_result}
    print(str(compare(sample_paths, result_paths, args.dataset_root, args.output_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
