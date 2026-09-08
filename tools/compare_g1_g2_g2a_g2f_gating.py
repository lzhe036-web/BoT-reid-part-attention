#!/usr/bin/env python
"""Compare actual gate weights on one identical, checkpoint-bound sample list.

Dense G1/G2/G2-A files use their recorded ``w=3p`` values.  G2-F uses the
actual sparse ``w=3*p_top2`` values, including zeros.  This tool refuses to
silently intersect or reorder sample lists, so a cross-version plot cannot be
mistakenly made from unrelated bounded selections.
"""

from __future__ import absolute_import

import argparse
import csv
import json
import os
import sys
import uuid
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCALES = (2, 4, 6)
VERSIONS = ("G1", "G2", "G2-A-tau0p5", "G2-F-Top2-tau0p5")


def _atomic(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}".format(path.name, uuid.uuid4().hex))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write(path, fields, rows):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    _atomic(path, buffer.getvalue())


def _read(path, version):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"stable_sample_key", "w2", "w4", "w6"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("{} has no compatible per-sample gate fields".format(version))
    keys = [row["stable_sample_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("{} repeats stable sample keys".format(version))
    return rows


def compare(paths, output_dir):
    rows_by_version = {version: _read(path, version) for version, path in paths.items()}
    reference = [row["stable_sample_key"] for row in rows_by_version[VERSIONS[0]]]
    for version in VERSIONS[1:]:
        if [row["stable_sample_key"] for row in rows_by_version[version]] != reference:
            raise ValueError("{} sample list/order differs from G1; refusing a silent intersection".format(version))
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite comparison output: {}".format(output_dir))
    output_dir.mkdir(parents=True)
    source = []
    for version, rows in rows_by_version.items():
        for row in rows:
            source.append({"version": version, "stable_sample_key": row["stable_sample_key"],
                           **{"w{}".format(scale): float(row["w{}".format(scale)]) for scale in SCALES}})
    _write(output_dir / "cross_version_plot_data.csv",
           ["version", "stable_sample_key", "w2", "w4", "w6"], source)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), dpi=180, sharey=True)
    bins = np.linspace(0.0, 3.0, 31)
    for index, scale in enumerate(SCALES):
        for version in VERSIONS:
            values = [float(row["w{}".format(scale)]) for row in rows_by_version[version]]
            axes[index].hist(values, bins=bins, histtype="step", linewidth=1.5,
                             label=version, density=True)
        axes[index].set(xlim=(0.0, 3.0), title="K{} actual fusion weight".format(scale),
                        xlabel="w = 3p (dense) or 3p_top2 (G2-F)")
        axes[index].grid(alpha=.25)
    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(str(output_dir / "g1_g2_g2a_g2f_actual_weight_distribution") + "." + suffix,
                       bbox_inches="tight")
    plt.close(figure)
    manifest = {"versions": list(VERSIONS), "sample_count": len(reference),
                "sample_identity_rule": "all versions must match G1 stable_sample_key order exactly",
                "weight_semantics": "G1/G2/G2-A: dense w=3p; G2-F: actual sparse w=3p_top2, zeros retained"}
    _atomic(output_dir / "comparison_manifest.json", json.dumps(manifest, ensure_ascii=False,
                                                                   indent=2, sort_keys=True) + "\n")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1", required=True); parser.add_argument("--g2", required=True)
    parser.add_argument("--g2a", required=True); parser.add_argument("--g2f", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(compare({"G1": args.g1, "G2": args.g2, "G2-A-tau0p5": args.g2a,
                   "G2-F-Top2-tau0p5": args.g2f}, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
