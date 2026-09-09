#!/usr/bin/env python
"""Replot a portable G2-E package without PyTorch, dataset, or checkpoint."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot(package_dir, output_dir):
    package_dir, output_dir = Path(package_dir), Path(output_dir)
    with (package_dir/"per_sample_gating.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    alpha = float(rows[0]["alpha"])
    if any(float(r["alpha"]) != alpha for r in rows):
        raise ValueError("Mixed alpha identities")
    output_dir.mkdir(parents=True, exist_ok=True)
    for prefix, maximum, label in (("w", 3, "w=3p"), ("c", 1+3*alpha, "c=1+alpha*w")):
        minimum = 0 if prefix == "w" else 1
        bins = np.linspace(minimum, maximum, 31)
        fig, ax = plt.subplots(figsize=(7,4))
        for k in (2,4,6):
            ax.hist([float(r[prefix+str(k)]) for r in rows], bins=bins, histtype="step", label="K{}".format(k))
        ax.set(xlabel=label, ylabel="Fixed sample count", xlim=(minimum,maximum), title="G2-E alpha={}".format(alpha))
        ax.legend()
        fig.tight_layout()
        for extension in ("png","pdf"):
            fig.savefig(str(output_dir/(prefix+"_distribution."+extension)), dpi=180)
        plt.close(fig)
    for k in (4,6):
        selected = sorted((r for r in rows if int(r["dominant_k"]) == k), key=lambda r:r["stable_sample_key"])[:6]
        fig, axes = plt.subplots(1, max(1,len(selected)), figsize=(2.8*max(1,len(selected)),5), squeeze=False)
        for ax in axes[0]:
            ax.axis("off")
        if not selected:
            axes[0][0].text(.5,.5,"0 K{}-dominant samples".format(k), ha="center")
        for index, (ax,row) in enumerate(zip(axes[0],selected),1):
            name = Path(row["relative_image_path"]).name
            with Image.open(package_dir/"sample_images"/"K{}_{}_{}".format(k,index,name)) as image:
                ax.imshow(image.convert("RGB"))
            parts = [name, "alpha={}; dominant=K{}".format(alpha,k), row["stable_sample_key"][:16]]
            parts += [prefix+"=("+",".join("{:.4f}".format(float(row[prefix+str(s)])) for s in (2,4,6))+")" for prefix in ("p","w","c")]
            ax.set_title("\n".join(parts), fontsize=7)
        fig.tight_layout()
        for extension in ("png","pdf"):
            fig.savefig(str(output_dir/("k{}_dominant_examples.{}".format(k,extension))), dpi=180)
        plt.close(fig)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(plot(args.package_dir,args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
