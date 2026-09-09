#!/usr/bin/env python
"""Offline, exact-sample comparison of checked G1/G2/G2-A and G2-E packages."""
import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ("rank1_percent", "rank5_percent", "rank10_percent", "map_percent")
LEGACY_METRICS = ("Rank-1", "Rank-5", "Rank-10", "mAP")
SCALES = (2,4,6)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if Path(path).suffix==".tsv" else ","))


def verify_package(root):
    root = Path(root).resolve()
    checksum = next((root/name for name in ("SHA256SUMS","SHA256SUMS.txt") if (root/name).is_file()), None)
    if checksum is None:
        raise FileNotFoundError("Missing package checksum list: {}".format(root))
    names = set()
    for line in checksum.read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ",1)
        if Path(name).is_absolute() or ".." in Path(name).parts or name in names:
            raise ValueError("Invalid/duplicate checksum path")
        if digest(root/name) != sha:
            raise ValueError("Changed package evidence: " + name)
        names.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != checksum}
    if actual != names:
        raise ValueError("Unlisted/missing package files; replot outside the source package")
    return digest(checksum)


def load_source(source, expected_count=256):
    root, label = Path(source["package_dir"]).resolve(), source["label"]
    checksum_sha = verify_package(root)
    if label in ("G1","G2","G2-A"):
        rows = [r for r in table(root/"per_sample_gating.csv") if r["version"] == label]
        matches = [r for r in table(root/"retrieval_metrics.csv") if r["version"] == label]
        if len(matches)!=1:
            raise ValueError("Missing/duplicate historical metrics for " + label)
        metric = matches[0]
        values = {k:float(metric[old]) for k,old in zip(METRICS,LEGACY_METRICS)}
        checkpoint_sha = metric["checkpoint_sha256"]
        alpha = None
        provenance = {"branch":metric["branch"], "training_commit":metric["training_commit"],
                      "selected_epoch":int(metric["selected_epoch"]), "checkpoint_sha256":checkpoint_sha,
                      "fusion":"pure_dynamic_weighted_concat", "tau":float(metric["tau_g"])}
    else:
        alpha = float(source["alpha"])
        if alpha not in (.1,.3,.5):
            raise ValueError("Expected alpha=0.1/0.3/0.5")
        manifest, result = read_json(root/"run_manifest.json"), read_json(root/"formal_result.json")
        if (float(manifest["static_dynamic_alpha"]) != alpha
                or float(result["static_dynamic_alpha"]) != alpha
                or float(manifest["gating_temperature"]) != .5
                or manifest["fusion_mode"] != "static_concat_plus_gated_residual"
                or manifest["dataset"] != "market1501" or manifest["seed"] != 42
                or manifest["selected_checkpoint"] != result["selected_checkpoint"]
                or manifest["training_commit"] != result["commit"] or manifest["branch"] != result["branch"]):
            raise ValueError("G2-E package identity mismatch")
        for file, field in (("config_source.yml","source_config_sha256"), ("config_resolved.yml","resolved_config_sha256"),
                            ("reproducibility.json","reproducibility_sha256")):
            if digest(root/file) != manifest[field]:
                raise ValueError("Config/provenance binding mismatch: "+file)
        values = {k:float(result["metrics"][k]) for k in METRICS}
        metric_rows = table(root/"retrieval_metrics.csv")
        if len(metric_rows)!=1 or any(not math.isclose(values[k],float(metric_rows[0][k]),abs_tol=1e-9) for k in METRICS):
            raise ValueError("Packaged metrics differ from formal evaluation")
        rows = table(root/"per_sample_gating.tsv")
        checkpoint_sha = result["selected_checkpoint"]["sha256"]
        provenance = {"branch":result["branch"],"training_commit":result["commit"],
                      "selected_epoch":result["selected_checkpoint"]["epoch"], "checkpoint_sha256":checkpoint_sha,
                      "fusion":result["fusion_mode"], "tau":.5}
    if not all(math.isfinite(v) and 0<=v<=100 for v in values.values()):
        raise ValueError("Invalid retrieval percentage")
    if len(rows)!=expected_count or len({r["stable_sample_key"] for r in rows})!=len(rows):
        raise ValueError("Expected exactly {} distinct samples for {}".format(expected_count,label))
    for row in rows:
        p=[float(row["p"+str(k)]) for k in SCALES]
        if (row["checkpoint_sha256"] != checkpoint_sha or not all(math.isfinite(v) and 0<=v<=1 for v in p)
                or not math.isclose(sum(p),1.,rel_tol=1e-6,abs_tol=1e-8)
                or int(row["dominant_k"]) != SCALES[p.index(max(p))]):
            raise ValueError("Invalid sample probability/checkpoint/dominant identity")
        for k,probability in zip(SCALES,p):
            if not math.isclose(float(row["w"+str(k)]),3*probability,rel_tol=1e-6,abs_tol=1e-8):
                raise ValueError("Invalid w=3p")
            if alpha is not None:
                if float(row["alpha"])!=alpha:
                    raise ValueError("Mixed per-sample alpha")
                for prefix,value in (("residual",alpha*3*probability),("c",1+alpha*3*probability)):
                    if not math.isclose(float(row[prefix+str(k)]),value,rel_tol=1e-6,abs_tol=1e-8):
                        raise ValueError("Invalid residual/c formula")
    return dict(label=label,alpha=alpha,rows=sorted(rows,key=lambda r:r["stable_sample_key"]),metrics=values,
                provenance=dict(provenance,package_dir=str(root),checksum_list_sha256=checksum_sha))


def compare(config_path, output_dir, expected_count=256):
    sources = read_json(config_path)["sources"]
    loaded, missing = [], []
    for source in sources:
        if not Path(source["package_dir"]).is_dir():
            missing.append({"label":source["label"],"missing_package_dir":source["package_dir"]})
            continue
        # Corrupt or mismatched evidence is an error, never silently excluded.
        loaded.append(load_source(source,expected_count))
    if not loaded:
        raise ValueError("No available verified packages")
    if len({s["label"] for s in loaded})!=len(loaded) or len({s["alpha"] for s in loaded if s["alpha"] is not None})!=sum(s["alpha"] is not None for s in loaded):
        raise ValueError("Duplicate version/alpha labels")
    identity = lambda rows:[(r["stable_sample_key"],r.get("split",r.get("dataset_split")),int(r["pid"]),int(r["camid"])) for r in rows]
    expected = identity(loaded[0]["rows"])
    for source in loaded[1:]:
        if identity(source["rows"])!=expected:
            raise ValueError("Exact sample identity mismatch for {}; no intersection was taken".format(source["label"]))
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    figures=out/"figures"; figures.mkdir()
    base=next((s for s in loaded if s["alpha"]==.3),None)
    metrics=[]
    for source in loaded:
        row=dict(version=source["label"],alpha=source["alpha"],**source["metrics"],**source["provenance"])
        for key in METRICS:
            row["delta_"+key+"_vs_alpha0p3_pp"] = source["metrics"][key]-base["metrics"][key] if base and source["alpha"] is not None else "not_applicable"
        metrics.append(row)
    def write(path, rows):
        with path.open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(out/"retrieval_metrics.csv",metrics)
    alphas=[s for s in loaded if s["alpha"] is not None]
    for name,group,prefix,limits in (("all_versions_w",loaded,"w",(0,3)), ("alpha_sweep_w",alphas,"w",(0,3)),
                                      ("alpha_sweep_c",alphas,"c",(1,2.5))):
        if not group: continue
        fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharex=True,sharey=True)
        bins=np.linspace(*limits,31)
        for ax,k in zip(axes,SCALES):
            for source in group:
                ax.hist([float(r[prefix+str(k)]) for r in source["rows"]],bins=bins,histtype="step",label=source["label"],linewidth=1.6)
            ax.set(title="K{}".format(k),xlabel="w=3p" if prefix=="w" else "c=1+alpha*w",xlim=limits,ylabel="Fixed sample count")
            ax.legend(fontsize=6)
        fig.tight_layout()
        for extension in ("png","pdf"):fig.savefig(str(figures/(name+"."+extension)),dpi=180)
        plt.close(fig)
    for index,source in enumerate(loaded):
        write(out/("samples_{:02d}.csv".format(index)),source["rows"])
    status={"sources":[dict(label=s["label"],alpha=s["alpha"],**s["provenance"]) for s in loaded],
            "missing_sources":missing, "complete_alpha_sweep":{s["alpha"] for s in alphas}=={.1,.3,.5},
            "sample_count":expected_count,"metric_unit":"percent; differences in percentage points",
            "histogram_bins":{"w":[0,3,30],"c":[1,2.5,30]},"sample_match":"exact stable key/split/pid/camid"}
    (out/"comparison_manifest.json").write_text(json.dumps(status,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    shutil.copyfile(str(config_path),str(out/"sources.json"))
    shutil.copyfile(__file__,str(out/Path(__file__).name))
    (out/"README.md").write_text("# Verified gating comparison\n\nFull evaluation metrics; fixed samples are used only for gate distributions. w=3p and c=1+alpha*w are separate figures with shared bins. G1/G2/G2-A differ in controller input or temperature/fusion; those differences are not alpha effects. Single seed results imply no statistical significance.\n\nComplete alpha sweep: {}. Missing sources: {}.\n".format(status["complete_alpha_sweep"],json.dumps(missing,ensure_ascii=False)),encoding="utf-8")
    return status


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources",required=True,help="JSON: sources=[{label,package_dir,alpha (E only)}]")
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--expected-count",type=int,default=256)
    args=parser.parse_args(argv)
    print(json.dumps(compare(args.sources,args.output_dir,args.expected_count),indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
