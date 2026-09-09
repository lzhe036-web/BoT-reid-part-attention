"""Validated raw evidence, idempotent ledger notes, and offline delivery helpers."""
import csv
import io
import json
import math
import os
import shutil
import tarfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from utils.dynamic_experiment_registry import _atomic_write, atomic_write_json, sha256_file

BASE_COMMIT = "63761021a40693694f037d850066deb2237a5c41"
SCALES = (2, 4, 6)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path, delimiter=","):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def checked_file(evidence):
    path = Path(evidence["path"])
    if not path.is_file() or sha256_file(path) != evidence["sha256"]:
        raise ValueError("Missing or changed evidence: {}".format(path))
    return path


def validate_raw(result):
    """Validate the complete residual schema before allowing registry writes."""
    alpha = float(result["static_dynamic_alpha"])
    if alpha not in (.1, .5) or result["fusion_mode"] != "static_concat_plus_gated_residual":
        raise ValueError("Unsupported alpha/fusion identity")
    manifest_path = checked_file({"path": result["evidence"]["analysis_manifest"],
                                  "sha256": result["evidence"]["analysis_manifest_sha256"]})
    manifest = read_json(manifest_path)
    if (float(manifest["static_dynamic_alpha"]) != alpha
            or manifest["checkpoint_sha256"] != result["selected_checkpoint"]["sha256"]):
        raise ValueError("Analysis identity differs from formal result")
    source = checked_file(manifest["files"]["per_sample_gating_tsv"])
    rows = read_rows(source, "\t")
    original = read_rows(checked_file(manifest["files"]["test_gate_samples_tsv"]), "\t")
    original_by_key = {r["stable_sample_key"]: r for r in original}
    keys = [r["stable_sample_key"] for r in rows]
    if (not rows or len(set(keys)) != len(rows) or len(original_by_key) != len(original)
            or set(keys) != set(original_by_key)):
        raise ValueError("Empty, duplicate, or inconsistent sample keys")
    for row in rows:
        key = row["stable_sample_key"]
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Expected SHA256 stable sample key")
        if row["dataset_split"] not in ("query", "gallery"):
            raise ValueError("Unexpected fixed-sample scope")
        for field in ("dataset_split", "pid", "camid", "checkpoint_sha256", "dominant_k"):
            if row[field] != original_by_key[key][field]:
                raise ValueError("Original and residual sample metadata differ: " + field)
        if row["checkpoint_sha256"] != result["selected_checkpoint"]["sha256"]:
            raise ValueError("Sample is bound to a different checkpoint")
        p = [float(row["p"+str(k)]) for k in SCALES]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in p):
            raise ValueError("Invalid probabilities")
        if not math.isclose(sum(p), 1., rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Probabilities must sum to one")
        if float(row["alpha"]) != alpha or int(row["dominant_k"]) != SCALES[p.index(max(p))]:
            raise ValueError("Invalid alpha or dominant scale (ties use first K)")
        for k, probability in zip(SCALES, p):
            for name, expected in (("w", 3*probability), ("residual", alpha*3*probability),
                                   ("c", 1+alpha*3*probability)):
                if not math.isclose(float(row[name+str(k)]), expected, rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError("Invalid residual formula in " + name)
            for prefix in ("p", "w"):
                if not math.isclose(float(row[prefix+str(k)]), float(original_by_key[key][prefix+str(k)]), abs_tol=1e-8):
                    raise ValueError("Raw probability/weight was changed")
    summary = read_json(checked_file(manifest["files"]["dynamic_gating_summary_json"]))
    if (summary["selected_sample_count"] != len(rows)
            or int(summary["training_epoch_statistics"]["epoch"]) != int(result["selected_checkpoint"]["epoch"])):
        raise ValueError("Sample count or training-statistics epoch mismatch")
    stats = read_rows(checked_file(manifest["files"]["gate_statistics_csv"]))
    indexed = {(r["quantity"],int(r["scale"])):r for r in stats}
    if len(indexed) != 12 or len(stats) != 12:
        raise ValueError("Expected twelve quantity/scale statistic rows")
    for quantity,prefix in (("probability","p"),("dynamic_weight","w"),
                            ("residual_alpha_times_w","residual"),("final_local_coefficient","c")):
        for k in SCALES:
            record = indexed[(quantity,k)]
            mean = sum(float(r[prefix+str(k)]) for r in rows)/len(rows)
            dominant = sum(int(r["dominant_k"]) == k for r in rows)/len(rows)
            if (int(record["sample_count"]) != len(rows)
                    or not math.isclose(float(record["mean"]),mean,rel_tol=1e-6,abs_tol=1e-8)
                    or not math.isclose(float(record["dominant_ratio"]),dominant,abs_tol=1e-8)):
                raise ValueError("Gate statistics do not match fixed-sample raw rows")
    return rows, source


def require_registered(result, records_root):
    checkpoint_sha = result["selected_checkpoint"]["sha256"]
    run_id = "{}-{}-{}".format(result["experiment_id"], result["commit"][:10], checkpoint_sha[:10])
    run_dir = Path(records_root) / "runs" / run_id
    manifest = read_json(run_dir / "run_manifest.json")
    status = read_json(run_dir / "run_status.json")
    for field in ("branch", "commit", "static_dynamic_alpha", "fusion_mode"):
        if manifest[field] != result[field]:
            raise ValueError("Registered identity mismatch: " + field)
    if (manifest["status"] != "success" or status["status"] != "success"
            or manifest["selected_checkpoint"]["sha256"] != checkpoint_sha):
        raise ValueError("Formal success must be registered before packaging")
    for field, value in result["metrics"].items():
        if not math.isclose(float(manifest["metrics"][field]), float(value), abs_tol=1e-9):
            raise ValueError("Registered metric mismatch: " + field)
    for relative in ("runs.csv", "tables/main_results.csv"):
        matches = [r for r in read_rows(Path(records_root)/relative) if r["run_id"] == run_id]
        if len(matches) != 1 or matches[0]["status"] != "success":
            raise ValueError("Missing/duplicate formal table entry: " + relative)
    return run_dir


def record_delivery(result, records_root, experiments_path, output_dir, stage, error=None,
                    package_dir=None, archive=None, archive_sha256=None):
    run_dir = require_registered(result, records_root)
    rows, source = validate_raw(result)
    output_dir = Path(output_dir)
    path = output_dir / "delivery_status.json"
    payload = read_json(path) if path.exists() else {}
    payload.update({"run_id": run_dir.name, "stage": stage, "error": str(error) if error else None,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "formal_metrics_registered": True})
    # A retry cannot keep an old archive labelled as its current successful delivery.
    for key, value in (("package_dir", package_dir), ("archive", archive), ("archive_sha256", archive_sha256)):
        payload[key] = str(value) if value is not None else None
    atomic_write_json(path, payload)
    scope = dict(Counter(r["dataset_split"] for r in rows))
    start, end = ("<!-- g2-e-alpha:{}:{} -->".format(run_dir.name, marker) for marker in ("start", "end"))
    lines = [start, "", "### " + run_dir.name, "",
             "- Branch: `{}`; direct baseline: `codex/g2-e-static-dynamic-alpha0p3-tau0p5` (`{}`).".format(result["branch"], BASE_COMMIT),
             "- Training SHA: `{}`; seed=42; Market1501; alpha={}; tau=0.5; descriptor=2816.".format(result["commit"], result["static_dynamic_alpha"]),
             "- Fusion: `concat(g,(1+alpha*w2)z2,(1+alpha*w4)z4,(1+alpha*w6)z6)`; w=3p.",
             "- Selected epoch: {}; checkpoint: `{}`; SHA256: `{}`.".format(result["selected_checkpoint"]["epoch"], result["selected_checkpoint"]["path"], result["selected_checkpoint"]["sha256"]),
             "- Full retrieval (%): " + "; ".join("{}={:.8f}".format(k,v) for k,v in result["metrics"].items()) + ".",
             "- Fixed evaluation sample count: {}; split counts: {}. These are separate from selected-epoch training statistics in the formal registry.".format(len(rows), json.dumps(scope, sort_keys=True)),
             "", "| Scale | mean p | dominant ratio | mean w | mean alpha*w | mean c |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k in SCALES:
        values = [sum(float(r[p+str(k)]) for r in rows)/len(rows) for p in ("p", "w", "residual", "c")]
        ratio = sum(int(r["dominant_k"]) == k for r in rows)/len(rows)
        lines.append("| K{} | {:.8f} | {:.8f} | {:.8f} | {:.8f} | {:.8f} |".format(k, values[0], ratio, *values[1:]))
    lines += ["", "- Raw rows: `{}`; source config: `{}`; formal result: `{}`.".format(source, result["evidence"]["config"], output_dir),
              "- Delivery stage: **{}**; error: {}.".format(stage, str(error).replace("\n", " ") if error else "none"),
              "- Figures: `{}`; archive: `{}`; SHA256: `{}`.".format(str(Path(package_dir)/"figures") if package_dir else "pending", archive or "pending", archive_sha256 or "pending"),
              "", end]
    replacement = "\n".join(lines)
    experiments_path = Path(experiments_path)
    content = experiments_path.read_text(encoding="utf-8")
    if content.count(start) != content.count(end) or content.count(start) > 1:
        raise ValueError("Malformed alpha ledger markers; history was not overwritten")
    if start in content:
        before, remainder = content.split(start, 1)
        _, after = remainder.split(end, 1)
        content = before + replacement + after
    else:
        content = content.rstrip() + "\n\n" + replacement + "\n"
    _atomic_write(experiments_path, content)
    return payload


def copy_registry_evidence(run_dir, records_root, experiments_path, package_dir):
    destination = Path(package_dir)/"registry"
    destination.mkdir()
    shutil.copytree(str(run_dir), str(destination/"runs"/run_dir.name))
    for relative in ("runs.csv", "tables/main_results.csv"):
        rows = [r for r in read_rows(Path(records_root)/relative) if r["run_id"] == run_dir.name]
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        _atomic_write(target, buffer.getvalue())
    shutil.copyfile(str(experiments_path), str(Path(package_dir)/"EXPERIMENTS.md"))


def archive_package(package_dir, archive_dir):
    package_dir, archive_dir = Path(package_dir), Path(archive_dir)
    entries = {}
    for line in (package_dir/"SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in entries:
            raise ValueError("Invalid checksum entry: " + name)
        entries[name] = digest
    actual = {p.relative_to(package_dir).as_posix() for p in package_dir.rglob("*") if p.is_file() and p != package_dir/"SHA256SUMS"}
    if set(entries) != actual:
        raise ValueError("Archive checksum inventory mismatch")
    for name, digest in entries.items():
        if sha256_file(package_dir/name) != digest:
            raise ValueError("Archive checksum failed: " + name)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir/(package_dir.name+".tar.gz")
    if archive.exists():
        raise FileExistsError(str(archive))
    temporary = archive.with_name(archive.name+".tmp."+uuid.uuid4().hex)
    try:
        with tarfile.open(str(temporary), "w:gz") as handle:
            handle.add(str(package_dir), arcname=package_dir.name)
        os.replace(str(temporary), str(archive))
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256_file(archive)
    _atomic_write(Path(str(archive)+".sha256"), "{}  {}\n".format(digest, archive.name))
    return archive, digest
