#!/usr/bin/env python
"""Compare every Dynamic Gating variant against one fixed formal G2 baseline.

This tool deliberately does *not* follow Git parentage for statistics.  Every
comparator is evaluated independently against the formal G2 global-local run
recorded at ``fa4e7f88...``.  It reads branch tables with ``git show`` and
only consumes an artifact when its recorded SHA256 is available locally (or
under an explicitly supplied artifact-search root).  Missing evidence remains
missing: the tool emits a machine-readable status and never substitutes smoke
runs, inherited rows, or another variant's numbers.

The fixed candidate manifest is generated from Market images before any gate
TSV is read.  Existing historical TSVs use a legacy hashed stable key, so the
manifest carries both an auditable human-readable identity and the compatible
legacy hash.  A comparison is fail-closed unless every fixed candidate is
present for both G2 and its comparator.
"""

from __future__ import absolute_import

import argparse
import csv
import hashlib
import inspect
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BASELINE_LABEL = "G2-global-local"
BASELINE_BRANCH = "origin/codex/g2-global-local-gating"
BASELINE_TRAINING_COMMIT = "fa4e7f88f7ab645e9ba6b9a8e6cffdd9056b36c8"
BASELINE_CHECKPOINT_SHA256 = (
    "49a766fb520cca5dfe9121f272994185db9fddee45709c9d61446c6781dc7d45"
)
FIXED_SELECTION_PREFIX = "g2-baseline-all-variants-v1|"
NOT_APPLICABLE = "N/A"
NOT_RECORDED = "not_recorded"
MISSING_FORMAL = "missing_formal_evidence"


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    branch: str
    experiment_id: str
    gate_input: str
    active_scales: tuple
    # This is a code-audited contract, not a normalisation assumption.  The
    # three-way implementations multiply p by 3; the two-way ablations use p
    # directly.  Raw TSV evidence must still satisfy this contract.
    expected_native_weight_sum: float
    config_filename: str
    smoke_script: str
    formal_script: str


VARIANTS = (
    VariantSpec(
        "g2", BASELINE_LABEL, BASELINE_BRANCH,
        "C2-L03-MGDG-G2-GL-T1-S42", "concat_global_local", (2, 4, 6), 3.0,
        "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_autodl.yml",
        "test_g2_global_local_gating_1epoch_autodl.sh",
        "train_g2_global_local_seed42_autodl.sh",
    ),
    VariantSpec(
        "g1", "G1", "origin/exp/c2-l03-multi-granularity-dynamic-gating",
        "C2-L03-MGDG-T1-S42", "global", (2, 4, 6), 3.0,
        "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_autodl.yml",
        "test_c2_l03_multi_granularity_dynamic_gating_1epoch.sh",
        "train_c2_l03_multi_granularity_dynamic_gating_autodl.sh",
    ),
    VariantSpec(
        "g2_local_only", "G2-local-only", "origin/codex/g2-local-only",
        "C2-L03-MGDG-G2-LOCAL-T1-S42", "concat_local", (2, 4, 6), 3.0,
        "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_local_only_autodl.yml",
        "test_g2_local_only_gating_1epoch_autodl.sh",
        "train_g2_local_only_seed42_autodl.sh",
    ),
    VariantSpec(
        "g2_without_z6", "G2-without-z6", "origin/codex/g2-without-z6",
        "C2-L03-MGDG-G2-WITHOUT-Z6-T1-S42", "concat_z2_z4", (2, 4), 1.0,
        "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_without_z6_autodl.yml",
        "test_g2_without_z6_gating_1epoch_autodl.sh",
        "train_g2_without_z6_seed42_autodl.sh",
    ),
    VariantSpec(
        "g2_without_z4", "G2-without-z4", "origin/codex/g2-without-z4",
        "C2-L03-MGDG-G2-WITHOUT-Z4-T1-S42", "concat_z2_z6", (2, 6), 1.0,
        "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_without_z4_autodl.yml",
        "test_g2_without_z4_gating_1epoch_autodl.sh",
        "train_g2_without_z4_seed42_autodl.sh",
    ),
    VariantSpec(
        "g2_without_z2", "G2-without-z2", "origin/codex/g2-without-z2",
        "C2-L03-MGDG-G2-WITHOUT-Z2-T1-S42", "concat_z4_z6", (4, 6), 1.0,
        "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_without_z2_autodl.yml",
        "test_g2_without_z2_gating_1epoch_autodl.sh",
        "train_g2_without_z2_seed42_autodl.sh",
    ),
)
SPEC_BY_KEY = {spec.key: spec for spec in VARIANTS}
COMPARATORS = tuple(spec for spec in VARIANTS if spec.key != "g2")


class AnalysisError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp".format(path.name))
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path, payload):
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_csv(path, fields, rows, delimiter=","):
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(fields), delimiter=delimiter,
        lineterminator="\n", extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, NOT_APPLICABLE) for field in fields})
    _atomic_text(path, stream.getvalue())


def _markdown_escape(value):
    return str(value if value not in (None, "") else NOT_APPLICABLE).replace("|", "\\|").replace("\n", " ")


def _write_markdown_from_rows(path, title, fields, rows, note=None):
    lines = ["# {}".format(title), ""]
    if note:
        lines.extend([note, ""])
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join("---" for _ in fields) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(row.get(field)) for field in fields) + " |")
    lines.append("")
    _atomic_text(path, "\n".join(lines))


def _git(repo_root, arguments):
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(repo_root).resolve())] + list(arguments),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        message = getattr(error, "stderr", b"")
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        raise AnalysisError("git {} failed: {}".format(" ".join(arguments), message))
    return result.stdout.decode("utf-8", errors="replace")


def read_csv_from_ref(repo_root, ref, path, delimiter=","):
    text = _git(repo_root, ["show", "{}:{}".format(ref, path)])
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def ref_commit(repo_root, ref):
    return _git(repo_root, ["rev-parse", ref]).strip()


def read_text_from_ref(repo_root, ref, path):
    return _git(repo_root, ["show", "{}:{}".format(ref, path)])


def audit_implementation_contract(repo_root, spec):
    """Confirm the branch-local gate semantics before consuming its TSV."""
    try:
        config_text = read_text_from_ref(
            repo_root, spec.branch, "configs/" + spec.config_filename
        )
        model_text = read_text_from_ref(repo_root, spec.branch, "modeling/baseline.py")
    except AnalysisError as error:
        return "not_confirmed: {}".format(error)
    configured = re.search(
        r"MULTI_GRANULARITY_GATING_INPUT\s*:\s*['\"]?([^,'\"}\s]+)", config_text
    )
    if not configured or configured.group(1).lower() != spec.gate_input:
        return "not_confirmed: config gating input mismatch"
    if spec.gate_input not in model_text:
        # G1's global mode predates explicit enum prose on some branches.
        if spec.gate_input != "global" or "gating_input == 'global'" not in model_text:
            return "not_confirmed: model does not implement gate input"
    if spec.expected_native_weight_sum == 3.0:
        if "float(self.num_scales) * probabilities" not in model_text and "float(self.gate_count) * probabilities" not in model_text:
            return "not_confirmed: code does not expose three-way scaled-softmax"
    else:
        expected_fragment = "probabilities if self.gating_input == '{}'".format(spec.gate_input)
        if expected_fragment not in model_text:
            return "not_confirmed: code does not expose two-way probability weights"
    return "confirmed"


def find_formal_row(rows, spec):
    candidates = [
        row for row in rows
        if row.get("experiment_id") == spec.experiment_id
        and row.get("run_kind") == "formal"
        and row.get("status") == "success"
    ]
    if len(candidates) > 1:
        raise AnalysisError("{} has multiple successful formal rows".format(spec.label))
    return candidates[0] if candidates else None


def _field(row, name):
    return row.get(name, NOT_RECORDED) if row else NOT_RECORDED


def _is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _artifact_path(row, stem):
    if not row:
        return None, None
    path = row.get("{}_path".format(stem), NOT_RECORDED)
    sha = row.get("{}_sha256".format(stem), NOT_RECORDED)
    if path in (None, "", NOT_RECORDED, "not_archived") or not _is_sha(sha):
        return None, None
    return Path(path), sha


def _find_artifact(row, stem, search_roots=()):
    expected_path, expected_sha = _artifact_path(row, stem)
    if expected_path is None:
        return None, "not_recorded"
    if expected_path.is_file() and sha256_file(expected_path) == expected_sha:
        return expected_path.resolve(), "recorded_absolute_path"
    for root in search_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for candidate in root.rglob(expected_path.name):
            if candidate.is_file() and sha256_file(candidate) == expected_sha:
                return candidate.resolve(), "artifact_search_root"
    return None, "not_archived"


def audit_variant(repo_root, spec, search_roots=()):
    result = {
        "variant": spec.label,
        "baseline": BASELINE_LABEL,
        "branch": spec.branch,
        "expected_experiment_id": spec.experiment_id,
        "expected_gate_input": spec.gate_input,
        "expected_active_scales": ",".join(str(item) for item in spec.active_scales),
        "expected_native_weight_sum": spec.expected_native_weight_sum,
    }
    try:
        result["branch_tip"] = ref_commit(repo_root, spec.branch)
        rows = read_csv_from_ref(repo_root, spec.branch, "experiment_records/runs.csv")
    except AnalysisError as error:
        result.update({"formal_status": MISSING_FORMAL, "audit_error": str(error)})
        return result, None
    result["implementation_contract"] = audit_implementation_contract(repo_root, spec)
    row = find_formal_row(rows, spec)
    if row is None:
        result.update({"formal_status": MISSING_FORMAL, "audit_error": NOT_APPLICABLE})
        return result, None
    result.update({
        "formal_status": "success",
        "audit_error": NOT_APPLICABLE,
        "training_branch": _field(row, "branch"),
        "formal_training_commit": _field(row, "commit"),
        "source_config_sha256": _field(row, "source_config_sha256"),
        "resolved_config_sha256": _field(row, "resolved_config_sha256"),
        "seed": _field(row, "seed"),
        "dataset": _field(row, "dataset"),
        "temperature": _field(row, "gating_temperature"),
        "normalization": _field(row, "gating_normalization"),
        "scale_order": _field(row, "scale_order"),
        "selected_checkpoint_sha256": _field(row, "selected_checkpoint_sha256"),
        "selected_epoch": _field(row, "selected_epoch"),
        "rank1": _field(row, "rank1_percent"),
        "map": _field(row, "map_percent"),
        "query_gallery_protocol": _field(row, "gating_sample_selection_rule"),
        "run_id": _field(row, "run_id"),
    })
    sources = {}
    for stem in ("source_config", "resolved_config", "selected_checkpoint", "gating_samples", "dynamic_gating_summary"):
        artifact, status = _find_artifact(row, stem, search_roots)
        sources[stem] = {"path": str(artifact) if artifact else NOT_APPLICABLE, "status": status}
        result["{}_artifact_status".format(stem)] = status
    result["formal_evidence_access"] = (
        "locally_verifiable" if all(
            sources[key]["status"] in ("recorded_absolute_path", "artifact_search_root")
            for key in ("source_config", "resolved_config", "selected_checkpoint", "gating_samples", "dynamic_gating_summary")
        ) else "formal_registered_artifacts_not_archived"
    )
    return result, {"row": row, "sources": sources}


def validate_baseline_record(inventory, evidence):
    baseline = inventory["g2"]
    if baseline.get("formal_status") != "success":
        raise AnalysisError("The required formal G2 baseline is absent")
    if baseline.get("formal_training_commit") != BASELINE_TRAINING_COMMIT:
        raise AnalysisError("G2 formal training commit does not match the fixed baseline")
    if baseline.get("selected_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256:
        raise AnalysisError("G2 formal checkpoint SHA256 does not match the fixed baseline")
    if baseline.get("training_branch") != "codex/g2-global-local-gating":
        raise AnalysisError("G2 formal record has the wrong training branch")
    if tuple(int(item) for item in str(baseline.get("scale_order", "")).split(",")) != (2, 4, 6):
        raise AnalysisError("G2 formal record has the wrong active scales")
    if evidence["g2"]["row"].get("gating_input") != "concat_global_local":
        raise AnalysisError("G2 formal record has the wrong gate input")


def _number(value):
    if value in (None, "", NOT_RECORDED, NOT_APPLICABLE):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def performance_rows(inventory):
    baseline = inventory["g2"]
    base_rank1, base_map = _number(baseline.get("rank1")), _number(baseline.get("map"))
    fields = (
        "variant", "baseline", "gate_input", "active_scales", "seed", "selected_epoch",
        "rank1", "map", "rank1_delta_vs_g2", "map_delta_vs_g2", "formal_commit",
        "checkpoint_sha256", "status",
    )
    rows = []
    for spec in VARIANTS:
        item = inventory[spec.key]
        rank1, mean_ap = _number(item.get("rank1")), _number(item.get("map"))
        status = item.get("formal_status", MISSING_FORMAL)
        rows.append({
            "variant": spec.label, "baseline": BASELINE_LABEL,
            "gate_input": spec.gate_input,
            "active_scales": ",".join(str(value) for value in spec.active_scales),
            "seed": item.get("seed", NOT_RECORDED),
            "selected_epoch": item.get("selected_epoch", NOT_RECORDED),
            "rank1": rank1 if rank1 is not None else NOT_APPLICABLE,
            "map": mean_ap if mean_ap is not None else NOT_APPLICABLE,
            "rank1_delta_vs_g2": (
                rank1 - base_rank1 if rank1 is not None and base_rank1 is not None else NOT_APPLICABLE
            ),
            "map_delta_vs_g2": (
                mean_ap - base_map if mean_ap is not None and base_map is not None else NOT_APPLICABLE
            ),
            "formal_commit": item.get("formal_training_commit", NOT_RECORDED),
            "checkpoint_sha256": item.get("selected_checkpoint_sha256", NOT_RECORDED),
            "status": status,
        })
    return fields, rows


def _quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left, right = int(math.floor(position)), int(math.ceil(position))
    if left == right:
        return ordered[left]
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def bootstrap_ci_mean(values, seed, replicates):
    if not values:
        return None, None
    generator = random.Random(seed)
    size = len(values)
    means = []
    for _index in range(int(replicates)):
        means.append(sum(values[generator.randrange(size)] for _ in range(size)) / float(size))
    return _quantile(means, 0.025), _quantile(means, 0.975)


def describe(values, seed, replicates):
    if not values:
        return {key: NOT_APPLICABLE for key in (
            "count", "mean", "std", "min", "max", "median", "q25", "q75", "ci95_low", "ci95_high"
        )}
    average = sum(values) / float(len(values))
    variance = sum((value - average) ** 2 for value in values) / float(len(values))
    low, high = bootstrap_ci_mean(values, seed, replicates)
    return {
        "count": len(values), "mean": average, "std": math.sqrt(variance),
        "min": min(values), "max": max(values), "median": _quantile(values, 0.5),
        "q25": _quantile(values, 0.25), "q75": _quantile(values, 0.75),
        "ci95_low": low, "ci95_high": high,
    }


def read_fixed_manifest(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    fields = ("stable_sample_key", "legacy_sample_key", "selection_hash", "split", "relative_path", "pid", "camid")
    if not rows or tuple(rows[0].keys()) != fields:
        raise AnalysisError("Fixed candidate manifest has an invalid schema")
    hashes = [row["selection_hash"] for row in rows]
    if hashes != sorted(hashes) or len(set(hashes)) != len(hashes):
        raise AnalysisError("Fixed candidate manifest is not uniquely hash-sorted")
    return rows


def _market_directories(market_root):
    root = Path(market_root)
    # This project loader uses ``market1501``; retaining the common upstream
    # directory spellings makes the analysis tool portable without guessing a
    # dataset path from any gating evidence.
    options = (root, root / "market1501", root / "Market-1501-v15.09")
    for candidate in options:
        if (candidate / "query").is_dir() and (candidate / "bounding_box_test").is_dir():
            return candidate / "query", candidate / "bounding_box_test"
    raise AnalysisError("Market1501 query/bounding_box_test directories are absent")


def build_fixed_manifest(market_root, destination, limit):
    query_dir, gallery_dir = _market_directories(market_root)
    candidates = []
    expression = re.compile(r"^([-\d]+)_c(\d+)")
    for split, directory in (("query", query_dir), ("gallery", gallery_dir)):
        for image in sorted(directory.glob("*.jpg")):
            matched = expression.match(image.name)
            if not matched:
                continue
            pid, camid = int(matched.group(1)), int(matched.group(2)) - 1
            # ``tools.analyze_dynamic_gating._stable_key`` is rooted at
            # DATASETS.ROOT_DIR, not at the Market directory.  Retaining the
            # ``market1501/`` segment yields a legacy hash that can be paired
            # with existing gate TSVs while keeping the human-readable key
            # explicit about split/path/pid/camid.
            relative = image.relative_to(directory.parent.parent).as_posix()
            stable = "{}|{}|{}|{}".format(split, relative, pid, camid)
            candidates.append({
                "stable_sample_key": stable,
                "legacy_sample_key": hashlib.sha256(stable.encode("utf-8")).hexdigest(),
                "selection_hash": hashlib.sha256((FIXED_SELECTION_PREFIX + stable).encode("utf-8")).hexdigest(),
                "split": split, "relative_path": relative, "pid": pid, "camid": camid,
            })
    candidates.sort(key=lambda row: row["selection_hash"])
    if limit is not None:
        candidates = candidates[:int(limit)]
    if not candidates:
        raise AnalysisError("No Market1501 candidates were parsed")
    _write_csv(destination, tuple(candidates[0]), candidates, delimiter="\t")
    return candidates


def read_gating_samples(path, spec, expected_checkpoint_sha256=None):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = ["stable_sample_key", "dataset_split", "pid", "camid"]
    expected += ["p{}".format(scale) for scale in spec.active_scales]
    expected += ["w{}".format(scale) for scale in spec.active_scales]
    expected += ["entropy", "dominant_k", "checkpoint_sha256"]
    if not rows or tuple(rows[0].keys()) != tuple(expected):
        raise AnalysisError("{} gating TSV schema is incompatible with its active scales".format(spec.label))
    parsed = {}
    for row in rows:
        key = row["stable_sample_key"]
        if key in parsed:
            raise AnalysisError("{} has duplicate gating sample keys".format(spec.label))
        probabilities = {scale: float(row["p{}".format(scale)]) for scale in spec.active_scales}
        weights = {scale: float(row["w{}".format(scale)]) for scale in spec.active_scales}
        if not all(math.isfinite(value) and value >= 0.0 for value in probabilities.values()):
            raise AnalysisError("{} has invalid gate probabilities".format(spec.label))
        if not all(math.isfinite(value) for value in weights.values()):
            raise AnalysisError("{} has invalid native gate weights".format(spec.label))
        if not all(value >= 0.0 for value in weights.values()):
            raise AnalysisError("{} has negative native gate weights".format(spec.label))
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-6, abs_tol=1e-9):
            raise AnalysisError("{} probabilities do not sum to one".format(spec.label))
        native_sum = sum(weights.values())
        if not math.isclose(native_sum, spec.expected_native_weight_sum, rel_tol=1e-6, abs_tol=1e-8):
            raise AnalysisError(
                "{} native applied weights sum to {!r}, expected {!r}".format(
                    spec.label, native_sum, spec.expected_native_weight_sum
                )
            )
        # The applied weights must be one shared scale factor times the
        # softmax probabilities.  This catches TSVs from a different model
        # variant rather than silently treating them as this experiment.
        for scale in spec.active_scales:
            expected_weight = spec.expected_native_weight_sum * probabilities[scale]
            if not math.isclose(weights[scale], expected_weight, rel_tol=1e-6, abs_tol=1e-8):
                raise AnalysisError(
                    "{} native weight w{} is incompatible with p{}".format(
                        spec.label, scale, scale
                    )
                )
        if expected_checkpoint_sha256 and row["checkpoint_sha256"] != expected_checkpoint_sha256:
            raise AnalysisError("{} gating TSV checkpoint SHA256 mismatches formal record".format(spec.label))
        dominant = int(row["dominant_k"])
        expected_dominant = max(probabilities, key=probabilities.get)
        if dominant != expected_dominant:
            raise AnalysisError("{} has an invalid dominant scale".format(spec.label))
        parsed[key] = {
            "probabilities": probabilities, "weights": weights,
            "dominant": dominant, "checkpoint_sha256": row["checkpoint_sha256"],
        }
    return parsed


def pair_fixed_samples(candidates, samples, spec):
    missing = [row["stable_sample_key"] for row in candidates if row["legacy_sample_key"] not in samples]
    if missing:
        raise AnalysisError(
            "{} lacks {} fixed samples; regenerate fixed-sample gate evidence instead of subsetting".format(
                spec.label, len(missing)
            )
        )
    result = {}
    for candidate in candidates:
        result[candidate["stable_sample_key"]] = samples[candidate["legacy_sample_key"]]
    return result


def collapse_metrics(sample_map, active_scales):
    output = []
    active_count = len(active_scales)
    for key, sample in sample_map.items():
        probabilities = [sample["probabilities"][scale] for scale in active_scales]
        entropy = -sum(value * math.log(max(value, sys.float_info.min)) for value in probabilities)
        ordered = sorted(probabilities, reverse=True)
        output.append({
            "stable_sample_key": key,
            "raw_entropy": entropy,
            "normalized_entropy": entropy / math.log(active_count) if active_count > 1 else 1.0,
            "maximum_probability": ordered[0],
            "first_second_margin": ordered[0] - ordered[1] if active_count > 1 else 0.0,
            "effective_active_scales": math.exp(entropy),
            "normalized_effective_active_scales": math.exp(entropy) / float(active_count),
            "uniform_total_variation": 0.5 * sum(abs(value - 1.0 / active_count) for value in probabilities),
            "dominant": sample["dominant"],
        })
    return output


WEIGHT_FIELDS = (
    "baseline", "variant", "gate_input", "active_scales", "weight", "statistic_type", "status",
    "native_weight_sum", "count", "mean", "std", "min", "max", "median", "q25", "q75",
    "dominant_ratio", "ci95_low", "ci95_high", "baseline_mean", "delta_vs_g2",
)
COLLAPSE_FIELDS = (
    "baseline", "variant", "gate_input", "active_scales", "status", "metric", "count", "mean",
    "std", "median", "q25", "q75", "ci95_low", "ci95_high", "baseline_mean", "delta_vs_g2",
)


def unavailable_weight_rows(spec, status):
    rows = []
    for scale in (2, 4, 6):
        scale_status = "active_{}_unavailable".format(status) if scale in spec.active_scales else "excluded"
        for statistic_type in ("native_applied_weight", "normalized_probability"):
            rows.append({
                "baseline": BASELINE_LABEL, "variant": spec.label, "gate_input": spec.gate_input,
                "active_scales": ",".join(str(item) for item in spec.active_scales),
                "weight": "w{}".format(scale), "statistic_type": statistic_type,
                "status": scale_status,
                "native_weight_sum": NOT_APPLICABLE,
                **{field: NOT_APPLICABLE for field in WEIGHT_FIELDS if field not in (
                    "baseline", "variant", "gate_input", "active_scales", "weight", "statistic_type", "status", "native_weight_sum"
                )},
            })
    return rows


def unavailable_collapse_rows(spec, status):
    return [{
        "baseline": BASELINE_LABEL, "variant": spec.label, "gate_input": spec.gate_input,
        "active_scales": ",".join(str(item) for item in spec.active_scales),
        "status": status, "metric": metric,
        **{field: NOT_APPLICABLE for field in COLLAPSE_FIELDS if field not in (
            "baseline", "variant", "gate_input", "active_scales", "status", "metric"
        )},
    } for metric in (
        "dominant_k2_ratio", "dominant_k4_ratio", "dominant_k6_ratio", "raw_entropy",
        "normalized_entropy", "maximum_probability", "first_second_margin",
        "effective_active_scales", "normalized_effective_active_scales",
        "uniform_total_variation", "dominant_branch_concentration",
    )]


def weight_statistics_rows(spec, sample_map, baseline_map, seed, replicates):
    rows = []
    native_sums = [sum(sample["weights"].values()) for sample in sample_map.values()]
    for scale in (2, 4, 6):
        if scale not in spec.active_scales:
            rows.extend(unavailable_weight_rows(spec, "excluded")[2 * (scale // 2 - 1):2 * (scale // 2)])
            continue
        for statistic_type, source in (("native_applied_weight", "weights"), ("normalized_probability", "probabilities")):
            values = [sample[source][scale] for sample in sample_map.values()]
            baseline_values = [sample[source][scale] for sample in baseline_map.values()] if scale in (2, 4, 6) else []
            record = {
                "baseline": BASELINE_LABEL, "variant": spec.label, "gate_input": spec.gate_input,
                "active_scales": ",".join(str(item) for item in spec.active_scales),
                "weight": "w{}".format(scale), "statistic_type": statistic_type, "status": "active",
                "native_weight_sum": describe(native_sums, seed, replicates)["mean"],
            }
            record.update(describe(values, seed, replicates))
            record["dominant_ratio"] = sum(
                1 for sample in sample_map.values() if sample["dominant"] == scale
            ) / float(len(sample_map))
            if scale in SPEC_BY_KEY["g2"].active_scales:
                baseline_mean = describe(baseline_values, seed, replicates)["mean"]
                record["baseline_mean"] = baseline_mean
                record["delta_vs_g2"] = record["mean"] - baseline_mean
            else:
                record["baseline_mean"] = NOT_APPLICABLE
                record["delta_vs_g2"] = NOT_APPLICABLE
            rows.append(record)
    return rows


def collapse_rows_for_pair(spec, sample_map, baseline_map, seed, replicates):
    current = collapse_metrics(sample_map, spec.active_scales)
    baseline = collapse_metrics(baseline_map, SPEC_BY_KEY["g2"].active_scales)
    current_by_key, baseline_by_key = (
        {row["stable_sample_key"]: row for row in current},
        {row["stable_sample_key"]: row for row in baseline},
    )
    metrics = (
        "raw_entropy", "normalized_entropy", "maximum_probability", "first_second_margin",
        "effective_active_scales", "normalized_effective_active_scales", "uniform_total_variation",
    )
    rows = []
    for metric in metrics:
        values = [row[metric] for row in current]
        base_values = [baseline_by_key[key][metric] for key in current_by_key]
        row = {
            "baseline": BASELINE_LABEL, "variant": spec.label, "gate_input": spec.gate_input,
            "active_scales": ",".join(str(item) for item in spec.active_scales),
            "status": "active", "metric": metric,
        }
        row.update(describe(values, seed, replicates))
        row["baseline_mean"] = describe(base_values, seed, replicates)["mean"]
        row["delta_vs_g2"] = row["mean"] - row["baseline_mean"]
        rows.append(row)
    dominant = Counter(row["dominant"] for row in current)
    baseline_dominant = Counter(row["dominant"] for row in baseline)
    for scale in (2, 4, 6):
        row = {
            "baseline": BASELINE_LABEL, "variant": spec.label, "gate_input": spec.gate_input,
            "active_scales": ",".join(str(item) for item in spec.active_scales),
            "metric": "dominant_k{}_ratio".format(scale),
        }
        if scale not in spec.active_scales:
            row.update({"status": "excluded", **{field: NOT_APPLICABLE for field in COLLAPSE_FIELDS if field not in row and field != "status"}})
        else:
            value = dominant[scale] / float(len(current))
            baseline_value = baseline_dominant[scale] / float(len(baseline))
            row.update({
                "status": "active", "count": len(current), "mean": value, "std": NOT_APPLICABLE,
                "median": NOT_APPLICABLE, "q25": NOT_APPLICABLE, "q75": NOT_APPLICABLE,
                "ci95_low": NOT_APPLICABLE, "ci95_high": NOT_APPLICABLE,
                "baseline_mean": baseline_value, "delta_vs_g2": value - baseline_value,
            })
        rows.append(row)
    concentration = max(dominant.values()) / float(len(current))
    base_concentration = max(baseline_dominant.values()) / float(len(baseline))
    rows.append({
        "baseline": BASELINE_LABEL, "variant": spec.label, "gate_input": spec.gate_input,
        "active_scales": ",".join(str(item) for item in spec.active_scales),
        "status": "active", "metric": "dominant_branch_concentration", "count": len(current),
        "mean": concentration, "std": NOT_APPLICABLE, "median": NOT_APPLICABLE,
        "q25": NOT_APPLICABLE, "q75": NOT_APPLICABLE, "ci95_low": NOT_APPLICABLE,
        "ci95_high": NOT_APPLICABLE, "baseline_mean": base_concentration,
        "delta_vs_g2": concentration - base_concentration,
    })
    return rows


def paired_sample_rows(spec, sample_map, baseline_map):
    """Create the CSV source of every paired plot without rounding values."""
    rows = []
    for sample_key in sorted(sample_map):
        current, baseline = sample_map[sample_key], baseline_map[sample_key]
        row = {
            "stable_sample_key": sample_key,
            "baseline": BASELINE_LABEL,
            "variant": spec.label,
            "gate_input": spec.gate_input,
            "active_scales": ",".join(str(scale) for scale in spec.active_scales),
            "baseline_dominant_k": baseline["dominant"],
            "variant_dominant_k": current["dominant"],
        }
        for scale in (2, 4, 6):
            if scale in spec.active_scales:
                row["variant_p{}".format(scale)] = current["probabilities"][scale]
                row["variant_w{}".format(scale)] = current["weights"][scale]
                row["delta_p{}_vs_g2".format(scale)] = (
                    current["probabilities"][scale] - baseline["probabilities"][scale]
                )
            else:
                row["variant_p{}".format(scale)] = "excluded"
                row["variant_w{}".format(scale)] = "excluded"
                row["delta_p{}_vs_g2".format(scale)] = "excluded"
            row["g2_p{}".format(scale)] = baseline["probabilities"][scale]
            row["g2_w{}".format(scale)] = baseline["weights"][scale]
        rows.append(row)
    return rows


PAIRED_SAMPLE_FIELDS = (
    "stable_sample_key", "baseline", "variant", "gate_input", "active_scales",
    "baseline_dominant_k", "variant_dominant_k",
    "g2_p2", "g2_p4", "g2_p6", "g2_w2", "g2_w4", "g2_w6",
    "variant_p2", "variant_p4", "variant_p6", "variant_w2", "variant_w4", "variant_w6",
    "delta_p2_vs_g2", "delta_p4_vs_g2", "delta_p6_vs_g2",
)


def _get_matplotlib():
    """Import Matplotlib only after strict evidence pairing succeeds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except (ImportError, RuntimeError) as error:
        raise AnalysisError("Matplotlib is unavailable for figures: {}".format(error))
    return pyplot


def _save_figure(figure, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    png, pdf = directory / (stem + ".png"), directory / (stem + ".pdf")
    figure.savefig(str(png), dpi=300, bbox_inches="tight")
    figure.savefig(str(pdf), bbox_inches="tight")
    return (png, pdf)


def _boxplot(axis, data, labels, **kwargs):
    """Use Matplotlib's renamed tick-label argument compatibly."""
    label_keyword = "tick_labels" if "tick_labels" in inspect.signature(axis.boxplot).parameters else "labels"
    kwargs[label_keyword] = labels
    return axis.boxplot(data, **kwargs)


def generate_pair_figures(spec, sample_map, baseline_map, output_directory):
    """Generate reproducible paired distributions only after all samples pair.

    These figures intentionally use the full frozen manifest, never a
    hand-picked display subset.  The underlying paired values are separately
    written to CSV by ``run_analysis``.
    """
    pyplot = _get_matplotlib()
    output_directory = Path(output_directory)
    colors = {2: "#1f77b4", 4: "#ff7f0e", 6: "#2ca02c"}
    baseline_values = {
        scale: [item["probabilities"][scale] for item in baseline_map.values()]
        for scale in (2, 4, 6)
    }
    current_values = {
        scale: [item["probabilities"][scale] for item in sample_map.values()]
        for scale in spec.active_scales
    }
    produced = []

    # Normalised-probability distributions.  Both rows share a fixed [0,1]
    # x-axis so a comparator cannot be visually exaggerated by rescaling.
    figure, axes = pyplot.subplots(2, 1, figsize=(8.4, 5.8), sharex=True)
    for scale in (2, 4, 6):
        axes[0].hist(baseline_values[scale], bins=24, range=(0.0, 1.0), density=True,
                     histtype="step", linewidth=1.6, color=colors[scale], label="G2 p{}".format(scale))
        if scale in spec.active_scales:
            axes[1].hist(current_values[scale], bins=24, range=(0.0, 1.0), density=True,
                         histtype="step", linewidth=1.6, color=colors[scale], label="{} p{}".format(spec.label, scale))
        else:
            axes[1].plot([], [], color=colors[scale], label="{} p{} excluded".format(spec.label, scale))
    axes[0].set_title("G2 baseline normalised gate probabilities")
    axes[1].set_title("{} normalised gate probabilities".format(spec.label))
    for axis in axes:
        axis.set_ylabel("density")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    axes[1].set_xlabel("normalised gate probability")
    produced += _save_figure(figure, output_directory, "normalised_probability_histograms")
    pyplot.close(figure)

    # Box plots use the same [0,1] probability axis and retain a labelled
    # blank/excluded position for a structurally absent scale.
    figure, axis = pyplot.subplots(figsize=(8.4, 4.8))
    box_data, box_labels, box_colors = [], [], []
    for scale in (2, 4, 6):
        box_data.append(baseline_values[scale])
        box_labels.append("G2 p{}".format(scale))
        box_colors.append("#4c78a8")
        if scale in spec.active_scales:
            box_data.append(current_values[scale])
            box_labels.append("{} p{}".format(spec.label, scale))
            box_colors.append(colors[scale])
        else:
            box_data.append([float("nan")])
            box_labels.append("{} p{} excluded".format(spec.label, scale))
            box_colors.append("#bbbbbb")
    boxes = _boxplot(axis, box_data, box_labels, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("normalised gate probability")
    axis.set_title("Normalised probability box plots: G2 vs {}".format(spec.label))
    axis.tick_params(axis="x", rotation=24)
    axis.grid(axis="y", alpha=0.2)
    produced += _save_figure(figure, output_directory, "normalised_probability_boxplots")
    pyplot.close(figure)

    # Dominant-scale fractions; excluded scale is visibly labelled instead of
    # being represented by a spurious zero-height probability bar.
    scales = (2, 4, 6)
    baseline_dominant = Counter(item["dominant"] for item in baseline_map.values())
    current_dominant = Counter(item["dominant"] for item in sample_map.values())
    figure, axis = pyplot.subplots(figsize=(7.2, 4.4))
    positions = list(range(len(scales)))
    width = 0.34
    axis.bar([position - width / 2 for position in positions],
             [baseline_dominant[scale] / float(len(baseline_map)) for scale in scales],
             width=width, color="#4c78a8", label="G2")
    active_values = []
    for scale in scales:
        active_values.append(
            current_dominant[scale] / float(len(sample_map)) if scale in spec.active_scales else 0.0
        )
    axis.bar([position + width / 2 for position in positions], active_values, width=width,
             color="#f58518", label=spec.label)
    for position, scale in enumerate(scales):
        if scale not in spec.active_scales:
            axis.text(position + width / 2, 0.02, "excluded", ha="center", va="bottom",
                      rotation=90, fontsize=8)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(["K{}".format(scale) for scale in scales])
    axis.set_ylabel("dominant-scale fraction")
    axis.set_title("Dominant scale: G2 vs {}".format(spec.label))
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    produced += _save_figure(figure, output_directory, "dominant_scale_ratio")
    pyplot.close(figure)

    # Matched-pair probability deltas.  Each active scale uses the same y
    # axis; x is the hash-sorted fixed sample ordinal rather than a cherry-
    # picked image ordering.
    figure, axis = pyplot.subplots(figsize=(8.4, 4.4))
    ordered_keys = sorted(sample_map)
    for scale in spec.active_scales:
        values = [
            sample_map[key]["probabilities"][scale] - baseline_map[key]["probabilities"][scale]
            for key in ordered_keys
        ]
        axis.scatter(range(len(values)), values, s=9, alpha=0.7, color=colors[scale], label="Δp{}".format(scale))
    axis.axhline(0.0, linewidth=1.0, color="black")
    axis.set_xlabel("fixed hash-sorted sample ordinal")
    axis.set_ylabel("variant probability − G2 probability")
    axis.set_title("Paired normalised probability deltas: {} vs G2".format(spec.label))
    axis.grid(alpha=0.2)
    axis.legend()
    produced += _save_figure(figure, output_directory, "paired_probability_deltas")
    pyplot.close(figure)

    # These metrics are all computed on the same fixed samples.  Raw entropy
    # is deliberately not plotted here because its upper bound changes from
    # log(3) to log(2); normalized entropy is the cross-model comparison.
    current_collapse = collapse_metrics(sample_map, spec.active_scales)
    baseline_collapse = collapse_metrics(baseline_map, (2, 4, 6))
    figure, axes = pyplot.subplots(1, 3, figsize=(11.8, 3.9))
    for axis, metric, title, lower, upper in (
            (axes[0], "normalized_entropy", "normalised entropy", 0.0, 1.0),
            (axes[1], "maximum_probability", "maximum probability", 0.0, 1.0),
            (axes[2], "first_second_margin", "first–second margin", 0.0, 1.0)):
        boxes = _boxplot(
            axis,
            [[item[metric] for item in baseline_collapse], [item[metric] for item in current_collapse]],
            ["G2", spec.label], patch_artist=True, showfliers=False,
        )
        boxes["boxes"][0].set_facecolor("#4c78a8")
        boxes["boxes"][1].set_facecolor("#f58518")
        for box in boxes["boxes"]:
            box.set_alpha(0.65)
        axis.set_ylim(lower, upper)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Fixed-sample collapse metrics: G2 vs {}".format(spec.label))
    produced += _save_figure(figure, output_directory, "collapse_metric_boxplots")
    pyplot.close(figure)

    # The transition matrix is calculated from paired fixed samples, so it is
    # meaningful only when strict full coverage has already passed.
    figure, axis = pyplot.subplots(figsize=(5.2, 4.5))
    matrix = [[0 for _column in scales] for _row in scales]
    for key in ordered_keys:
        matrix[scales.index(baseline_map[key]["dominant"])][scales.index(sample_map[key]["dominant"])] += 1
    image = axis.imshow(matrix, vmin=0, vmax=max(1, len(ordered_keys)), cmap="Blues")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            if scales[column_index] in spec.active_scales:
                axis.text(column_index, row_index, str(value), ha="center", va="center")
            else:
                axis.text(column_index, row_index, "excluded", ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(scales)))
    axis.set_yticks(range(len(scales)))
    axis.set_xticklabels(["{} K{}".format(spec.label, scale) for scale in scales], rotation=25, ha="right")
    axis.set_yticklabels(["G2 K{}".format(scale) for scale in scales])
    axis.set_title("Paired dominant-scale transition")
    figure.colorbar(image, ax=axis, label="fixed sample count")
    produced += _save_figure(figure, output_directory, "dominant_scale_transition")
    pyplot.close(figure)
    return [str(path) for path in produced]


def write_missing_formal_commands(path, inventory):
    lines = ["# 缺少正式证据的 AutoDL 命令", "", "这些命令不会使用 smoke 指标替代正式结果。执行前请确认分支 SHA 并保持工作树干净。", ""]
    for spec in COMPARATORS:
        if inventory[spec.key].get("formal_status") != MISSING_FORMAL:
            continue
        lines.extend([
            "## {}".format(spec.label), "", "```bash",
            "git fetch origin {}".format(spec.branch.replace("origin/", "")),
            "git switch {}".format(spec.branch.replace("origin/", "")),
            "test -z \"$(git status --porcelain=v1 --untracked-files=all)\" || { echo 'Dirty worktree'; exit 1; }",
            "bash scripts/{}".format(spec.smoke_script),
            "bash scripts/{}".format(spec.formal_script),
            "```", "",
        ])
    if len(lines) == 4:
        lines.extend(["当前五个 comparator 均已有正式登记；仍需归档可校验的原始门控证据。", ""])
    _atomic_text(path, "\n".join(lines))


def write_annotation_template(path):
    fields = ("stable_sample_key", "category", "annotation_version", "reason")
    if not Path(path).exists():
        _write_csv(path, fields, [], delimiter="\t")
    readme = Path(path).with_name("IMAGE_TYPE_ANNOTATION_README.md")
    _atomic_text(readme, """# 固定样本盲标注说明

只能在未展示任何模型权重、检索指标或预测结果的情况下查看图片后填写
`image_type_annotations.tsv`。允许同一 `stable_sample_key` 多行，从而表达多标签。
类别限定为：`clear`、`occluded`、`misaligned`、`side_view`、`back_view`、`blurred`。
每个类别按固定 manifest 的 `selection_hash` 升序选取，不能依据任何门控统计补选图片。
""")


IMAGE_CATEGORIES = ("clear", "occluded", "misaligned", "side_view", "back_view", "blurred")


def read_blind_annotations(path, candidates):
    """Read only a pre-existing blind annotation TSV; never infer categories."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    fields = ("stable_sample_key", "category", "annotation_version", "reason")
    if not rows:
        return []
    if tuple(rows[0].keys()) != fields:
        raise AnalysisError("Image-type annotation TSV has an invalid schema")
    available = {row["stable_sample_key"] for row in candidates}
    seen = set()
    for row in rows:
        identity = (row["stable_sample_key"], row["category"])
        if row["stable_sample_key"] not in available:
            raise AnalysisError("Blind annotation refers to a non-fixed sample")
        if row["category"] not in IMAGE_CATEGORIES:
            raise AnalysisError("Blind annotation has an unsupported category")
        if not row["annotation_version"] or not row["reason"]:
            raise AnalysisError("Blind annotation must record version and reason")
        if identity in seen:
            raise AnalysisError("Blind annotation duplicates a sample/category pair")
        seen.add(identity)
    return rows


def _candidate_image_path(market_root, candidate):
    root = Path(market_root)
    relative = Path(candidate["relative_path"])
    direct = root / relative
    if direct.is_file():
        return direct
    # ``relative_path`` contains market1501/ when Market root is its parent.
    # If the user supplied the Market directory itself, discard that leading
    # component instead of guessing any other dataset location.
    if relative.parts and relative.parts[0].lower() in ("market1501", "market-1501-v15.09"):
        nested = root / Path(*relative.parts[1:])
        if nested.is_file():
            return nested
    return None


def _gate_text(label, sample, active_scales, baseline_sample=None):
    fields = []
    for scale in (2, 4, 6):
        if scale not in active_scales:
            fields.append("K{}: excluded".format(scale))
            continue
        value = "K{}: p={:.4f}, w={:.4f}".format(
            scale, sample["probabilities"][scale], sample["weights"][scale]
        )
        if baseline_sample is not None:
            value += ", Δp={:+.4f}".format(
                sample["probabilities"][scale] - baseline_sample["probabilities"][scale]
            )
        fields.append(value)
    return "{} | dominant K{}\n{}".format(label, sample["dominant"], "; ".join(fields))


def generate_fixed_sample_panels(spec, sample_map, baseline_map, candidates, annotations,
                                 market_root, output_directory, per_category):
    """Render blind-annotated fixed sample panels once evidence is paired.

    Category membership is read verbatim from the user-maintained blind TSV;
    this function does not inspect weights while selecting the category or
    rank.  It merely filters previously frozen candidates and orders each
    category by its immutable selection hash.
    """
    if not annotations:
        return []
    pyplot = _get_matplotlib()
    candidate_by_key = {row["stable_sample_key"]: row for row in candidates}
    annotation_keys = {}
    for annotation in annotations:
        annotation_keys.setdefault(annotation["category"], set()).add(annotation["stable_sample_key"])
    paths = []
    for category in IMAGE_CATEGORIES:
        selected = [
            candidate_by_key[key] for key in annotation_keys.get(category, set())
            if key in sample_map and key in baseline_map
        ]
        selected.sort(key=lambda row: row["selection_hash"])
        selected = selected[:int(per_category)]
        if not selected:
            continue
        figure, axes = pyplot.subplots(len(selected), 1, figsize=(9.4, 4.7 * len(selected)))
        if len(selected) == 1:
            axes = [axes]
        for axis, candidate in zip(axes, selected):
            image_path = _candidate_image_path(market_root, candidate)
            if image_path is None:
                pyplot.close(figure)
                raise AnalysisError("Fixed sample image is absent: {}".format(candidate["relative_path"]))
            axis.imshow(pyplot.imread(str(image_path)))
            axis.axis("off")
            key = candidate["stable_sample_key"]
            title = (
                "{} | split={} pid={} camid={} | category={}\n{}\n{}"
            ).format(
                candidate["relative_path"], candidate["split"], candidate["pid"], candidate["camid"], category,
                _gate_text("G2", baseline_map[key], (2, 4, 6)),
                _gate_text(spec.label, sample_map[key], spec.active_scales, baseline_map[key]),
            )
            axis.set_title(title, fontsize=8, loc="left")
        figure.suptitle("Fixed blind-annotated samples: G2 vs {} ({})".format(spec.label, category), y=1.0)
        paths += _save_figure(figure, Path(output_directory) / category, "g2_vs_{}".format(spec.key))
        pyplot.close(figure)
    return [str(path) for path in paths]


def write_output_checksums(output_dir):
    output_dir = Path(output_dir)
    rows = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.name != "SHA256SUMS.txt"):
        rows.append("{}  {}".format(sha256_file(path), path.relative_to(output_dir).as_posix()))
    _atomic_text(output_dir / "SHA256SUMS.txt", "\n".join(rows) + ("\n" if rows else ""))


def run_analysis(repo_root, output_dir, artifact_roots=(), market_root=None,
                 fixed_manifest=None, fixed_limit=256, bootstrap_seed=42,
                 bootstrap_replicates=1000, display_per_category=10):
    """Run the audit and only compute paired gate statistics when evidence exists."""
    repo_root, output_dir = Path(repo_root).resolve(), Path(output_dir).resolve()
    tables_dir, manifests_dir, figures_dir, samples_dir, evidence_dir = (
        output_dir / "tables", output_dir / "manifests", output_dir / "figures", output_dir / "samples", output_dir / "evidence"
    )
    for directory in (tables_dir, manifests_dir, figures_dir, samples_dir, evidence_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for spec in COMPARATORS:
        (figures_dir / "g2_vs_{}".format(spec.key)).mkdir(exist_ok=True)
    for category in ("clear", "occluded", "misaligned", "side_view", "back_view", "blurred"):
        (samples_dir / category).mkdir(exist_ok=True)

    inventory, evidence = {}, {}
    for spec in VARIANTS:
        row, details = audit_variant(repo_root, spec, artifact_roots)
        inventory[spec.key], evidence[spec.key] = row, details
    validate_baseline_record(inventory, evidence)

    inventory_fields = (
        "variant", "baseline", "branch", "branch_tip", "expected_experiment_id", "formal_status",
        "implementation_contract", "expected_native_weight_sum", "formal_evidence_access", "training_branch", "formal_training_commit", "source_config_sha256",
        "resolved_config_sha256", "seed", "dataset", "temperature", "normalization", "scale_order",
        "selected_checkpoint_sha256", "selected_epoch", "rank1", "map", "query_gallery_protocol",
        "source_config_artifact_status", "resolved_config_artifact_status", "selected_checkpoint_artifact_status",
        "gating_samples_artifact_status", "dynamic_gating_summary_artifact_status", "audit_error",
    )
    inventory_rows = [inventory[spec.key] for spec in VARIANTS]
    _write_csv(output_dir / "evidence_inventory.csv", inventory_fields, inventory_rows)
    _write_markdown_from_rows(output_dir / "evidence_inventory.md", "六个 Dynamic Gating 版本证据库存", inventory_fields, inventory_rows,
                              "训练 commit 与后续记录 branch tip 分列保存；所有 comparator 的统计 baseline 固定为 G2-global-local。")
    _write_json(output_dir / "evidence_inventory.json", {"baseline": BASELINE_LABEL, "rows": inventory_rows})
    _write_csv(evidence_dir / "artifact_access_inventory.csv", inventory_fields, inventory_rows)
    _atomic_text(evidence_dir / "README.md", """# 原始证据访问边界

此目录仅保存机器生成的路径、SHA256 和可访问性台账，绝不复制 Market1501 图像、checkpoint、训练日志或原始门控 TSV。`formal_registered_artifacts_not_archived` 表示分支 Git 表中已登记 SHA256，但当前分析工作区没有同 SHA256 的原始文件；该状态不能用于生成门控统计或图表。
""")

    perf_fields, perf_rows = performance_rows(inventory)
    _write_csv(tables_dir / "performance_vs_g2.csv", perf_fields, perf_rows)
    _write_markdown_from_rows(tables_dir / "performance_vs_g2.md", "性能与统一 G2 baseline 的比较", perf_fields, perf_rows,
                              "所有 delta 均相对 G2-global-local；缺少对应 formal run 的行不含虚构指标。")
    write_missing_formal_commands(output_dir / "missing_formal_evidence_commands.md", inventory)
    write_annotation_template(manifests_dir / "image_type_annotations.tsv")

    if fixed_manifest:
        candidates = read_fixed_manifest(fixed_manifest)
        fixed_manifest_path = Path(fixed_manifest).resolve()
    elif market_root:
        fixed_manifest_path = manifests_dir / "fixed_candidate_samples.tsv"
        candidates = build_fixed_manifest(market_root, fixed_manifest_path, fixed_limit)
    else:
        candidates, fixed_manifest_path = None, None

    weight_rows, collapse_rows, data_status, figure_paths, sample_panel_paths = [], [], {}, {}, {}
    if candidates is None:
        for spec in VARIANTS:
            status = MISSING_FORMAL if inventory[spec.key].get("formal_status") != "success" else "fixed_candidate_manifest_required"
            weight_rows.extend(unavailable_weight_rows(spec, status))
        for spec in COMPARATORS:
            status = MISSING_FORMAL if inventory[spec.key].get("formal_status") != "success" else "fixed_candidate_manifest_required"
            collapse_rows.extend(unavailable_collapse_rows(spec, status))
        data_status["fixed_samples"] = "not_generated: supply --market-root or --fixed-manifest before any TSV is read"
    else:
        annotations = read_blind_annotations(
            manifests_dir / "image_type_annotations.tsv", candidates
        )
        paired = {}
        for spec in VARIANTS:
            detail = evidence[spec.key]
            if detail is None:
                data_status[spec.key] = MISSING_FORMAL
                continue
            if inventory[spec.key].get("implementation_contract") != "confirmed":
                data_status[spec.key] = "implementation_contract_not_confirmed"
                continue
            source = detail["sources"].get("gating_samples", {})
            path = source.get("path")
            if not path or path == NOT_APPLICABLE:
                data_status[spec.key] = "gating_samples_not_archived"
                continue
            try:
                paired[spec.key] = pair_fixed_samples(
                    candidates,
                    read_gating_samples(
                        path, spec,
                        expected_checkpoint_sha256=inventory[spec.key].get("selected_checkpoint_sha256"),
                    ),
                    spec,
                )
                data_status[spec.key] = "paired_fixed_samples_ready"
            except AnalysisError as error:
                data_status[spec.key] = "fixed_sample_coverage_incomplete: {}".format(error)
        baseline_map = paired.get("g2")
        for spec in VARIANTS:
            if spec.key in paired and baseline_map is not None:
                weight_rows.extend(weight_statistics_rows(spec, paired[spec.key], baseline_map, bootstrap_seed, bootstrap_replicates))
            else:
                weight_rows.extend(unavailable_weight_rows(spec, data_status.get(spec.key, "unavailable")))
        for spec in COMPARATORS:
            if spec.key in paired and baseline_map is not None:
                collapse_rows.extend(collapse_rows_for_pair(spec, paired[spec.key], baseline_map, bootstrap_seed, bootstrap_replicates))
                paired_rows = paired_sample_rows(spec, paired[spec.key], baseline_map)
                paired_path = tables_dir / "paired_samples_g2_vs_{}.csv".format(spec.key)
                _write_csv(paired_path, PAIRED_SAMPLE_FIELDS, paired_rows)
                try:
                    figure_paths[spec.key] = generate_pair_figures(
                        spec, paired[spec.key], baseline_map,
                        figures_dir / "g2_vs_{}".format(spec.key),
                    )
                except AnalysisError as error:
                    figure_paths[spec.key] = "not_generated: {}".format(error)
                if market_root and annotations:
                    try:
                        sample_panel_paths[spec.key] = generate_fixed_sample_panels(
                            spec, paired[spec.key], baseline_map, candidates, annotations,
                            market_root, samples_dir, display_per_category,
                        )
                    except AnalysisError as error:
                        sample_panel_paths[spec.key] = "not_generated: {}".format(error)
            else:
                collapse_rows.extend(unavailable_collapse_rows(spec, data_status.get(spec.key, "unavailable")))
        _write_json(manifests_dir / "fixed_candidate_manifest_provenance.json", {
            "selection_prefix": FIXED_SELECTION_PREFIX,
            "path": str(fixed_manifest_path), "sha256": sha256_file(fixed_manifest_path),
            "sample_count": len(candidates), "selection_before_weights": True,
        })

    _write_csv(tables_dir / "gating_weight_statistics.csv", WEIGHT_FIELDS, weight_rows)
    _write_markdown_from_rows(tables_dir / "gating_weight_statistics.md", "门控权重统计（统一 G2 baseline）", WEIGHT_FIELDS, weight_rows,
                              "native_applied_weight 为模型实际使用的 w；normalized_probability 为和为 1 的 p。excluded 表示结构未提供该分支，绝不等同于零。")
    _write_json(tables_dir / "gating_weight_statistics.json", {"baseline": BASELINE_LABEL, "rows": weight_rows})
    _write_csv(tables_dir / "gating_collapse_comparison_vs_g2.csv", COLLAPSE_FIELDS, collapse_rows)
    _write_markdown_from_rows(tables_dir / "gating_collapse_comparison_vs_g2.md", "门控坍缩指标与统一 G2 baseline 的比较", COLLAPSE_FIELDS, collapse_rows,
                              "跨两路/三路模型优先解释 normalized_entropy；被删除 scale 标为 excluded/N/A。")
    _write_json(tables_dir / "gating_collapse_comparison_vs_g2.json", {"baseline": BASELINE_LABEL, "rows": collapse_rows})

    _atomic_text(figures_dir / "README.md", """# 图表状态

只有同时满足以下条件时，才允许生成跨版本门控图：固定候选 manifest 已在读取任何权重前冻结；G2 和 comparator 的原始门控 TSV 均可校验；每一个固定样本均可跨版本配对。当前状态详见 `analysis_status.json`。工具不会用不同候选集、smoke 或其他分支的 TSV 生成替代图表。

成功配对时，每组 `g2_vs_*` 都会从对应的 `tables/paired_samples_g2_vs_*.csv` 生成 PNG 和 PDF（300 DPI）：归一化概率直方图、dominant K 比例、配对概率差和 dominant K 转移矩阵。`samples/` 仅用于盲标注后按固定 hash 顺序制作展示面板；未完成盲标注时不会输出挑选的图像面板。
""")
    _atomic_text(samples_dir / "README.md", """# 固定样本图像面板

本目录的 `clear`、`occluded`、`misaligned`、`side_view`、`back_view`、`blurred` 是为盲标注后的展示面板保留的目录。Market1501 不提供这些官方类别；标注必须先于读取门控权重，且写入 `manifests/image_type_annotations.tsv`。当前工具不会把结构删除的 scale 写成 0；它必须显示为 `excluded`。
""")
    status = {
        "baseline": BASELINE_LABEL,
        "baseline_training_commit": BASELINE_TRAINING_COMMIT,
        "baseline_branch_tip": inventory["g2"].get("branch_tip"),
        "fixed_candidate_manifest": str(fixed_manifest_path) if fixed_manifest_path else NOT_APPLICABLE,
        "variant_data_status": data_status,
        "figures": figure_paths if figure_paths else "not_generated_without_complete_paired_fixed_samples",
        "sample_panels": sample_panel_paths if sample_panel_paths else "not_generated_without_blind_annotations_and_complete_paired_fixed_samples",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
    }
    _write_json(output_dir / "analysis_status.json", status)
    write_output_checksums(output_dir)
    return {"inventory": inventory, "performance_rows": perf_rows, "status": status}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "analysis_outputs" / "g2_baseline_all_variants"))
    parser.add_argument("--artifact-root", action="append", default=[],
                        help="Optional root searched by recorded filename and SHA256; repeatable.")
    parser.add_argument("--market-root", default=None,
                        help="Market1501 root used to freeze a candidate manifest before TSV reading.")
    parser.add_argument("--fixed-manifest", default=None,
                        help="Previously frozen fixed_candidate_samples.tsv; mutually exclusive with --market-root.")
    parser.add_argument("--fixed-sample-limit", type=int, default=256)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--display-per-category", type=int, default=10,
                        help="Maximum blind-annotated fixed samples per category (5-10 recommended).")
    args = parser.parse_args(argv)
    if args.market_root and args.fixed_manifest:
        parser.error("--market-root and --fixed-manifest are mutually exclusive")
    if args.fixed_sample_limit <= 0 or args.bootstrap_replicates <= 0 or args.display_per_category <= 0:
        parser.error("sample limit, bootstrap replicates, and display count must be positive")
    result = run_analysis(
        args.repo_root, args.output_dir, artifact_roots=args.artifact_root,
        market_root=args.market_root, fixed_manifest=args.fixed_manifest,
        fixed_limit=args.fixed_sample_limit, bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates, display_per_category=args.display_per_category,
    )
    print(json.dumps(result["status"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
