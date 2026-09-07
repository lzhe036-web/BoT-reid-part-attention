#!/usr/bin/env python
"""Fail closed unless G2-D1 differs from G2-A only in the declared gate input."""

from __future__ import absolute_import

import argparse
import json
from pathlib import Path

import yaml


class ProtocolError(ValueError):
    pass


EXPECTED_COMMON = {
    ("SEED",): 42,
    ("MODEL", "MULTI_GRANULARITY_DYNAMIC_GATING"): True,
    ("MODEL", "MULTI_GRANULARITY_GATING_TAU"): 0.5,
    ("MODEL", "MULTI_GRANULARITY_GATING_NORMALIZATION"): "scaled_softmax",
    ("MODEL", "MULTI_GRANULARITY_PART_SCALES"): [2, 4, 6],
    ("MODEL", "MULTI_GRANULARITY_PART_DIM"): 256,
    ("SOLVER", "MAX_EPOCHS"): 120,
    ("DATASETS", "NAMES"): "market1501",
    ("TEST", "RE_RANKING"): "no",
}
ALLOWED_DIFFERENCES = {
    ("MODEL", "MULTI_GRANULARITY_GATING_INPUT"), ("OUTPUT_DIR",),
}


def _read(path):
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ProtocolError("Cannot read YAML {}: {}".format(path, error))
    if not isinstance(value, dict):
        raise ProtocolError("YAML must be a mapping")
    return value


def _at(value, path):
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ProtocolError("Missing {}".format(".".join(path)))
        value = value[key]
    return value


def _diff(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append((prefix + (key,), left.get(key), right.get(key)))
            else:
                result.extend(_diff(left[key], right[key], prefix + (key,)))
        return result
    return [] if left == right else [(prefix, left, right)]


def verify(baseline_path, candidate_path):
    baseline, candidate = _read(baseline_path), _read(candidate_path)
    for path, expected in EXPECTED_COMMON.items():
        if _at(baseline, path) != expected or _at(candidate, path) != expected:
            raise ProtocolError("Protocol mismatch {}".format(".".join(path)))
    if _at(baseline, ("MODEL", "MULTI_GRANULARITY_GATING_INPUT")) != "concat_global_local":
        raise ProtocolError("Direct baseline is not G2-A concat_global_local")
    if _at(candidate, ("MODEL", "MULTI_GRANULARITY_GATING_INPUT")) != "concat_global_local_diff46":
        raise ProtocolError("Candidate is not concat_global_local_diff46")
    differences = _diff(baseline, candidate)
    unexpected = [item for item in differences if item[0] not in ALLOWED_DIFFERENCES]
    if unexpected:
        raise ProtocolError("Unexpected config differences: {}".format(unexpected))
    return {
        "status": "verified", "baseline_config": str(Path(baseline_path).resolve()),
        "candidate_config": str(Path(candidate_path).resolve()),
        "baseline_commit": "d724a6536e4a819c5d2932412e90b7dea224041b",
        "declared_change": "Gate input [g,z2,z4,z6] -> [g,z2,z4,z6,abs(z4-z6)]; Linear(2816,3) -> Linear(3072,3)",
        "tau_g": 0.5, "retrieval_feature_dim": 2816,
        "allowed_differences": [".".join(path) for path, _left, _right in differences],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.baseline_config, args.candidate_config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
