#!/usr/bin/env python
"""Fail closed unless a τg=0.5 config differs from formal G2 only as allowed."""

from __future__ import absolute_import

import argparse
import json
from pathlib import Path

import yaml


class ProtocolError(ValueError):
    """The candidate would not be a single-variable G2 temperature ablation."""


EXPECTED = {
    ("SEED",): 42,
    ("MODEL", "MULTI_GRANULARITY_DYNAMIC_GATING"): True,
    ("MODEL", "MULTI_GRANULARITY_GATING_INPUT"): "concat_global_local",
    ("MODEL", "MULTI_GRANULARITY_GATING_NORMALIZATION"): "scaled_softmax",
    ("MODEL", "MULTI_GRANULARITY_PART_SCALES"): [2, 4, 6],
    ("SOLVER", "MAX_EPOCHS"): 120,
    ("DATASETS", "NAMES"): "market1501",
    ("TEST", "RE_RANKING"): "no",
}
ALLOWED_LEAF_DIFFERENCES = {
    ("MODEL", "MULTI_GRANULARITY_GATING_TAU"),
    ("OUTPUT_DIR",),
}


def _read(path):
    path = Path(path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ProtocolError("Cannot read YAML {}: {}".format(path, error))
    if not isinstance(payload, dict):
        raise ProtocolError("YAML must be a mapping: {}".format(path))
    return payload


def _at(mapping, path):
    value = mapping
    for item in path:
        if not isinstance(value, dict) or item not in value:
            raise ProtocolError("Missing required setting {}".format(".".join(path)))
        value = value[item]
    return value


def _leaf_differences(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        values = []
        for key in sorted(keys):
            if key not in left or key not in right:
                values.append((prefix + (key,), left.get(key), right.get(key)))
            else:
                values.extend(_leaf_differences(left[key], right[key], prefix + (key,)))
        return values
    return [] if left == right else [(prefix, left, right)]


def verify(baseline_path, candidate_path):
    baseline = _read(baseline_path)
    candidate = _read(candidate_path)
    for path, expected in EXPECTED.items():
        if _at(baseline, path) != expected or _at(candidate, path) != expected:
            raise ProtocolError("Protocol mismatch {}".format(".".join(path)))
    if float(_at(baseline, ("MODEL", "MULTI_GRANULARITY_GATING_TAU"))) != 1.0:
        raise ProtocolError("Baseline gate temperature is not 1.0")
    if float(_at(candidate, ("MODEL", "MULTI_GRANULARITY_GATING_TAU"))) != 0.5:
        raise ProtocolError("Candidate gate temperature is not 0.5")
    differences = _leaf_differences(baseline, candidate)
    unexpected = [item for item in differences if item[0] not in ALLOWED_LEAF_DIFFERENCES]
    if unexpected:
        raise ProtocolError("Unexpected config differences: {}".format(unexpected))
    return {
        "baseline_config": str(Path(baseline_path).resolve()),
        "candidate_config": str(Path(candidate_path).resolve()),
        "algorithm_variable": "MODEL.MULTI_GRANULARITY_GATING_TAU: 1.0 -> 0.5",
        "allowed_differences": [".".join(item[0]) for item in differences],
        "status": "verified",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.baseline_config, args.candidate_config),
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
