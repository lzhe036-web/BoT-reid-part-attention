#!/usr/bin/env python
"""Fail closed unless G2-E is exactly the declared G2-A fusion ablation."""

from __future__ import absolute_import

import argparse
import json
from pathlib import Path

import yaml


class ProtocolError(ValueError):
    pass


EXPECTED = {
    ("SEED",): 42,
    ("DATASETS", "NAMES"): "market1501",
    ("MODEL", "MULTI_GRANULARITY_DYNAMIC_GATING"): True,
    ("MODEL", "MULTI_GRANULARITY_GATING_INPUT"): "concat_global_local",
    ("MODEL", "MULTI_GRANULARITY_GATING_TAU"): 0.5,
    ("MODEL", "MULTI_GRANULARITY_GATING_NORMALIZATION"): "scaled_softmax",
    ("MODEL", "MULTI_GRANULARITY_PART_SCALES"): [2, 4, 6],
    ("SOLVER", "MAX_EPOCHS"): 120,
    ("TEST", "RE_RANKING"): "no",
}
ALLOWED_DIFFERENCES = {
    ("MODEL", "MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL"),
    ("MODEL", "MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA"),
    ("OUTPUT_DIR",),
}


def _read(path):
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ProtocolError("Cannot read YAML {}: {}".format(path, error))
    if not isinstance(value, dict):
        raise ProtocolError("YAML must be a mapping: {}".format(path))
    return value


def _at(mapping, path):
    value = mapping
    for item in path:
        if not isinstance(value, dict) or item not in value:
            raise ProtocolError("Missing required setting {}".format(".".join(path)))
        value = value[item]
    return value


def _differences(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append((prefix + (key,), left.get(key), right.get(key)))
            else:
                result.extend(_differences(left[key], right[key], prefix + (key,)))
        return result
    return [] if left == right else [(prefix, left, right)]


def verify(baseline_path, candidate_path):
    baseline, candidate = _read(baseline_path), _read(candidate_path)
    for path, expected in EXPECTED.items():
        if _at(baseline, path) != expected or _at(candidate, path) != expected:
            raise ProtocolError("Protocol mismatch {}".format(".".join(path)))
    if _at(candidate, ("MODEL", "MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL")) is not True:
        raise ProtocolError("G2-E residual switch must be enabled")
    if float(_at(candidate, ("MODEL", "MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA"))) != 0.3:
        raise ProtocolError("G2-E residual alpha must be 0.3")
    differences = _differences(baseline, candidate)
    unexpected = [item for item in differences if item[0] not in ALLOWED_DIFFERENCES]
    if unexpected:
        raise ProtocolError("Unexpected config differences: {}".format(unexpected))
    return {
        "status": "verified",
        "baseline_config": str(Path(baseline_path).resolve()),
        "candidate_config": str(Path(candidate_path).resolve()),
        "algorithm_variable": "static concatenation + gated residual, alpha=0.3; tau_g=0.5",
        "allowed_differences": [".".join(item[0]) for item in differences],
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
