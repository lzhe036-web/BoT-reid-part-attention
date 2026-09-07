#!/usr/bin/env python
"""Fail closed unless G2-C differs from original G2 only as declared.

The comparison is intentionally configuration-only.  Source-code changes are
captured separately by Git commit and diff evidence; this tool proves that the
formal training YAML changes only controller capacity, gate temperature, and
the required independent output location.
"""

from __future__ import absolute_import

import argparse
import json
from pathlib import Path

import yaml


ALLOWED_CHANGES = {
    "MODEL.MULTI_GRANULARITY_GATING_CONTROLLER": ("linear", "mlp"),
    "MODEL.MULTI_GRANULARITY_GATING_HIDDEN_DIM": (0, 256),
    "MODEL.MULTI_GRANULARITY_GATING_TAU": (1.0, 0.5),
    "OUTPUT_DIR": (
        "/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_global_local_tau1_seed42_market1501",
        "/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_c_hidden256_tau0p5_seed42_market1501",
    ),
}


def _load(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("{} is not a YAML mapping".format(path))
    data.setdefault("MODEL", {})
    # The old G2 YAML predates explicit controller fields.  These are the
    # canonical defaults and therefore not an unrecorded experiment change.
    data["MODEL"].setdefault("MULTI_GRANULARITY_GATING_CONTROLLER", "linear")
    data["MODEL"].setdefault("MULTI_GRANULARITY_GATING_HIDDEN_DIM", 0)
    return data


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        rows = {}
        for key in sorted(value):
            child = "{}.{}".format(prefix, key) if prefix else str(key)
            rows.update(_flatten(value[key], child))
        return rows
    return {prefix: value}


def verify(base_config, candidate_config):
    base = _flatten(_load(base_config))
    candidate = _flatten(_load(candidate_config))
    differences = {
        key: (base.get(key), candidate.get(key))
        for key in sorted(set(base) | set(candidate))
        if base.get(key) != candidate.get(key)
    }
    unexpected = {
        key: value for key, value in differences.items()
        if key not in ALLOWED_CHANGES or value != ALLOWED_CHANGES[key]
    }
    missing = {
        key: expected for key, expected in ALLOWED_CHANGES.items()
        if differences.get(key) != expected
    }
    if unexpected or missing:
        raise ValueError(
            "G2-C protocol mismatch; unexpected={}, missing_or_wrong={}".format(
                unexpected, missing
            )
        )
    return {
        "status": "verified",
        "base_config": str(Path(base_config).resolve()),
        "candidate_config": str(Path(candidate_config).resolve()),
        "declared_algorithm_changes": {
            "controller": "Linear(2816,3) -> Linear(2816,256) -> ReLU -> Linear(256,3)",
            "gating_temperature": "1.0 -> 0.5",
        },
        "allowed_configuration_differences": differences,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    result = verify(args.base_config, args.candidate_config)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
