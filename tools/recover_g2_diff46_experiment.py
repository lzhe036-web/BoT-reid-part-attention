#!/usr/bin/env python
"""Recover/register only a completed formal G2-D1 run without retraining."""

from __future__ import absolute_import

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import recover_g2_global_local_experiment as shared


EXPERIMENT_ID = "C2-L03-MGDG-G2-D1-T0P5-S42"
EXPECTED_BRANCH = "codex/g2-d1"
EXPECTED_PARENT_BRANCH = "codex/g2-global-local-gating-tau0p5"
EXPECTED_PARENT_COMMIT = "d724a6536e4a819c5d2932412e90b7dea224041b"
EXPECTED_GATING_INPUT = "concat_global_local_diff46"
EXPECTED_GATING_INPUT_DESCRIPTION = "concat([g, z2, z4, z6, abs(z4-z6)])"
EXPECTED_CONTROLLER_INPUT_DIM = 3072
EXPECTED_DELTA46_DEFINITION = (
    "torch.abs(z4-z6); unweighted; graph-connected; controller input only"
)
RESULT_FILENAME = "g2_d1_formal_result.json"
DEFAULT_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_diff46_autodl.yml"
)


def _profile():
    shared.EXPERIMENT_ID = EXPERIMENT_ID
    shared.EXPECTED_BRANCH = EXPECTED_BRANCH
    shared.EXPECTED_PARENT_BRANCH = EXPECTED_PARENT_BRANCH
    shared.EXPECTED_PARENT_COMMIT = EXPECTED_PARENT_COMMIT
    shared.EXPECTED_GATING_INPUT = EXPECTED_GATING_INPUT
    shared.EXPECTED_GATING_INPUT_DESCRIPTION = EXPECTED_GATING_INPUT_DESCRIPTION
    shared.EXPECTED_CONTROLLER_INPUT_DIM = EXPECTED_CONTROLLER_INPUT_DIM
    shared.EXPECTED_DELTA46_DEFINITION = EXPECTED_DELTA46_DEFINITION
    shared.EXPECTED_GATING_TAU = 0.5
    shared.RESULT_FILENAME = RESULT_FILENAME
    shared.METHOD_VARIANT = "g2_d1_abs_z4_z6_per_sample_dynamic_gating"
    shared.METHOD_LABEL = (
        "G2-D1 Dynamic Gating [g,z2,z4,z6,abs(z4-z6)] -> [w2,w4,w6]"
    )
    shared.BASELINE_LABEL = "G2-A Dynamic Gating [g,z2,z4,z6] (tau=0.5)"
    shared.REQUIRE_RESULT_GATING_TEMPERATURE = True
    shared.REQUIRE_RESULT_INPUT_METADATA = True
    shared.DEFAULT_CONFIG = DEFAULT_CONFIG


def recover(*args, **kwargs):
    _profile()
    return shared.recover(*args, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--console-log", required=True)
    parser.add_argument("--records-root", default=str(REPO_ROOT / "experiment_records"))
    parser.add_argument("--experiments-path", default=str(REPO_ROOT / "EXPERIMENTS.md"))
    parser.add_argument("--started-at-utc", default=None)
    parser.add_argument("--ended-at-utc", default=None)
    parser.add_argument("--runtime-seconds", type=float, default=None)
    args = parser.parse_args(argv)
    run_dir, row, created = recover(
        args.config_file, args.output_dir, args.console_log, args.records_root,
        args.experiments_path, started_at_utc=args.started_at_utc,
        ended_at_utc=args.ended_at_utc, runtime_seconds=args.runtime_seconds,
    )
    print(json.dumps({"created": created, "run_dir": str(run_dir),
                      "run_id": row["run_id"], "status": row["status"],
                      "rank1_percent": row["rank1_percent"],
                      "map_percent": row["map_percent"],
                      "selected_epoch": row["selected_epoch"]},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
