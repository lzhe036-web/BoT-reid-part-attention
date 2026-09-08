#!/usr/bin/env python
"""Recover/register a completed G2-E formal run without retraining."""

from __future__ import absolute_import

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import recover_g2_global_local_experiment as recovery


EXPERIMENT_ID = "C2-L03-MGDG-G2-E-SD-A03-T0P5-S42"
EXPECTED_BRANCH = "codex/g2-e-static-dynamic-alpha0p3-tau0p5"
EXPECTED_PARENT_BRANCH = "codex/g2-global-local-gating-tau0p5"
EXPECTED_PARENT_COMMIT = "d724a6536e4a819c5d2932412e90b7dea224041b"
RESULT_FILENAME = "g2_e_static_dynamic_alpha0p3_tau0p5_formal_result.json"
DEFAULT_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_e_static_dynamic_alpha0p3_tau0p5_autodl.yml"
)


def _configure_profile():
    recovery.EXPERIMENT_ID = EXPERIMENT_ID
    recovery.EXPECTED_BRANCH = EXPECTED_BRANCH
    recovery.EXPECTED_PARENT_BRANCH = EXPECTED_PARENT_BRANCH
    recovery.EXPECTED_PARENT_COMMIT = EXPECTED_PARENT_COMMIT
    recovery.EXPECTED_GATING_TAU = 0.5
    recovery.RESULT_FILENAME = RESULT_FILENAME
    recovery.FORMAL_RESULT_ARTIFACT_TYPE = "g2_e_static_dynamic_alpha0p3_tau0p5_formal_result"
    recovery.METHOD_VARIANT = "g2_e_static_concat_plus_gated_residual_alpha0p3_tau0p5"
    recovery.METHOD_LABEL = (
        "C2-L03 + G2-E static concat + gated residual "
        "[g,(1+0.3w2)z2,(1+0.3w4)z4,(1+0.3w6)z6]"
    )
    recovery.BASELINE_LABEL = "C2-L03 + G2-A Dynamic Gating (tau_g=0.5)"
    recovery.REQUIRE_STATIC_DYNAMIC_RESIDUAL = True
    recovery.EXPECTED_STATIC_DYNAMIC_ALPHA = 0.3
    recovery.DEFAULT_CONFIG = DEFAULT_CONFIG


def recover(*args, **kwargs):
    _configure_profile()
    return recovery.recover(*args, **kwargs)


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
                      "selected_epoch": row["selected_epoch"],
                      "selected_checkpoint_sha256": row["selected_checkpoint_sha256"]},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
