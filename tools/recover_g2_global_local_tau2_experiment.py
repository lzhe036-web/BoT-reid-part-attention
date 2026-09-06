#!/usr/bin/env python
"""Recover/register the formal G2 gate-temperature τg=2.0 ablation."""

from __future__ import absolute_import

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import recover_g2_global_local_experiment as g2_recovery


EXPERIMENT_ID = "C2-L03-MGDG-G2-GL-T2-S42"
EXPECTED_BRANCH = "codex/g2-global-local-gating-tau2"
EXPECTED_PARENT_BRANCH = "codex/g2-global-local-gating-tau0p5"
EXPECTED_PARENT_COMMIT = "d724a6536e4a819c5d2932412e90b7dea224041b"
EXPECTED_GATING_TAU = 2.0
RESULT_FILENAME = "g2_tau2_formal_result.json"
DEFAULT_CONFIG = (
    REPO_ROOT / "configs" /
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_global_local_tau2_autodl.yml"
)


def _configure_candidate_profile():
    g2_recovery.EXPERIMENT_ID = EXPERIMENT_ID
    g2_recovery.EXPECTED_BRANCH = EXPECTED_BRANCH
    g2_recovery.EXPECTED_PARENT_BRANCH = EXPECTED_PARENT_BRANCH
    g2_recovery.EXPECTED_PARENT_COMMIT = EXPECTED_PARENT_COMMIT
    g2_recovery.EXPECTED_GATING_TAU = EXPECTED_GATING_TAU
    g2_recovery.RESULT_FILENAME = RESULT_FILENAME
    g2_recovery.METHOD_VARIANT = "g2_global_local_tau2_per_sample_dynamic_gating"
    g2_recovery.METHOD_LABEL = (
        "C2-L03 + G2 Dynamic Gating [g,z2,z4,z6] -> [w2,w4,w6] (τg=2.0)"
    )
    g2_recovery.BASELINE_LABEL = "C2-L03 + G2 Dynamic Gating (τg=0.5)"
    g2_recovery.REQUIRE_RESULT_GATING_TEMPERATURE = True
    g2_recovery.DEFAULT_CONFIG = DEFAULT_CONFIG


def recover(*args, **kwargs):
    _configure_candidate_profile()
    return g2_recovery.recover(*args, **kwargs)


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
    print(json.dumps({
        "created": created, "run_dir": str(run_dir), "run_id": row["run_id"],
        "status": row["status"], "rank1_percent": row["rank1_percent"],
        "map_percent": row["map_percent"], "selected_epoch": row["selected_epoch"],
        "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
