#!/usr/bin/env python
"""Recover and register completed G2-local-only evidence without retraining."""

from __future__ import absolute_import

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.g2_dynamic_gating_profiles import G2_LOCAL_ONLY_PROFILE
from tools.recover_g2_global_local_experiment import G2RecoveryError
from tools.recover_g2_global_local_experiment import main_for_profile
from tools.recover_g2_global_local_experiment import recover as _recover


EXPERIMENT_ID = G2_LOCAL_ONLY_PROFILE.experiment_id
EXPECTED_BRANCH = G2_LOCAL_ONLY_PROFILE.expected_branch
EXPECTED_GATING_INPUT = G2_LOCAL_ONLY_PROFILE.gating_input


def recover(config_path, output_dir, console_log, records_root, experiments_path,
            started_at_utc=None, ended_at_utc=None, runtime_seconds=None):
    return _recover(
        config_path, output_dir, console_log, records_root, experiments_path,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        runtime_seconds=runtime_seconds,
        profile=G2_LOCAL_ONLY_PROFILE,
    )


def main(argv=None):
    return main_for_profile(G2_LOCAL_ONLY_PROFILE, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
