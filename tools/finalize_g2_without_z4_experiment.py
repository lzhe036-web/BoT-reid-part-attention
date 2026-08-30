#!/usr/bin/env python
"""Select and seal a formal G2-without-z4 experiment without retraining."""

from __future__ import absolute_import

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.finalize_g2_global_local_experiment import finalize as _finalize, main_for_profile
from tools.g2_dynamic_gating_profiles import G2_WITHOUT_Z4_PROFILE


def finalize(config_path, output_dir):
    return _finalize(config_path, output_dir, profile=G2_WITHOUT_Z4_PROFILE)


def main(argv=None):
    return main_for_profile(G2_WITHOUT_Z4_PROFILE, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
