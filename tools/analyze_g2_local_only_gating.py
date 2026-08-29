#!/usr/bin/env python
"""Export observation-only evidence for G2-local-only Dynamic Gating.

The controller is restricted to ``concat([z2,z4,z6])``.  The global descriptor
``g`` remains in the final retrieval descriptor but is not a controller input
and therefore never appears in the controller-block analysis.
"""

from __future__ import absolute_import

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_g2_global_local_gating import analyze as _analyze
from tools.analyze_g2_global_local_gating import main_for_profile
from tools.g2_dynamic_gating_profiles import G2_LOCAL_ONLY_PROFILE


def analyze(config_path, checkpoint_path, output_dir, epoch_stats_path,
            sample_limit=256, device=None):
    return _analyze(
        config_path, checkpoint_path, output_dir, epoch_stats_path,
        sample_limit=sample_limit, device=device,
        profile=G2_LOCAL_ONLY_PROFILE,
    )


def main(argv=None):
    return main_for_profile(G2_LOCAL_ONLY_PROFILE, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
