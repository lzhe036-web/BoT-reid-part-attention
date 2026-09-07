#!/usr/bin/env python
"""Seal a completed G2-D1 run, then create its single-version evidence package."""

from __future__ import absolute_import

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.finalize_g2_global_local_experiment import finalize as shared_finalize
from tools.package_g2_diff46_result import package


EXPECTED_BRANCH = "codex/g2-d1"
EXPECTED_INPUT = "concat_global_local_diff46"
RESULT_FILENAME = "g2_d1_formal_result.json"
ANALYSIS_DIRNAME = "g2_d1_gating_analysis"


def finalize(config_path, output_dir):
    result_path, result = shared_finalize(
        config_path, output_dir, expected_branch=EXPECTED_BRANCH,
        expected_gating_tau=0.5, expected_gating_input=EXPECTED_INPUT,
        experiment_label="G2-D1 Dynamic Gating (abs(z4-z6), tau=0.5)",
        result_filename=RESULT_FILENAME, analysis_dirname=ANALYSIS_DIRNAME,
    )
    package_dir = package(output_dir)
    return result_path, result, package_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result_path, result, package_dir = finalize(args.config_file, args.output_dir)
    print(json.dumps({"result_path": str(result_path), "package_dir": str(package_dir),
                      "metrics": result["metrics"]}, ensure_ascii=False,
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
