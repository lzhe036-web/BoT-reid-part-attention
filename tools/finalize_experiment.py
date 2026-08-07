#!/usr/bin/env python
# encoding: utf-8
"""Finalize an already trained run using the same strict evidence pipeline."""

from __future__ import absolute_import

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_recording import finalize_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Finalize formal experiment evidence")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--records-root", default=str(REPO_ROOT / "experiment_records")
    )
    args = parser.parse_args(argv)
    records_root = Path(args.records_root)
    try:
        result = finalize_run(
            run_dir=records_root / "runs" / args.run_id,
            records_root=records_root,
            repo_root=REPO_ROOT,
            experiments_path=REPO_ROOT / "EXPERIMENTS.md",
            run_analyses=True,
            verify_git=True,
        )
    except BaseException as error:
        print("Finalization failed closed: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
