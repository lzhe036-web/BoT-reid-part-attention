#!/usr/bin/env python
"""Regenerate the authoritative sections in the existing EXPERIMENTS.md."""

from __future__ import absolute_import

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_recording import refresh_experiments_markdown


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-root", default=str(REPO_ROOT / "experiment_records"))
    parser.add_argument("--experiments", default=str(REPO_ROOT / "EXPERIMENTS.md"))
    args = parser.parse_args(argv)
    refresh_experiments_markdown(args.experiments, args.records_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
