#!/usr/bin/env python
# encoding: utf-8
"""Regenerate every Markdown experiment table from its CSV source."""

from __future__ import absolute_import

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_recording import (
    TABLE_SCHEMAS,
    csv_to_markdown,
    ensure_record_layout,
    update_experiments_markdown,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate Markdown result tables")
    parser.add_argument(
        "--records-root", default=str(REPO_ROOT / "experiment_records")
    )
    parser.add_argument(
        "--experiments", default=str(REPO_ROOT / "EXPERIMENTS.md")
    )
    args = parser.parse_args(argv)
    root = ensure_record_layout(args.records_root)
    for name in TABLE_SCHEMAS:
        csv_to_markdown(
            root / "tables" / "{}.csv".format(name),
            root / "tables" / "{}.md".format(name),
        )
    update_experiments_markdown(args.experiments, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
