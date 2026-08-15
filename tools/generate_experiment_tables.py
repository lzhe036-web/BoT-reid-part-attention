#!/usr/bin/env python
"""Regenerate the authoritative sections in the existing EXPERIMENTS.md."""

from __future__ import absolute_import

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_recording import (
    migrate_unified_schema,
    refresh_experiments_markdown,
)
from utils.experiment_schema import EVIDENCE_FIELDS, FORMAL_FIELDS, RUN_FIELDS


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-root", default=str(REPO_ROOT / "experiment_records"))
    parser.add_argument("--experiments", default=str(REPO_ROOT / "EXPERIMENTS.md"))
    args = parser.parse_args(argv)
    records_root = Path(args.records_root)
    migrate_unified_schema(records_root / "runs.csv", RUN_FIELDS)
    migrate_unified_schema(
        records_root / "tables" / "main_results.csv", FORMAL_FIELDS
    )
    migrate_unified_schema(
        records_root / "evidence_manifest.tsv", EVIDENCE_FIELDS,
        delimiter="\t",
    )
    refresh_experiments_markdown(args.experiments, args.records_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
