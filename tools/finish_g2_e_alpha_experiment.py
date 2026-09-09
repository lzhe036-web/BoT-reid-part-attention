#!/usr/bin/env python
"""Select/evaluate, register, then plot/package. Retry without retraining."""
import argparse
import importlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils.g2_e_alpha_delivery import record_delivery, archive_package


def module_for(action, alpha):
    token = {0.1: "0p1", 0.5: "0p5"}[float(alpha)]
    suffix = "result" if action == "package" else "experiment"
    return importlib.import_module("tools.{}_g2_e_static_dynamic_alpha{}_tau0p5_{}".format(action, token, suffix))


def finish(alpha, config_file, output_dir, console_log, records_root=None,
           experiments_path=None, archive_dir=None, register_only=False):
    records_root = Path(records_root or REPO_ROOT/"experiment_records").resolve()
    experiments_path = Path(experiments_path or REPO_ROOT/"EXPERIMENTS.md").resolve()
    output_dir = Path(output_dir).resolve()
    archive_dir = Path(archive_dir or "/root/autodl-tmp/exports")
    stage, result, run_dir, package_dir = "finalize_raw_evidence", None, None, None
    try:
        _, result = module_for("finalize", alpha).finalize(config_file, output_dir)
        stage = "register"
        run_dir, row, _ = module_for("recover", alpha).recover(
            config_file, output_dir, console_log, records_root, experiments_path)
        if register_only:
            return {"run_id": row["run_id"], "stage": "registered"}
        stage = "plot_and_package"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
        package_dir = output_dir/("g2_e_alpha{}_tau0p5_{}".format(str(alpha).replace(".", "p"), stamp))
        record_delivery(result, records_root, experiments_path, output_dir, stage, package_dir=package_dir)
        module_for("package", alpha).package(output_dir, package_dir, records_root, experiments_path)
        stage = "archive"
        archive, digest = archive_package(package_dir, archive_dir)
        payload = record_delivery(result, records_root, experiments_path, output_dir, "complete",
                                  package_dir=package_dir, archive=archive, archive_sha256=digest)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    except Exception as error:
        if run_dir is not None:
            try:
                record_delivery(result, records_root, experiments_path, output_dir, stage+"_failed",
                                error=error, package_dir=package_dir)
            except Exception as record_error:
                print("Delivery status update failed: {}".format(record_error), file=sys.stderr)
        print("Stage {} failed: {}. Formal registry is not rolled back.".format(stage, error), file=sys.stderr)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", required=True, type=float, choices=(.1, .5))
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--console-log", required=True)
    parser.add_argument("--records-root")
    parser.add_argument("--experiments-path")
    parser.add_argument("--archive-dir")
    parser.add_argument("--register-only", action="store_true")
    args = vars(parser.parse_args(argv))
    finish(**args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
