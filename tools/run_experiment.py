#!/usr/bin/env python
# encoding: utf-8
"""Unified formal/smoke entry point: verify, train, finalize, archive."""

from __future__ import absolute_import

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from data.datasets import init_dataset
from utils.experiment_recording import (
    NOT_RECORDED,
    SCHEMA_VERSION,
    atomic_write_json,
    build_dataset_manifest,
    collect_environment,
    experiment_identity,
    finalize_run,
    generate_run_id,
    initialize_run,
    record_run_failure,
    record_training_exit,
    validate_git_preflight,
    validate_parent_lineage,
)
from utils.multigranular_signature import (
    canonical_multigranular_feature_signature,
)
from utils.reproducibility import (
    RUNNER_SEED_ENV,
    validate_seed,
    validate_seed_evidence_chain,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one recorded Re-ID experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--experiment-family", required=True)
    parser.add_argument(
        "--run-kind", choices=("formal", "smoke"), default="formal"
    )
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--run-id")
    parser.add_argument("--notes", default="")
    parser.add_argument("--method")
    parser.add_argument("--baseline-method", default=NOT_RECORDED)
    parser.add_argument("--baseline-commit", default=NOT_RECORDED)
    parser.add_argument("--parent-branch", default=NOT_RECORDED)
    parser.add_argument("--parent-commit", default=NOT_RECORDED)
    parser.add_argument(
        "--records-root",
        default=str(REPO_ROOT / "experiment_records"),
    )
    parser.add_argument(
        "opts",
        help="YACS config overrides, used by isolated smoke runs",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(argv)


def _load_config(config_path, opts=None):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(config_path))
    local_cfg.merge_from_list(opts or [])
    local_cfg.freeze()
    return local_cfg


def _validate_run_overrides(run_kind, opts):
    overrides = list(opts or [])
    if run_kind == "formal" and overrides:
        raise RuntimeError("Formal runs forbid command-line config overrides")
    if len(overrides) % 2:
        raise RuntimeError("Config overrides must contain key/value pairs")
    allowed_smoke = {
        "SOLVER.MAX_EPOCHS",
        "SOLVER.CHECKPOINT_PERIOD",
        "SOLVER.EVAL_PERIOD",
        "OUTPUT_DIR",
    }
    keys = set(overrides[::2])
    unexpected = sorted(keys - allowed_smoke)
    if run_kind == "smoke" and unexpected:
        raise RuntimeError(
            "Smoke overrides contain non-isolated fields: {}".format(
                unexpected
            )
        )


def _load_explicit_source_seed(config_path):
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle) or {}
    if "SEED" not in source:
        raise RuntimeError("Formal config must explicitly declare SEED")
    return validate_seed(source["SEED"])


def _load_source_output_dir(config_path):
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle) or {}
    return source.get("OUTPUT_DIR", NOT_RECORDED)


def _validate_resolved_run_config(configuration, run_kind,
                                  source_output_dir):
    """Validate effective smoke isolation after every override is applied."""
    solver = configuration.get("SOLVER", {})
    output_dir = configuration.get("OUTPUT_DIR", NOT_RECORDED)
    if output_dir in (None, "", NOT_RECORDED):
        raise RuntimeError("Resolved OUTPUT_DIR is missing")
    if run_kind == "smoke":
        expected = {
            "MAX_EPOCHS": 1,
            "CHECKPOINT_PERIOD": 1,
            "EVAL_PERIOD": 1,
        }
        conflicts = {
            key: solver.get(key, NOT_RECORDED)
            for key, value in expected.items()
            if solver.get(key, NOT_RECORDED) != value
        }
        if conflicts:
            raise RuntimeError(
                "Resolved smoke config is not isolated: {}".format(conflicts)
            )
        if str(output_dir) == str(source_output_dir):
            raise RuntimeError(
                "Smoke OUTPUT_DIR must differ from the formal OUTPUT_DIR"
            )
    return configuration


def _build_training_environment(resolved_seed, base_environment=None):
    seed = validate_seed(resolved_seed)
    training_env = dict(
        os.environ if base_environment is None else base_environment
    )
    training_env["PYTHONHASHSEED"] = str(seed)
    training_env[RUNNER_SEED_ENV] = str(seed)
    return training_env


def _launch_training_subprocess(train_command, training_env):
    return subprocess.run(
        train_command,
        cwd=str(REPO_ROOT),
        check=False,
        env=training_env,
    )


def _plain_config(local_cfg):
    import yaml
    from utils.reproducibility import resolved_config_text

    return yaml.safe_load(resolved_config_text(local_cfg))


def _require_new_output_dir(path):
    output = Path(path)
    if output.exists():
        if not output.is_dir():
            raise RuntimeError("OUTPUT_DIR exists and is not a directory: {}".format(output))
        if next(output.iterdir(), None) is not None:
            raise RuntimeError(
                "OUTPUT_DIR is not empty; refusing to overwrite evidence: {}"
                .format(output)
            )
    return output


def _model_manifest(configuration, method=None,
                    baseline_method=NOT_RECORDED,
                    baseline_commit=NOT_RECORDED,
                    parent_branch=NOT_RECORDED,
                    parent_commit=NOT_RECORDED):
    identity = experiment_identity(configuration)
    signature, signature_sha256 = (
        canonical_multigranular_feature_signature(configuration)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "backbone": configuration.get("MODEL", {}).get("NAME", NOT_RECORDED),
        "neck": configuration.get("MODEL", {}).get("NECK", NOT_RECORDED),
        "method": method or identity["method"],
        "method_family": identity["method_family"],
        "method_variant": identity["method_variant"],
        "baseline": identity["baseline"],
        "baseline_method": baseline_method,
        "baseline_commit": baseline_commit,
        "parent_branch": parent_branch,
        "parent_commit": parent_commit,
        "multigranular_feature_signature": signature,
        "multigranular_feature_signature_sha256": signature_sha256,
        "modules": identity["modules"],
        "part_correspondence_consistency": {
            "enabled": identity["pcc_enabled"],
            "parts": identity["pcc_parts"],
            "mode": identity["pcc_mode"],
            "alignment_strategy": identity["alignment_strategy"],
            "alignment_mode": identity["alignment_mode"],
            "alignment_temperature": identity["alignment_temperature"],
            "gating_mode": identity["gating_mode"],
            "gating_temperature": identity["gating_temperature"],
            "lambda": identity["pcc_lambda"],
        },
        "cross_camera_positive_lambda": identity[
            "cross_camera_positive_lambda"
        ],
        "total_params": NOT_RECORDED,
        "trainable_params": NOT_RECORDED,
        "FLOPs": NOT_RECORDED,
        "source": "resolved_config; efficiency values populated post-training",
    }


def run(args):
    config_path = Path(args.config).resolve()
    _validate_run_overrides(args.run_kind, args.opts)
    git_info = validate_git_preflight(
        REPO_ROOT, args.expected_branch, expected_commit=args.expected_commit
    )
    local_cfg = _load_config(config_path, args.opts)
    output_dir = _require_new_output_dir(local_cfg.OUTPUT_DIR)
    configuration = _plain_config(local_cfg)
    _validate_resolved_run_config(
        configuration, args.run_kind, _load_source_output_dir(config_path)
    )
    if ((args.parent_branch == NOT_RECORDED)
            != (args.parent_commit == NOT_RECORDED)):
        raise RuntimeError(
            "Parent branch and parent commit must be recorded together"
        )
    if args.parent_branch != NOT_RECORDED:
        validate_parent_lineage(
            REPO_ROOT, args.parent_branch, args.parent_commit,
            child_commit=git_info["commit"],
        )
    source_seed = _load_explicit_source_seed(config_path)
    resolved_seed = validate_seed(configuration.get("SEED", NOT_RECORDED))
    validate_seed_evidence_chain(source_seed, resolved_seed, resolved_seed)
    training_env = _build_training_environment(resolved_seed)
    run_id = args.run_id or generate_run_id(
        args.experiment_id, git_info["commit"], resolved_seed
    )
    run_dir, manifest = initialize_run(
        records_root=args.records_root,
        experiment_id=args.experiment_id,
        experiment_family=args.experiment_family,
        run_id=run_id,
        config_file=str(config_path),
        resolved_cfg=local_cfg,
        output_dir=str(output_dir),
        git_info=git_info,
        notes=args.notes,
        command=sys.argv,
        expected_branch=args.expected_branch,
        method=args.method,
        baseline_method=args.baseline_method,
        baseline_commit=args.baseline_commit,
        run_kind=args.run_kind,
        parent_branch=args.parent_branch,
        parent_commit=args.parent_commit,
    )
    try:
        environment = collect_environment(
            run_dir,
            REPO_ROOT,
            training_pythonhashseed=training_env["PYTHONHASHSEED"],
            expected_branch=git_info["branch"],
            expected_commit=git_info["commit"],
        )
        atomic_write_json(run_dir / "environment.json", environment)
        dataset = init_dataset(
            local_cfg.DATASETS.NAMES, root=local_cfg.DATASETS.ROOT_DIR
        )
        dataset_manifest = build_dataset_manifest(
            dataset, configuration, local_cfg.DATASETS.ROOT_DIR
        )
        atomic_write_json(run_dir / "dataset_manifest.json", dataset_manifest)
        atomic_write_json(
            run_dir / "model_manifest.json",
            _model_manifest(
                configuration,
                method=args.method,
                baseline_method=args.baseline_method,
                baseline_commit=args.baseline_commit,
                parent_branch=args.parent_branch,
                parent_commit=args.parent_commit,
            ),
        )
        train_command = [
            sys.executable,
            "tools/train.py",
            "--config_file",
            str(config_path),
        ] + list(args.opts or [])
        started = time.monotonic()
        completed = _launch_training_subprocess(train_command, training_env)
        runtime = time.monotonic() - started
        record_training_exit(run_dir, completed.returncode, runtime)
        if completed.returncode != 0:
            raise RuntimeError(
                "Training exited with code {}".format(completed.returncode)
            )
        result = finalize_run(
            run_dir=run_dir,
            records_root=args.records_root,
            repo_root=REPO_ROOT,
            experiments_path=REPO_ROOT / "EXPERIMENTS.md",
            run_analyses=True,
            verify_git=True,
        )
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        return 0
    except BaseException as error:
        current_status = "failed"
        try:
            status_path = run_dir / "run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("training_exit_code") == 0:
                current_status = "incomplete"
        except Exception:
            pass
        record_run_failure(run_dir, error, status=current_status)
        raise


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except BaseException as error:
        print("Experiment failed closed: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
