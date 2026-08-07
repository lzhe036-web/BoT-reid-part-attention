#!/usr/bin/env python
# encoding: utf-8
"""Single formal entry point: preflight, train, analyze, finalize, archive."""

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
)
from utils.reproducibility import (
    RUNNER_SEED_ENV,
    validate_seed,
    validate_seed_evidence_chain,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one formal Re-ID experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--experiment-family", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--run-id")
    parser.add_argument("--notes", default="")
    parser.add_argument("--method")
    parser.add_argument("--baseline-method", default=NOT_RECORDED)
    parser.add_argument("--baseline-commit", default=NOT_RECORDED)
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


def _load_explicit_source_seed(config_path):
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle) or {}
    if "SEED" not in source:
        raise RuntimeError("Formal config must explicitly declare SEED")
    return validate_seed(source["SEED"])


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
                "Formal OUTPUT_DIR is not empty; refusing to overwrite evidence: {}"
                .format(output)
            )
    return output


def _model_manifest(configuration, method=None,
                    baseline_method=NOT_RECORDED,
                    baseline_commit=NOT_RECORDED):
    identity = experiment_identity(configuration)
    model = configuration.get("MODEL", {})
    scales = model.get("MULTI_GRANULARITY_SCALES", [])
    projection_dim = model.get("MULTI_GRANULARITY_DIM", NOT_RECORDED)
    multi_granularity_enabled = identity["modules"]["multi_granularity"]
    descriptor_dim = NOT_RECORDED
    if (multi_granularity_enabled
            and model.get("NAME") == "resnet50"
            and projection_dim != NOT_RECORDED):
        descriptor_dim = 2048 + len(scales) * int(projection_dim)
    return {
        "schema_version": 1,
        "backbone": configuration.get("MODEL", {}).get("NAME", NOT_RECORDED),
        "neck": configuration.get("MODEL", {}).get("NECK", NOT_RECORDED),
        "method": method or identity["method"],
        "baseline": identity["baseline"],
        "baseline_method": baseline_method,
        "baseline_commit": baseline_commit,
        "modules": identity["modules"],
        "part_correspondence_consistency": {
            "enabled": identity["pcc_enabled"],
            "parts": identity["pcc_parts"],
            "mode": identity["pcc_mode"],
            "alignment_strategy": identity["alignment_strategy"],
            "lambda": identity["pcc_lambda"],
        },
        "cross_camera_positive_lambda": identity[
            "cross_camera_positive_lambda"
        ],
        "baseline_experiment": (
            "C2-L03" if multi_granularity_enabled else NOT_RECORDED
        ),
        "baseline_existing_attention": bool(model.get("PART_ATTENTION", False)),
        "new_module_attention": (
            False if multi_granularity_enabled else NOT_RECORDED
        ),
        "multi_granularity_scales": scales,
        "multi_granularity_projection_dim": projection_dim,
        "multi_granularity_aggregation": model.get(
            "MULTI_GRANULARITY_AGGREGATION", NOT_RECORDED
        ),
        "descriptor_dim": descriptor_dim,
        "total_params": NOT_RECORDED,
        "trainable_params": NOT_RECORDED,
        "FLOPs": NOT_RECORDED,
        "source": "resolved_config; efficiency values populated post-training",
    }


def run(args):
    config_path = Path(args.config).resolve()
    git_info = validate_git_preflight(
        REPO_ROOT, args.expected_branch, expected_commit=args.expected_commit
    )
    local_cfg = _load_config(config_path, args.opts)
    output_dir = _require_new_output_dir(local_cfg.OUTPUT_DIR)
    configuration = _plain_config(local_cfg)
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
        print("Formal experiment failed closed: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
