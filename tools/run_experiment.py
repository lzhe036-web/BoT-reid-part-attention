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
    MISSING_EVIDENCE,
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one formal Re-ID experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--experiment-family", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--run-id")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--records-root",
        default=str(REPO_ROOT / "experiment_records"),
    )
    return parser.parse_args(argv)


def _load_config(config_path):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(config_path))
    local_cfg.freeze()
    return local_cfg


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


def _model_manifest(configuration):
    identity = experiment_identity(configuration)
    return {
        "schema_version": 1,
        "backbone": configuration.get("MODEL", {}).get("NAME", NOT_RECORDED),
        "neck": configuration.get("MODEL", {}).get("NECK", NOT_RECORDED),
        "method": identity["method"],
        "modules": identity["modules"],
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
    local_cfg = _load_config(config_path)
    output_dir = _require_new_output_dir(local_cfg.OUTPUT_DIR)
    configuration = _plain_config(local_cfg)
    resolved_seed = configuration.get("SEED", NOT_RECORDED)
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
    )
    try:
        environment = collect_environment(run_dir, REPO_ROOT)
        atomic_write_json(run_dir / "environment.json", environment)
        dataset = init_dataset(
            local_cfg.DATASETS.NAMES, root=local_cfg.DATASETS.ROOT_DIR
        )
        dataset_manifest = build_dataset_manifest(
            dataset, configuration, local_cfg.DATASETS.ROOT_DIR
        )
        atomic_write_json(run_dir / "dataset_manifest.json", dataset_manifest)
        atomic_write_json(run_dir / "model_manifest.json", _model_manifest(configuration))
        atomic_write_json(run_dir / "reproducibility.json", {
            "schema_version": 1,
            "source_seed": configuration.get("SEED", NOT_RECORDED),
            "resolved_seed": resolved_seed,
            "applied_seed": NOT_RECORDED,
            "seed": NOT_RECORDED,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", NOT_RECORDED),
            "python_random_seed": NOT_RECORDED,
            "numpy_seed": NOT_RECORDED,
            "torch_cpu_seed": NOT_RECORDED,
            "torch_cuda_seed": NOT_RECORDED,
            "dataloader_worker_seed_base": NOT_RECORDED,
            "dataloader_worker_seed_strategy": NOT_RECORDED,
            "sampler_seed": NOT_RECORDED,
            "sampler_seed_strategy": NOT_RECORDED,
            "cudnn_deterministic": NOT_RECORDED,
            "cudnn_benchmark": NOT_RECORDED,
            "status": MISSING_EVIDENCE,
            "reason": (
                "The legacy training code does not emit applied RNG evidence; "
                "the bypass recorder does not alter or infer training seeds."
            ),
        })
        train_command = [
            sys.executable,
            "tools/train.py",
            "--config_file",
            str(config_path),
        ]
        started = time.monotonic()
        completed = subprocess.run(
            train_command,
            cwd=str(REPO_ROOT),
            check=False,
        )
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
