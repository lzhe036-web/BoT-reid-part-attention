#!/usr/bin/env python
"""Unified formal/dry-run entry point for C2-MGP-K246."""

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

from utils.experiment_recording import (
    EXPECTED_BRANCH,
    EXPECTED_EVIDENCE_ID,
    EXPECTED_EXPERIMENT_FAMILY,
    EXPECTED_RUN_ID,
    EXPECTED_TRAINING_SEED,
    atomic_write_json,
    atomic_write_text,
    collect_environment,
    finalize_run,
    finish_run_timing,
    formal_preflight,
    initialize_run,
    require_temporary_fixture,
    sha256_file,
    utc_now,
)


DEFAULT_CONFIG = (
    REPO_ROOT / "configs" /
    "softmax_triplet_c2_l03_multi_granularity_part_autodl.yml"
)
LAUNCH_SCRIPT = (
    REPO_ROOT / "scripts" /
    "train_c2_l03_multi_granularity_part_autodl.sh"
)
RECORD_DIR = (
    REPO_ROOT / "experiment_records" / "c2_l03_multi_granularity_part"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run and record C2-MGP-K246")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--experiment-family", default=EXPECTED_EXPERIMENT_FAMILY
    )
    parser.add_argument("--run-id", default=EXPECTED_RUN_ID)
    parser.add_argument("--evidence-id", default=EXPECTED_EVIDENCE_ID)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixture-dir")
    return parser.parse_args(argv)


def _load_config(path):
    from config import cfg

    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(path))
    local_cfg.freeze()
    return local_cfg


def _model_manifest(local_cfg, num_classes, preflight):
    import torch
    from modeling import build_model

    model_cfg = local_cfg.clone()
    model_cfg.defrost()
    model_cfg.MODEL.PRETRAIN_CHOICE = "none"
    model_cfg.MODEL.PRETRAIN_PATH = ""
    model_cfg.freeze()
    model = build_model(model_cfg, num_classes=num_classes)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    input_height, input_width = [int(value) for value in local_cfg.INPUT.SIZE_TRAIN]
    model.eval()
    with torch.no_grad():
        feature_map = model.base(torch.zeros(1, 3, input_height, input_width))
    manifest = {
        "schema_version": 1,
        "recorded_at_utc": utc_now(),
        "backbone": str(local_cfg.MODEL.NAME),
        "feature_map_shape": list(feature_map.shape),
        "branches": ["Global", "K2", "K4", "K6"],
        "scales": list(local_cfg.MODEL.MULTI_GRANULARITY_PART_SCALES),
        "projection_dim": int(local_cfg.MODEL.MULTI_GRANULARITY_PART_DIM),
        "aggregation": str(local_cfg.MODEL.MULTI_GRANULARITY_PART_AGGREGATION),
        "fusion": str(local_cfg.MODEL.MULTI_GRANULARITY_PART_FUSION),
        "descriptor_dim": int(model.feature_dim),
        "num_classes": int(num_classes),
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "pretrained_weight_path": preflight["pretrained_weight_path"],
        "pretrained_weight_sha256": preflight["pretrained_weight_sha256"],
    }
    del feature_map
    del model
    return manifest


def _append_run_log(output_dir, message):
    path = Path(output_dir) / "log.txt"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    atomic_write_text(path, existing + message.rstrip() + "\n")


def _run_formal(args):
    # The total-run clock begins before config parsing and preflight. It is
    # carried into finalization so successful-run total time also includes
    # evidence validation and the declared sealing boundary.
    started_monotonic = time.monotonic()
    started_at_utc = utc_now()
    config_path = Path(args.config).resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError(
            "Formal experiment IDs may only use {}".format(DEFAULT_CONFIG.resolve())
        )
    local_cfg = _load_config(config_path)
    output_dir = Path(local_cfg.OUTPUT_DIR).resolve()
    preflight = formal_preflight(
        REPO_ROOT,
        config_path,
        LAUNCH_SCRIPT,
        output_dir,
        args.experiment_family,
        args.run_id,
        args.evidence_id,
        resolved_config=local_cfg,
        record_dir=RECORD_DIR,
    )

    os.environ["PYTHONHASHSEED"] = str(EXPECTED_TRAINING_SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["BOT_EXPECTED_TRAINING_SEED"] = str(EXPECTED_TRAINING_SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_cfg.MODEL.DEVICE_ID)

    train_command = [
        sys.executable,
        "tools/train.py",
        "--config_file",
        str(config_path),
    ]
    shell_command = "bash scripts/train_c2_l03_multi_granularity_part_autodl.sh"
    initialize_run(
        output_dir,
        preflight,
        args.experiment_family,
        args.run_id,
        args.evidence_id,
        sys.argv,
        shell_command,
        REPO_ROOT,
        str(local_cfg.DATASETS.NAMES),
        execution_mode="formal",
        started_at_utc=started_at_utc,
    )
    resolved_text = str(local_cfg).rstrip() + "\n"
    resolved_path = output_dir / "config_resolved.yml"
    atomic_write_text(resolved_path, resolved_text)
    resolved_hash = sha256_file(resolved_path)
    environment = None
    environment_runtime = 0.0
    profiling_runtime = 0.0
    training_runtime = 0.0
    exit_code = 1
    try:
        environment_started = time.monotonic()
        environment = collect_environment(output_dir)
        environment_runtime = time.monotonic() - environment_started
        atomic_write_json(output_dir / "environment.json", environment)
        from data.build import collect_dataset_protocol
        dataset_manifest, num_classes = collect_dataset_protocol(local_cfg)
        atomic_write_json(output_dir / "dataset_manifest.json", dataset_manifest)
        atomic_write_json(
            output_dir / "model_manifest.json",
            _model_manifest(local_cfg, num_classes, preflight),
        )
        profile_command = [
            sys.executable,
            "tools/profile_multi_granularity_part.py",
            "--config", str(config_path),
            "--resolved-config", str(resolved_path),
            "--source-config-sha256", preflight["source_config_sha256"],
            "--resolved-config-sha256", resolved_hash,
            "--mode", "formal",
            "--measurement-seed", str(EXPECTED_TRAINING_SEED),
            "--device", "cuda",
            "--batch-size", str(local_cfg.SOLVER.IMS_PER_BATCH),
            "--input-height", str(local_cfg.INPUT.SIZE_TRAIN[0]),
            "--input-width", str(local_cfg.INPUT.SIZE_TRAIN[1]),
            "--dtype", "float32",
            "--warmup", "5",
            "--measurement-repeats", "20",
            "--num-classes", str(num_classes),
            "--output-file", str(output_dir / "efficiency_profile.json"),
        ]
        profiling_started = time.monotonic()
        profile_result = subprocess.run(
            profile_command, cwd=str(REPO_ROOT), check=False
        )
        profiling_runtime = time.monotonic() - profiling_started
        if profile_result.returncode != 0:
            raise RuntimeError("Mandatory formal efficiency profiler failed")
        _append_run_log(
            output_dir,
            "[experiment recorder] environment: host={}, gpu_count={}, seed={}, "
            "PYTHONHASHSEED={}, CUBLAS_WORKSPACE_CONFIG={}".format(
                environment["hostname"], environment["gpu_count"],
                EXPECTED_TRAINING_SEED, environment["pythonhashseed"],
                environment["cublas_workspace_config"],
            ),
        )
        training_started = time.monotonic()
        completed = subprocess.run(
            train_command,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            check=False,
        )
        training_runtime = time.monotonic() - training_started
        exit_code = int(completed.returncode)
    except Exception:
        exit_code = 1
        raise
    finally:
        total_runtime = time.monotonic() - started_monotonic
        finish_run_timing(output_dir, {
            "total_run_runtime_seconds": total_runtime,
            "environment_collection_runtime_seconds": environment_runtime,
            "profiling_runtime_seconds": profiling_runtime,
            "training_runtime_seconds": training_runtime,
            "finalization_runtime_seconds": 0.0,
        }, exit_code)
        environment_summary = ""
        if environment is not None:
            environment_summary = (
                "host={}, gpu_count={}, seed={}, PYTHONHASHSEED={}, "
                "CUBLAS_WORKSPACE_CONFIG={}; "
            ).format(
                environment["hostname"], environment["gpu_count"],
                EXPECTED_TRAINING_SEED, environment["pythonhashseed"],
                environment["cublas_workspace_config"],
            )
        _append_run_log(
            output_dir,
            "[experiment recorder] {}ended_at_utc={}, exit_code={}, "
            "pre_finalization_elapsed_seconds={:.6f}, "
            "environment_collection_runtime_seconds={:.6f}, "
            "profiling_runtime_seconds={:.6f}, training_runtime_seconds={:.6f}, "
            "runtime_source=time.monotonic".format(
                environment_summary, utc_now(), exit_code, total_runtime,
                environment_runtime, profiling_runtime, training_runtime,
            ),
        )
    if exit_code != 0:
        raise RuntimeError("Training exited with code {}".format(exit_code))
    result = finalize_run(
        output_dir, RECORD_DIR,
        total_runtime_started=started_monotonic,
    )
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    return 0


def _run_dry_fixture(args):
    if not args.fixture_dir:
        raise ValueError("--dry-run requires --fixture-dir inside a temporary directory")
    fixture = require_temporary_fixture(args.fixture_dir)
    result = finalize_run(
        fixture,
        fixture / "experiment_records" / "c2_l03_multi_granularity_part",
    )
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.dry_run:
            return _run_dry_fixture(args)
        if args.fixture_dir:
            raise ValueError("--fixture-dir is only valid together with --dry-run")
        return _run_formal(args)
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
