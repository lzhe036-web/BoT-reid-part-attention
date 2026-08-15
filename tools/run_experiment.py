#!/usr/bin/env python
"""Unified formal/smoke runner for recorded Dynamic Gating experiments."""

from __future__ import absolute_import

import argparse
import codecs
import csv
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_serialization import cfg_node_to_plain_mapping, serialize_cfg_node_yaml
from utils.dynamic_gating_evidence import read_gating_epoch_records
from utils.experiment_recording import (
    DynamicExperimentEvidenceError,
    atomic_write_json,
    build_dynamic_checkpoint_manifest,
    collect_environment,
    initialize_dynamic_run,
    model_state_dict_schema,
    read_validation_history,
    seal_dynamic_run_evidence,
    select_dynamic_checkpoint,
    sha256_file,
    transition_dynamic_run,
    validate_dynamic_configuration,
    validate_dynamic_lineage,
    validate_dynamic_runtime_worktree,
)
from utils.multigranularity_signatures import (
    DYNAMIC_CONFIG_PATH,
    STATIC_BASELINE_SHA,
    STATIC_CONFIG_PATH,
    build_feature_compatibility_evidence,
    require_feature_compatibility,
)


DYNAMIC_BRANCH = "exp/c2-l03-multi-granularity-dynamic-gating"
DEFAULT_CONFIG = REPO_ROOT / DYNAMIC_CONFIG_PATH
STATIC_CONFIG = REPO_ROOT / STATIC_CONFIG_PATH
RECORDS_ROOT = REPO_ROOT / "experiment_records"
EXPERIMENTS_PATH = REPO_ROOT / "EXPERIMENTS.md"
FORMAL_EXPERIMENT_ID = "C2-L03-MGDG-T1-S42"
SMOKE_EXPERIMENT_ID = "C2-L03-MGDG-T1-S42-SMOKE"
SMOKE_OUTPUT_DIR = Path(
    "/root/autodl-tmp/experiments/BoT/"
    "c2_l03_multi_granularity_dynamic_gating_tau1_seed42_market1501_smoke"
)


class TrainingInterrupted(KeyboardInterrupt):
    def __init__(self, child_return_code=None):
        super(TrainingInterrupted, self).__init__("training interrupted")
        self.child_return_code = child_return_code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-kind", choices=("formal", "smoke"), required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--feature-reference-commit", default=STATIC_BASELINE_SHA)
    return parser.parse_args(argv)


def _load_cfg(path, run_kind, output_dir=None):
    from config import cfg
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(path))
    overrides = []
    if run_kind == "smoke":
        smoke_output = Path(output_dir).resolve() if output_dir else SMOKE_OUTPUT_DIR
        overrides = [
            "SOLVER.MAX_EPOCHS", "1", "SOLVER.CHECKPOINT_PERIOD", "1",
            "SOLVER.EVAL_PERIOD", "1", "OUTPUT_DIR", str(smoke_output),
        ]
        local_cfg.merge_from_list(overrides)
    elif output_dir:
        raise DynamicExperimentEvidenceError(
            "Formal OUTPUT_DIR must come only from the formal YAML"
        )
    local_cfg.freeze()
    return local_cfg, overrides


def _source_protocol(path, run_kind, output_dir):
    with Path(path).open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle)
    if run_kind == "smoke":
        source = json.loads(json.dumps(source))
        source["SOLVER"]["MAX_EPOCHS"] = 1
        source["SOLVER"]["CHECKPOINT_PERIOD"] = 1
        source["SOLVER"]["EVAL_PERIOD"] = 1
        source["OUTPUT_DIR"] = str(output_dir)
    return source


def _write_console_chunk(decoder, chunk, console_handle, terminal_stream,
                         final=False):
    text = decoder.decode(chunk, final=final)
    if text:
        console_handle.write(text)
        console_handle.flush()
        terminal_stream.write(text)
        terminal_stream.flush()


def launch_training_subprocess(command, environment, console_log_path,
                               on_started=None):
    """Stream a merged stdout/stderr pipe to UTF-8 evidence and the terminal."""
    console_path = Path(console_log_path)
    console_path.parent.mkdir(parents=True, exist_ok=True)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    process = None
    started = time.monotonic()
    previous_sigterm = None

    def interrupt_for_signal(_signum, _frame):
        raise TrainingInterrupted()

    if hasattr(signal, "SIGTERM"):
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, interrupt_for_signal)
    with console_path.open("w", encoding="utf-8", errors="replace", newline="\n") as handle:
        handle.write("[recorder] command={}\n".format(json.dumps(command, ensure_ascii=False)))
        handle.flush()
        try:
            process = subprocess.Popen(
                command, cwd=str(REPO_ROOT), env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
            )
            if on_started is not None:
                on_started(process)
            while True:
                chunk = process.stdout.read(4096)
                if chunk:
                    _write_console_chunk(decoder, chunk, handle, sys.stdout)
                if not chunk and process.poll() is not None:
                    break
            _write_console_chunk(decoder, b"", handle, sys.stdout, final=True)
            return int(process.returncode), time.monotonic() - started
        except KeyboardInterrupt as error:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
                remaining = process.stdout.read() if process.stdout is not None else b""
                if remaining:
                    _write_console_chunk(decoder, remaining, handle, sys.stdout)
            _write_console_chunk(decoder, b"", handle, sys.stdout, final=True)
            raise TrainingInterrupted(
                process.returncode if process is not None else None
            ) from error
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
                remaining = process.stdout.read() if process.stdout is not None else b""
                if remaining:
                    _write_console_chunk(decoder, remaining, handle, sys.stdout)
            _write_console_chunk(decoder, b"", handle, sys.stdout, final=True)
            raise
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            handle.flush()
            os.fsync(handle.fileno())
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)


def _model_manifest(configuration, feature_evidence, pretrained_path,
                    feature_evidence_path):
    from modeling import build_model
    model_cfg = configuration.clone()
    model_cfg.defrost()
    model_cfg.MODEL.PRETRAIN_CHOICE = "none"
    model_cfg.MODEL.PRETRAIN_PATH = ""
    model_cfg.freeze()
    model = build_model(model_cfg, num_classes=751)
    manifest = {
        "schema_version": 2, "backbone": str(configuration.MODEL.NAME),
        "branches": ["global", "K2", "K4", "K6"],
        "scales": list(configuration.MODEL.MULTI_GRANULARITY_PART_SCALES),
        "projection_dim": int(configuration.MODEL.MULTI_GRANULARITY_PART_DIM),
        "aggregation": str(configuration.MODEL.MULTI_GRANULARITY_PART_AGGREGATION),
        "fusion": str(configuration.MODEL.MULTI_GRANULARITY_PART_FUSION),
        "dynamic_gating": True, "gating_input": "global",
        "gating_temperature": float(configuration.MODEL.MULTI_GRANULARITY_GATING_TAU),
        "gating_normalization": "scaled_softmax", "descriptor_dim": int(model.feature_dim),
        "global_descriptor_dim": int(model.in_planes),
        "local_descriptor_dims": [256, 256, 256],
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "state_dict_schema": model_state_dict_schema(dict(model.state_dict())),
        "shared_feature_signature_sha256": feature_evidence["current_feature_signature_sha256"],
        "feature_reference_commit": feature_evidence["feature_reference_commit"],
        "feature_reference_signature_sha256": feature_evidence["feature_reference_signature_sha256"],
        "current_feature_signature_sha256": feature_evidence["current_feature_signature_sha256"],
        "feature_compatibility_status": feature_evidence["feature_compatibility_status"],
        "feature_compatibility_evidence_path": str(Path(feature_evidence_path).resolve()),
        "feature_compatibility_evidence_size_bytes": Path(feature_evidence_path).stat().st_size,
        "feature_compatibility_evidence_sha256": sha256_file(feature_evidence_path),
        "gating_signature_sha256": feature_evidence["fusion_gating_signature"]["current_sha256"],
        "gating_controller_excluded_from_shared_parameter_schema": True,
        "pretrained_weight_path": str(pretrained_path.resolve()),
        "pretrained_weight_sha256": sha256_file(pretrained_path),
        "single_backbone_forward": True,
    }
    if manifest["descriptor_dim"] != 2816:
        raise DynamicExperimentEvidenceError("Model descriptor dimension is not 2816")
    return manifest


def _require_runtime_inputs(configuration):
    data_root = Path(str(configuration.DATASETS.ROOT_DIR))
    pretrained = Path(str(configuration.MODEL.PRETRAIN_PATH))
    if not data_root.is_dir():
        raise DynamicExperimentEvidenceError("Market1501 root is unavailable: {}".format(data_root))
    if not pretrained.is_file():
        raise DynamicExperimentEvidenceError("ImageNet weights are unavailable: {}".format(pretrained))
    if configuration.MODEL.DEVICE == "cuda" and not torch.cuda.is_available():
        raise DynamicExperimentEvidenceError("CUDA is required by the formal/smoke protocol")
    return data_root, pretrained


def _require_successful_smoke_before_formal(records_root):
    registry_path = Path(records_root) / "runs.csv"
    if not registry_path.is_file():
        raise DynamicExperimentEvidenceError(
            "Formal training requires a successfully recorded 1-epoch smoke first"
        )
    with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [
        row for row in rows
        if row.get("experiment_id") == SMOKE_EXPERIMENT_ID
        and row.get("run_kind") == "smoke"
        and row.get("status") == "success"
        and row.get("feature_compatibility_status") == "compatible"
        and row.get("gating_temperature") in ("1", "1.0")
        and row.get("gating_sample_count") not in (
            None, "", "0", "not_recorded", "missing_evidence"
        )
    ]
    if not eligible:
        raise DynamicExperimentEvidenceError(
            "Formal training is blocked until a successful, compatible tau=1.0 "
            "smoke with nonzero gating samples is committed to runs.csv"
        )
    return eligible[-1]


def _metrics(selected_validation):
    return {
        "rank1_percent": float(selected_validation["rank1_percent"]),
        "rank5_percent": float(selected_validation["rank5_percent"]),
        "rank10_percent": float(selected_validation["rank10_percent"]),
        "map_percent": float(selected_validation["map_percent"]),
        "best_epoch": int(selected_validation["epoch"]),
        "selected_epoch": int(selected_validation["epoch"]),
    }


def run(args):
    started = time.monotonic()
    started_at_utc = dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0
    ).isoformat()
    config_path = Path(args.config).resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise DynamicExperimentEvidenceError("Dynamic identity requires the fixed formal YAML")
    if args.feature_reference_commit != STATIC_BASELINE_SHA:
        raise DynamicExperimentEvidenceError("Feature reference commit is fixed and immutable")
    lineage = validate_dynamic_lineage(REPO_ROOT)
    local_cfg, overrides = _load_cfg(config_path, args.run_kind, args.output_dir)
    output_dir = Path(str(local_cfg.OUTPUT_DIR)).resolve()
    static_source = yaml.safe_load(STATIC_CONFIG.read_text(encoding="utf-8"))
    dynamic_source = _source_protocol(config_path, args.run_kind, output_dir)
    validate_dynamic_configuration(dynamic_source, static_source, args.run_kind)
    if args.run_kind == "formal":
        _require_successful_smoke_before_formal(RECORDS_ROOT)
    _require_runtime_inputs(local_cfg)
    feature = require_feature_compatibility(
        build_feature_compatibility_evidence(
            REPO_ROOT, STATIC_BASELINE_SHA, lineage["commit"],
            python_executable=sys.executable,
        )
    )
    experiment_id = FORMAL_EXPERIMENT_ID if args.run_kind == "formal" else SMOKE_EXPERIMENT_ID
    train_command = [sys.executable, "tools/train.py", "--config_file", str(config_path)] + overrides
    run_dir, _manifest = initialize_dynamic_run(
        RECORDS_ROOT, EXPERIMENTS_PATH, experiment_id, args.run_kind,
        config_path, serialize_cfg_node_yaml(local_cfg), output_dir, lineage,
        feature, train_command,
        started_at_utc=started_at_utc,
    )
    environment = None
    dataset_manifest = None
    model_manifest = None
    try:
        os.environ["PYTHONHASHSEED"] = "42"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_cfg.MODEL.DEVICE_ID)
        environment = collect_environment(run_dir)
        atomic_write_json(run_dir / "environment.json", environment)
        from data.build import collect_dataset_protocol
        dataset_manifest, _num_classes = collect_dataset_protocol(local_cfg)
        atomic_write_json(run_dir / "dataset_manifest.json", dataset_manifest)
        model_manifest = _model_manifest(
            local_cfg, feature, Path(str(local_cfg.MODEL.PRETRAIN_PATH)),
            run_dir / "shared_feature_compatibility.json",
        )
        atomic_write_json(run_dir / "model_manifest.json", model_manifest)
        training_env = os.environ.copy()
        training_env.update({
            "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": str(local_cfg.MODEL.DEVICE_ID),
        })
        return_code, _training_runtime = launch_training_subprocess(
            train_command, training_env, run_dir / "console.log",
            on_started=lambda _process: transition_dynamic_run(
                run_dir, "running"
            ),
        )
        runtime = time.monotonic() - started
        if return_code != 0:
            transition_dynamic_run(
                run_dir, "failed", return_code=return_code,
                runtime_seconds=runtime,
                error="Training exited with code {}".format(return_code),
            )
            return return_code
        transition_dynamic_run(
            run_dir, "training_complete", return_code=0,
            runtime_seconds=runtime,
        )
        validate_dynamic_runtime_worktree(REPO_ROOT, run_dir, output_dir)
        validation_records = read_validation_history(
            output_dir / "validation_history.jsonl"
        )
        checkpoints = build_dynamic_checkpoint_manifest(
            output_dir, validation_records
        )
        selected_checkpoint, selected_validation = select_dynamic_checkpoint(
            checkpoints, validation_records
        )
        epoch_rows = read_gating_epoch_records(
            output_dir / "dynamic_gating_epoch_stats.jsonl"
        )
        selected_epoch_rows = [
            row for row in epoch_rows
            if int(row["epoch"]) == int(selected_validation["epoch"])
        ]
        if len(selected_epoch_rows) != 1:
            raise DynamicExperimentEvidenceError(
                "Selected epoch lacks unique Dynamic Gating statistics"
            )
        gating_statistics = {
            key: selected_epoch_rows[0][key]
            for key in selected_epoch_rows[0]
            if key not in ("epoch", "global_iteration", "epoch_length")
        }
        from tools.analyze_dynamic_gating import generate_dynamic_gating_evidence
        summary_path, samples_path, _summary = generate_dynamic_gating_evidence(
            local_cfg,
            output_dir / selected_checkpoint["relative_path"],
            run_dir,
            gating_statistics,
        )
        manifest = seal_dynamic_run_evidence(
            run_dir, environment, dataset_manifest, model_manifest,
            checkpoints, selected_checkpoint, _metrics(selected_validation),
            summary_path, samples_path, gating_statistics,
            runtime_seconds=time.monotonic() - started, return_code=0,
        )
        validate_dynamic_runtime_worktree(REPO_ROOT, run_dir, output_dir)
        print(json.dumps({"run_id": manifest["run_id"], "status": "success"}, indent=2))
        return 0
    except TrainingInterrupted as error:
        child_return_code = (
            error.child_return_code
            if error.child_return_code is not None else 130
        )
        transition_dynamic_run(
            run_dir, "interrupted", return_code=child_return_code,
            runtime_seconds=time.monotonic() - started, error=error,
        )
        return 130
    except BaseException as error:
        try:
            transition_dynamic_run(
                run_dir, "incomplete", return_code=1,
                runtime_seconds=time.monotonic() - started, error=error,
            )
        except BaseException as recording_error:
            print("Recorder failure: {}".format(recording_error), file=sys.stderr)
        raise


def main(argv=None):
    try:
        return run(parse_args(argv))
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
