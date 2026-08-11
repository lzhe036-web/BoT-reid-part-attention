import csv
import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml
import torch
from yacs.config import CfgNode

from config import cfg
from modeling import build_model
from modeling.baseline import MultiGranularityPartHead
from tools.run_c2_l03_multi_granularity_part import main as runner_main
import tools.run_c2_l03_multi_granularity_part as runner
from utils.experiment_recording import (
    CHECKPOINT_SELECTION_RULE,
    EXPECTED_BRANCH,
    EXPECTED_EVIDENCE_ID,
    EXPECTED_EXPERIMENT_FAMILY,
    EXPECTED_RUN_ID,
    FORMAL_CONFIG_RELATIVE_PATH,
    FORMAL_PROTOCOL,
    LOCAL_EVIDENCE_PENDING,
    NOT_APPLICABLE,
    NOT_ARCHIVED,
    TRAINING_COMPLETE,
    EvidenceIncompleteError,
    PreflightError,
    SAMPLER_EPOCH_SEED_RULE,
    append_validation_record,
    assert_path_allowed,
    atomic_write_json,
    atomic_write_text,
    build_dataset_manifest,
    collect_environment,
    deserialize_cfg_node_yaml,
    finalize_run,
    formal_preflight,
    recover_existing_run,
    select_best_validation,
    serialize_cfg_node_yaml,
    sha256_file,
    utc_now,
    validate_efficiency_profile,
    validate_environment_schema,
    validate_formal_protocol,
)
from utils.reproducibility import data_loader_generator_metadata
from utils.config_serialization import BOT_CFG_TYPE_TAG
import utils.experiment_recording as recording


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_TOOL = REPO_ROOT / "tools" / "profile_multi_granularity_part.py"
EXPERIMENT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "softmax_triplet_c2_l03_multi_granularity_part_autodl.yml"
)


class FakeMonotonicClock(object):
    def __init__(self, value=100.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


def run_git(repo, *args):
    subprocess.check_call(
        ["git", "-C", str(repo)] + list(args),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def formal_source(output, data_root, pretrained):
    source = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    source["OUTPUT_DIR"] = str(output)
    source["DATASETS"]["ROOT_DIR"] = str(data_root)
    source["MODEL"]["PRETRAIN_PATH"] = str(pretrained)
    return source


def formal_runner_fixture(root):
    output = root / "formal_output"
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(EXPERIMENT_CONFIG))
    local_cfg.defrost()
    local_cfg.OUTPUT_DIR = str(output)
    local_cfg.freeze()
    args = SimpleNamespace(
        config=str(runner.DEFAULT_CONFIG),
        experiment_family=EXPECTED_EXPERIMENT_FAMILY,
        run_id=EXPECTED_RUN_ID,
        evidence_id=EXPECTED_EVIDENCE_ID,
    )
    preflight = {
        "branch": EXPECTED_BRANCH,
        "training_commit": "a" * 40,
        "dirty": False,
        "source_config_path": str(EXPERIMENT_CONFIG),
        "source_config_sha256": sha256_file(EXPERIMENT_CONFIG),
        "launch_script_path": str(
            REPO_ROOT / "scripts" /
            "train_c2_l03_multi_granularity_part_autodl.sh"
        ),
        "launch_script_sha256": "b" * 64,
        "pretrained_weight_path": str(root / "unused.pth"),
        "pretrained_weight_sha256": "c" * 64,
        "data_root": str(root / "synthetic_data"),
        "output_dir": str(output),
        "training_seed": 42,
    }
    environment = {
        "hostname": "synthetic-host",
        "gpu_count": 1,
        "pythonhashseed": "42",
        "cublas_workspace_config": ":4096:8",
    }
    return output, local_cfg, args, preflight, environment


def set_dotted(mapping, dotted_path, value):
    current = mapping
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def flatten_typed_config_leaves(node, prefix=""):
    leaves = {}
    for key, value in node.items():
        dotted_path = "{}.{}".format(prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            leaves.update(flatten_typed_config_leaves(value, dotted_path))
        else:
            leaves[dotted_path] = value
    return leaves


def make_preflight_repository(root, branch=EXPECTED_BRANCH, seed=42,
                               include_seed=True, dirty=False,
                               nonempty_output=False):
    repo = root / "repo"
    repo.mkdir()
    data_root = root / "synthetic_data"
    data_root.mkdir()
    pretrained = root / "imagenet_pretrained.pth"
    pretrained.write_bytes(b"synthetic ImageNet initialization")
    output = root / "c2_l03_multi_granularity_part_market1501"
    config_path = repo / FORMAL_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    source = formal_source(output, data_root, pretrained)
    if include_seed:
        source["SEED"] = seed
    else:
        source.pop("SEED", None)
    config_path.write_text(
        yaml.safe_dump(source, sort_keys=False), encoding="utf-8", newline="\n"
    )
    launch = repo / "launch.sh"
    launch.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")

    run_git(repo, "init", "-b", branch)
    run_git(repo, "config", "user.name", "Fixture")
    run_git(repo, "config", "user.email", "fixture@example.invalid")
    run_git(repo, "add", FORMAL_CONFIG_RELATIVE_PATH, "launch.sh")
    run_git(repo, "commit", "-m", "fixture")
    if dirty:
        launch.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8", newline="\n")
    if nonempty_output:
        output.mkdir()
        (output / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    return repo, config_path, launch, output


def run_preflight(repo, config_path, launch, output, record_dir=None):
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return formal_preflight(
        repo, config_path, launch, output,
        EXPECTED_EXPERIMENT_FAMILY, EXPECTED_RUN_ID, EXPECTED_EVIDENCE_ID,
        resolved_config=resolved, record_dir=record_dir,
    )


def validation_records():
    return [
        {
            "epoch": 40,
            "global_iteration": 7440,
            "timestamp_utc": "2026-01-01T00:00:40Z",
            "rank1_percent": 80.0,
            "rank5_percent": 90.0,
            "rank10_percent": 93.0,
            "map_percent": 70.0,
            "re_ranking": "no",
            "neck_feat": "after",
            "feat_norm": "yes",
        },
        {
            "epoch": 80,
            "global_iteration": 14880,
            "timestamp_utc": "2026-01-01T00:01:20Z",
            "rank1_percent": 85.0,
            "rank5_percent": 94.0,
            "rank10_percent": 96.0,
            "map_percent": 72.0,
            "re_ranking": "no",
            "neck_feat": "after",
            "feat_norm": "yes",
        },
        {
            "epoch": 120,
            "global_iteration": 22320,
            "timestamp_utc": "2026-01-01T00:02:00Z",
            "rank1_percent": 85.0,
            "rank5_percent": 95.0,
            "rank10_percent": 97.0,
            "map_percent": 75.0,
            "re_ranking": "no",
            "neck_feat": "after",
            "feat_norm": "yes",
        },
    ]


def synthetic_model_state_dict():
    return {
        "base.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "bn.running_mean": torch.zeros(2, dtype=torch.float32),
        "bn.num_batches_tracked": torch.tensor(0, dtype=torch.int64),
    }


def fixture_efficiency(source_hash, resolved_hash):
    baseline = {
        "name": "C2-L03",
        "variant": "legacy",
        "measurement_seed": 42,
        "feature_dim": 2048,
        "total_parameters": 25000000,
        "trainable_parameters": 24997952,
        "flops": 8000000000,
        "macs": 4000000000,
        "inference_latency_median_ms": 10.0,
        "inference_latency_p95_ms": 12.0,
        "throughput_images_per_second": 6400.0,
        "forward_peak_memory": {
            "peak_allocated_mib": NOT_APPLICABLE,
            "peak_reserved_mib": NOT_APPLICABLE,
        },
        "forward_backward_peak_memory": {
            "peak_allocated_mib": NOT_APPLICABLE,
            "peak_reserved_mib": NOT_APPLICABLE,
        },
    }
    experiment = dict(baseline)
    experiment.update({
        "name": "C2-L03 + Multi-Granularity K={2,4,6}",
        "variant": "multi_granularity",
        "feature_dim": 2816,
        "total_parameters": 27202880,
        "trainable_parameters": 27200064,
        "flops": 8200000000,
        "macs": 4100000000,
        "inference_latency_median_ms": 11.0,
        "inference_latency_p95_ms": 13.0,
        "throughput_images_per_second": 64.0 * 1000.0 / 11.0,
    })
    deltas = {}
    pairs = {
        "total_parameters": (25000000, 27202880),
        "trainable_parameters": (24997952, 27200064),
        "descriptor_dim": (2048, 2816),
        "flops": (8000000000, 8200000000),
        "macs": (4000000000, 4100000000),
        "latency_median_ms": (10.0, 11.0),
        "latency_p95_ms": (12.0, 13.0),
        "throughput_images_per_second": (6400.0, 64.0 * 1000.0 / 11.0),
    }
    for name, (before, after) in pairs.items():
        deltas[name] = {
            "absolute": after - before,
            "percent": 100.0 * (after - before) / before,
        }
    for name in (
            "memory_forward_peak_allocated_mib",
            "memory_forward_peak_reserved_mib",
            "memory_forward_backward_peak_allocated_mib",
            "memory_forward_backward_peak_reserved_mib"):
        deltas[name] = {
            "absolute": NOT_APPLICABLE,
            "percent": NOT_APPLICABLE,
        }
    return {
        "schema_version": 2,
        "status": "complete",
        "mode": "fixture",
        "measurement_timestamp_utc": utc_now(),
        "measurement_seed": 42,
        "seed_source": "explicit --measurement-seed",
        "python_executable": sys.executable,
        "argv": [str(PROFILE_TOOL.resolve()), "--mode", "fixture"],
        "display_command": shlex.join([
            sys.executable, str(PROFILE_TOOL.resolve()), "--mode", "fixture",
        ]),
        "profiler_script_sha256": sha256_file(PROFILE_TOOL),
        "source_config_sha256": source_hash,
        "resolved_config_sha256": resolved_hash,
        "config_overrides": {
            "MODEL.PRETRAIN_CHOICE": "none",
            "MODEL.PRETRAIN_PATH": "",
        },
        "measurement": {
            "num_classes": 751,
            "batch_size": 64,
            "input_size": [256, 128],
            "dtype": "float32",
            "device": "cpu",
            "gpu_name": NOT_APPLICABLE,
            "gpu_total_memory_mib": NOT_APPLICABLE,
            "nvidia_driver": NOT_APPLICABLE,
            "pytorch_version": "fixture",
            "cuda_runtime": NOT_APPLICABLE,
            "cudnn_version": NOT_APPLICABLE,
            "warmup": 0,
            "measurement_repeats": 1,
            "worker_isolation": recording.PROFILER_WORKER_ISOLATION,
            "operation_count_convention": recording.OPERATION_COUNT_CONVENTION,
        },
        "variants": [baseline, experiment],
        "deltas": deltas,
    }


def formal_efficiency(output, source_config, resolved_config, environment):
    profile = fixture_efficiency(
        sha256_file(source_config), sha256_file(resolved_config)
    )
    profile["mode"] = "formal"
    profile["python_executable"] = environment["python_executable"]
    argv = [
        str(PROFILE_TOOL.resolve()),
        "--config", str(source_config.resolve()),
        "--resolved-config", str(resolved_config.resolve()),
        "--source-config-sha256", sha256_file(source_config),
        "--resolved-config-sha256", sha256_file(resolved_config),
        "--mode", "formal",
        "--measurement-seed", "42",
        "--device", "cuda",
        "--batch-size", "64",
        "--input-height", "256",
        "--input-width", "128",
        "--dtype", "float32",
        "--warmup", "5",
        "--measurement-repeats", "20",
        "--num-classes", "751",
        "--output-file", str((output / "efficiency_profile.json").resolve()),
    ]
    profile["argv"] = argv
    profile["display_command"] = shlex.join(
        [profile["python_executable"]] + argv
    )
    profile["profiler_script_sha256"] = sha256_file(PROFILE_TOOL)
    profile["measurement"].update({
        "device": "cuda",
        "gpu_name": environment["gpus"][0]["name"],
        "gpu_total_memory_mib": environment["gpus"][0]["total_memory_mib"],
        "nvidia_driver": environment["nvidia_driver"],
        "pytorch_version": environment["pytorch_version"],
        "cuda_runtime": environment["cuda_runtime"],
        "cudnn_version": environment["cudnn_version"],
        "warmup": 5,
        "measurement_repeats": 20,
    })
    for index, variant in enumerate(profile["variants"]):
        variant["forward_peak_memory"] = {
            "peak_allocated_mib": 1000.0 + 100.0 * index,
            "peak_reserved_mib": 1200.0 + 100.0 * index,
        }
        variant["forward_backward_peak_memory"] = {
            "peak_allocated_mib": 3000.0 + 200.0 * index,
            "peak_reserved_mib": 3400.0 + 200.0 * index,
        }
    for name, before, after in (
            ("memory_forward_peak_allocated_mib", 1000.0, 1100.0),
            ("memory_forward_peak_reserved_mib", 1200.0, 1300.0),
            ("memory_forward_backward_peak_allocated_mib", 3000.0, 3200.0),
            ("memory_forward_backward_peak_reserved_mib", 3400.0, 3600.0)):
        profile["deltas"][name] = {
            "absolute": after - before,
            "percent": 100.0 * (after - before) / before,
        }
    return profile


def make_success_fixture(root, source_seed=True, metadata_seed=42,
                          selected_checkpoint=True, exit_code=0,
                          execution_mode="fixture", finalization_complete=True):
    output = root / "fixture"
    output.mkdir()
    source_config = (
        output / FORMAL_CONFIG_RELATIVE_PATH
        if execution_mode == "formal" else output / "source.yml"
    )
    source_config.parent.mkdir(parents=True, exist_ok=True)
    data_root = output / "synthetic_dataset"
    pretrained = output / "imagenet_pretrained.pth"
    source = formal_source(output, data_root, pretrained)
    if source_seed:
        source["SEED"] = 42
    else:
        source.pop("SEED", None)
    atomic_write_text(source_config, yaml.safe_dump(source, sort_keys=False))
    resolved_config = output / "config_resolved.yml"
    resolved_cfg = cfg.clone()
    resolved_cfg.merge_from_file(str(source_config))
    resolved_cfg.freeze()
    atomic_write_text(resolved_config, serialize_cfg_node_yaml(resolved_cfg))
    launch_script = output / "launch.sh"
    atomic_write_text(launch_script, "#!/usr/bin/env bash\nexit 0\n")
    pretrained.write_bytes(b"synthetic pretrained weight")

    manifest = {
        "schema_version": 1,
        "experiment_family": EXPECTED_EXPERIMENT_FAMILY,
        "run_id": EXPECTED_RUN_ID,
        "evidence_id": EXPECTED_EVIDENCE_ID,
        "branch": EXPECTED_BRANCH,
        "training_commit": "a" * 40,
        "dirty": False,
        "source_config": {
            "path": str(source_config),
            "sha256": sha256_file(source_config),
        },
        "resolved_config": {
            "path": "config_resolved.yml",
            "sha256": sha256_file(resolved_config),
        },
        "launch_script": {
            "path": str(launch_script),
            "sha256": sha256_file(launch_script),
        },
        "argv": ["python", "tools/run_c2_l03_multi_granularity_part.py"],
        "shell_launch_command": "bash scripts/train_c2_l03_multi_granularity_part_autodl.sh",
        "cwd": str(output),
        "started_at_utc": "2026-01-01T00:00:00Z",
        "ended_at_utc": "2026-01-01T00:00:12Z",
        "timezone": "UTC",
        "total_run_runtime_seconds": 20.0,
        "environment_collection_runtime_seconds": 1.5,
        "profiling_runtime_seconds": 5.0,
        "training_runtime_seconds": 12.5,
        "finalization_runtime_seconds": 1.0 if finalization_complete else 0.0,
        "finalization_runtime_complete": finalization_complete,
        "runtime_source": "time.monotonic",
        "exit_code": exit_code,
        "status": "training_succeeded_pending_evidence" if exit_code == 0 else "failed",
        "training_seed": 42,
        "seed_source": "source_config.SEED",
        "dataset": "market1501",
        "data_root": str(output / "synthetic_dataset"),
        "output_dir": str(output),
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "execution_mode": execution_mode,
    }
    if finalization_complete:
        manifest["finalization_timing_boundary"] = (
            recording.FINALIZATION_TIMING_BOUNDARY
        )
    status = {
        "schema_version": 1,
        "run_id": EXPECTED_RUN_ID,
        "status": manifest["status"],
        "started_at_utc": manifest["started_at_utc"],
        "ended_at_utc": manifest["ended_at_utc"],
        "total_run_runtime_seconds": 20.0,
        "environment_collection_runtime_seconds": 1.5,
        "profiling_runtime_seconds": 5.0,
        "training_runtime_seconds": 12.5,
        "finalization_runtime_seconds": 1.0 if finalization_complete else 0.0,
        "finalization_runtime_complete": finalization_complete,
        "runtime_source": "time.monotonic",
        "exit_code": exit_code,
        "errors": [],
    }
    if finalization_complete:
        status["finalization_timing_boundary"] = (
            recording.FINALIZATION_TIMING_BOUNDARY
        )
    atomic_write_json(output / "run_manifest.json", manifest)
    atomic_write_json(output / "run_status.json", status)

    reproducibility = {
        "schema_version": 1,
        "seed": metadata_seed,
        "seed_source": "resolved_config.SEED",
        "seed_applied_before_data_loading": True,
        "seed_chain": {
            "source_config_seed": 42,
            "resolved_config_seed": 42,
            "applied_training_seed": metadata_seed,
            "reproducibility_metadata_seed": metadata_seed,
        },
        "random_state": {
            "seed": metadata_seed,
            "python_random_seed": metadata_seed,
            "python_random_seeded": True,
            "numpy_seed": metadata_seed,
            "numpy_seeded": True,
            "torch_cpu_seed": metadata_seed,
            "torch_cpu_seeded": True,
            "torch_cuda_manual_seed_all_seed": "not_recorded",
            "torch_cuda_manual_seed_all_called": False,
            "torch_cuda_all_seeded": False,
            "cuda_available": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "pythonhashseed": "42",
            "cublas_workspace_config": ":4096:8",
        },
        "data_loader_worker_seeding": {
            "enabled": True,
            "scheme": "torch.initial_seed() modulo 2**32 -> Python random and NumPy",
            "num_workers": 8,
        },
        "random_identity_sampler": {
            "base_seed": 42,
            "epoch_seed_rule": SAMPLER_EPOCH_SEED_RULE,
        },
        "data_loader_generators": data_loader_generator_metadata(42),
        "configuration": {
            "source_file_sha256": sha256_file(source_config),
            "resolved_file_sha256": sha256_file(resolved_config),
        },
    }
    if execution_mode == "formal":
        reproducibility["random_state"]["torch_cuda_manual_seed_all_seed"] = 42
        reproducibility["random_state"]["torch_cuda_manual_seed_all_called"] = True
        reproducibility["random_state"]["torch_cuda_all_seeded"] = True
        reproducibility["random_state"]["cuda_available"] = True
    atomic_write_json(output / "reproducibility.json", reproducibility)

    atomic_write_text(output / "environment_packages.txt", "synthetic==1.0\n")
    environment = {
        "schema_version": 1,
        "hostname": "synthetic-host",
        "os": "synthetic-os",
        "kernel": "synthetic-kernel",
        "machine_architecture": "x86_64",
        "python_version": "3.test",
        "python_executable": sys.executable,
        "pytorch_version": "test",
        "torchvision_version": "test",
        "ignite_version": "test",
        "yacs_version": "test",
        "numpy_version": "test",
        "pillow_version": "test",
        "cuda_runtime": "not_recorded",
        "cudnn_version": "not_recorded",
        "nvidia_driver": "not_recorded",
        "gpu_count": 0,
        "gpus": [],
        "cuda_visible_devices": "not_recorded",
        "pythonhashseed": "42",
        "cublas_workspace_config": ":4096:8",
        "timezone": "UTC",
        "pip_freeze_path": "environment_packages.txt",
        "pip_freeze_sha256": sha256_file(output / "environment_packages.txt"),
    }
    if execution_mode == "formal":
        environment.update({
            "pytorch_version": "synthetic-pytorch",
            "cuda_runtime": "synthetic-cuda",
            "cudnn_version": 9999,
            "nvidia_driver": "synthetic-driver",
            "gpu_count": 1,
            "gpus": [{
                "index": 0,
                "name": "Synthetic GPU",
                "uuid": "GPU-SYNTHETIC",
                "total_memory_mib": 24576.0,
                "driver": "synthetic-driver",
            }],
        })
    atomic_write_json(output / "environment.json", environment)

    data_root.mkdir()
    splits = {
        "train": [
            (data_root / "train_b.jpg", 1, 1),
            (data_root / "train_a.jpg", 1, 0),
            (data_root / "train_c.jpg", 2, 0),
        ],
        "query": [(data_root / "query.jpg", 1, 0)],
        "gallery": [(data_root / "gallery.jpg", 1, 1)],
    }
    dataset_manifest = build_dataset_manifest(
        splits, data_root, "market1501", "softmax_triplet", 64, 4, 8, 186, 42,
        data_loader_generators=data_loader_generator_metadata(42),
    )
    atomic_write_json(output / "dataset_manifest.json", dataset_manifest)
    atomic_write_json(output / "model_manifest.json", {
        "schema_version": 1,
        "backbone": "resnet50",
        "feature_map_shape": [1, 2048, 16, 8],
        "branches": ["Global", "K2", "K4", "K6"],
        "scales": [2, 4, 6],
        "projection_dim": 256,
        "aggregation": "mean",
        "fusion": "concat",
        "descriptor_dim": 2816,
        "num_classes": 751,
        "total_parameters": 27202880,
        "trainable_parameters": 27200064,
        "state_dict_schema": recording.model_state_dict_schema(
            synthetic_model_state_dict()
        ),
        "pretrained_weight_path": str(pretrained),
        "pretrained_weight_sha256": sha256_file(pretrained),
    })
    efficiency = (
        formal_efficiency(output, source_config, resolved_config, environment)
        if execution_mode == "formal"
        else fixture_efficiency(
            sha256_file(source_config), sha256_file(resolved_config)
        )
    )
    atomic_write_json(output / "efficiency_profile.json", efficiency)
    for record in validation_records():
        append_validation_record(output, record)
    for record in validation_records():
        if record["epoch"] == 120 and not selected_checkpoint:
            continue
        torch.save(
            synthetic_model_state_dict(),
            output / "resnet50_checkpoint_{}.pt".format(
                record["global_iteration"]
            ),
        )
    atomic_write_text(
        output / "log.txt",
        "".join(
            "Epoch[{}] Iteration[186/186] synthetic training log\n".format(
                record["epoch"]
            )
            for record in validation_records()
        ),
    )
    return output


def make_recovery_fixture(root):
    output = make_success_fixture(
        root,
        execution_mode="formal",
        finalization_complete=False,
    )
    run_git(output, "init", "-b", EXPECTED_BRANCH)
    run_git(output, "config", "user.name", "Recovery Fixture")
    run_git(output, "config", "user.email", "recovery@example.invalid")
    committed_profiler = output / "tools" / PROFILE_TOOL.name
    committed_profiler.parent.mkdir(parents=True)
    committed_profiler.write_bytes(PROFILE_TOOL.read_bytes())
    run_git(
        output,
        "add",
        FORMAL_CONFIG_RELATIVE_PATH,
        "tools/{}".format(PROFILE_TOOL.name),
    )
    run_git(output, "commit", "-m", "synthetic training commit")
    training_commit = subprocess.check_output(
        ["git", "-C", str(output), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    marker = output / "recovery_code_marker.txt"
    marker.write_text("type-safe recovery code\n", encoding="utf-8")
    run_git(output, "add", marker.name)
    run_git(output, "commit", "-m", "synthetic recovery commit")
    finalization_commit = subprocess.check_output(
        ["git", "-C", str(output), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    source_config = output / FORMAL_CONFIG_RELATIVE_PATH
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(source_config))
    broken_text = str(local_cfg).rstrip() + "\n"
    resolved_path = output / "config_resolved.yml"
    atomic_write_text(resolved_path, broken_text)
    broken_hash = sha256_file(resolved_path)

    manifest = recording.read_json(output / "run_manifest.json")
    manifest["training_commit"] = training_commit
    manifest["resolved_config"] = {
        "path": "config_resolved.yml",
        "sha256": broken_hash,
    }
    atomic_write_json(output / "run_manifest.json", manifest)
    reproducibility = recording.read_json(output / "reproducibility.json")
    reproducibility["configuration"]["resolved_file_sha256"] = broken_hash
    atomic_write_json(output / "reproducibility.json", reproducibility)
    efficiency = recording.read_json(output / "efficiency_profile.json")
    recording._replace_profile_resolved_hash(efficiency, broken_hash)
    atomic_write_json(output / "efficiency_profile.json", efficiency)
    return output, training_commit, finalization_commit, broken_text.encode("utf-8")


def mark_known_post_training_finalization_failure(output):
    manifest_path = output / "run_manifest.json"
    status_path = output / "run_status.json"
    manifest = recording.read_json(manifest_path)
    status = recording.read_json(status_path)
    manifest["exit_code"] = 1
    manifest["status"] = "failed"
    status["exit_code"] = 1
    status["status"] = "failed"
    status["errors"] = [recording.KNOWN_CONFIG_FINALIZATION_ERROR]
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(status_path, status)
    log_path = output / "log.txt"
    existing = log_path.read_text(encoding="utf-8")
    completion = (
        "[experiment recorder] ended_at_utc={}, exit_code=0, "
        "pre_finalization_elapsed_seconds=19.500000, "
        "environment_collection_runtime_seconds=1.500000, "
        "profiling_runtime_seconds=5.000000, "
        "training_runtime_seconds=12.500000, "
        "runtime_source=time.monotonic\n"
    ).format(manifest["ended_at_utc"])
    if completion not in existing:
        atomic_write_text(log_path, existing + completion)


def make_checkpoint_binding_fixture(root, records=None, checkpoint_iterations=None,
                                    log_iterations_per_epoch=186,
                                    structured_iterations_per_epoch=186):
    output = root / "checkpoint_binding"
    output.mkdir()
    records = list(records if records is not None else validation_records())
    for record in records:
        append_validation_record(output, record)
    if log_iterations_per_epoch is not None:
        atomic_write_text(
            output / "log.txt",
            "".join(
                "Epoch[{}] Iteration[{}/{}] fixture\n".format(
                    record["epoch"],
                    log_iterations_per_epoch,
                    log_iterations_per_epoch,
                )
                for record in records
            ),
        )
    if structured_iterations_per_epoch is not None:
        atomic_write_json(
            output / "dataset_manifest.json",
            {"train_loader_batches": structured_iterations_per_epoch},
        )
    iterations = (
        checkpoint_iterations
        if checkpoint_iterations is not None
        else [record["global_iteration"] for record in records]
    )
    for iteration in iterations:
        (output / "resnet50_checkpoint_{}.pt".format(iteration)).write_bytes(
            "checkpoint {}\n".format(iteration).encode("ascii")
        )
    return output, records, select_best_validation(records)


class MultiGranularityExperimentRecordingTest(unittest.TestCase):
    def test_resolved_config_yaml_round_trip_preserves_all_leaf_types(self):
        source_cfg = cfg.clone()
        source_cfg.merge_from_file(str(EXPERIMENT_CONFIG))
        source_cfg.freeze()

        self.assertIs(type(source_cfg.MODEL.DEVICE_ID), str)
        self.assertEqual(source_cfg.MODEL.DEVICE_ID, "0")
        self.assertIs(type(source_cfg.MODEL.IF_LABELSMOOTH), str)
        self.assertIs(type(source_cfg.MODEL.IF_WITH_CENTER), str)

        with tempfile.TemporaryDirectory() as directory:
            resolved_path = Path(directory) / "resolved.yml"
            resolved_text = serialize_cfg_node_yaml(source_cfg)
            atomic_write_text(resolved_path, resolved_text)

            self.assertIn(BOT_CFG_TYPE_TAG + ": str", resolved_text)
            restored = deserialize_cfg_node_yaml(resolved_text)
            from tools.profile_multi_granularity_part import (
                declared_resolved_output_dir,
            )
            self.assertEqual(
                declared_resolved_output_dir(resolved_path),
                Path(str(source_cfg.OUTPUT_DIR)).resolve(),
            )

        source_leaves = flatten_typed_config_leaves(source_cfg)
        restored_leaves = flatten_typed_config_leaves(restored)
        self.assertEqual(set(source_leaves), set(restored_leaves))
        for dotted_path, source_value in source_leaves.items():
            with self.subTest(config_key=dotted_path):
                restored_value = restored_leaves[dotted_path]
                self.assertEqual(restored_value, source_value)
                self.assertIs(type(restored_value), type(source_value))

        self.assertIs(type(restored["MODEL"]["DEVICE_ID"]), str)
        self.assertEqual(restored["MODEL"]["DEVICE_ID"], "0")
        self.assertIs(type(restored["MODEL"]["IF_LABELSMOOTH"]), str)
        self.assertEqual(restored["MODEL"]["IF_LABELSMOOTH"], "on")
        self.assertIs(type(restored["MODEL"]["IF_WITH_CENTER"]), str)
        self.assertEqual(restored["MODEL"]["IF_WITH_CENTER"], "no")
        self.assertIs(restored["MODEL"]["CROSS_CAMERA_POSITIVE_ONLY"], True)
        self.assertIs(restored["MODEL"]["MULTI_GRANULARITY_PART"], True)
        self.assertIs(type(restored["SEED"]), int)
        self.assertEqual(restored["SEED"], 42)
        self.assertIs(type(restored["MODEL"]["MULTI_GRANULARITY_PART_SCALES"]), list)
        self.assertIs(type(source_cfg.SOLVER.STEPS), tuple)
        self.assertIs(type(restored["SOLVER"]["STEPS"]), tuple)
        self.assertEqual(restored["SOLVER"]["STEPS"], source_cfg.SOLVER.STEPS)
        self.assertTrue(validate_formal_protocol(restored, "resolved"))

    def test_typed_config_serialization_preserves_ambiguous_strings_and_scalars(self):
        ambiguous = [
            "0", "1", "on", "off", "yes", "no", "true", "false",
            "none", "null", "[]", "()",
        ]
        source = CfgNode({
            "STRINGS": ambiguous,
            "BOOL": True,
            "INT": 42,
            "FLOAT": 0.5,
            "NONE": None,
            "LIST": [1, "on", False],
            "TUPLE": ("0", 2, True),
        })
        restored = deserialize_cfg_node_yaml(serialize_cfg_node_yaml(source))
        self.assertEqual(restored, dict(source))
        for expected, actual in zip(ambiguous, restored["STRINGS"]):
            self.assertIs(type(actual), str)
            self.assertEqual(actual, expected)
        for key, expected_type in (
                ("BOOL", bool), ("INT", int), ("FLOAT", float),
                ("NONE", type(None)), ("LIST", list), ("TUPLE", tuple)):
            self.assertIs(type(restored[key]), expected_type)
        self.assertIs(type(restored["LIST"][1]), str)
        self.assertIs(type(restored["TUPLE"][0]), str)

    def test_typed_config_serialization_rejects_malformed_or_reserved_tags(self):
        malformed = (
            {BOT_CFG_TYPE_TAG: "str"},
            {BOT_CFG_TYPE_TAG: "str", "value": True},
            {BOT_CFG_TYPE_TAG: "tuple"},
            {BOT_CFG_TYPE_TAG: "tuple", "items": "not-a-list"},
            {BOT_CFG_TYPE_TAG: "unknown", "value": "x"},
            {"PLAIN": "unprotected"},
        )
        for payload in malformed:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    deserialize_cfg_node_yaml(yaml.safe_dump(payload))
        with self.assertRaisesRegex(ValueError, "reserved"):
            serialize_cfg_node_yaml(CfgNode({BOT_CFG_TYPE_TAG: "user"}))

    def test_finalization_only_recovery_dry_run_is_safe_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, training_commit, finalization_commit, broken_bytes = (
                make_recovery_fixture(root)
            )
            records = output / "experiment_records" / "c2_l03_multi_granularity_part"
            log_path = output / "log.txt"
            checkpoint_path = output / "resnet50_checkpoint_22320.pt"
            log_before = log_path.read_bytes()
            checkpoint_before = checkpoint_path.read_bytes()
            log_sha256_before = sha256_file(log_path)
            checkpoint_sha256_before = sha256_file(checkpoint_path)

            first = recover_existing_run(
                output, records, output, fixture_root=output
            )
            repair_path = output / "config_repair_manifest.json"
            backup_path = output / "config_resolved.pre_type_fix.yml"
            repair_before = repair_path.read_bytes()
            backup_before = backup_path.read_bytes()
            runs_before = (records / "runs.csv").read_text(encoding="utf-8")

            self.assertEqual(backup_before, broken_bytes)
            self.assertEqual(log_path.read_bytes(), log_before)
            self.assertEqual(checkpoint_path.read_bytes(), checkpoint_before)
            self.assertEqual(sha256_file(log_path), log_sha256_before)
            self.assertEqual(
                sha256_file(checkpoint_path), checkpoint_sha256_before
            )
            self.assertEqual(first["metrics"]["selected_epoch"], 120)
            self.assertEqual(
                first["metrics"]["selected_global_iteration"], 22320
            )
            self.assertEqual(
                first["metrics"]["selected_checkpoint"],
                "resnet50_checkpoint_22320.pt",
            )
            self.assertEqual(first["metrics"]["rank1_percent"], 85.0)
            self.assertEqual(first["metrics"]["map_percent"], 75.0)
            manifest = recording.read_json(output / "run_manifest.json")
            self.assertEqual(manifest["training_commit"], training_commit)
            self.assertEqual(manifest["finalization_commit"], finalization_commit)
            self.assertNotEqual(training_commit, finalization_commit)
            self.assertEqual(manifest["finalization_mode"], "recover_existing_run")
            restored = deserialize_cfg_node_yaml(
                (output / "config_resolved.yml").read_text(encoding="utf-8")
            )
            self.assertTrue(validate_formal_protocol(restored, "recovered"))

            second = recover_existing_run(
                output, records, output, fixture_root=output
            )
            self.assertEqual(second["metrics"], first["metrics"])
            self.assertEqual(repair_path.read_bytes(), repair_before)
            self.assertEqual(backup_path.read_bytes(), backup_before)
            self.assertEqual(log_path.read_bytes(), log_before)
            self.assertEqual(checkpoint_path.read_bytes(), checkpoint_before)
            self.assertEqual(sha256_file(log_path), log_sha256_before)
            self.assertEqual(
                sha256_file(checkpoint_path), checkpoint_sha256_before
            )
            self.assertEqual(
                (records / "runs.csv").read_text(encoding="utf-8"),
                runs_before,
            )
            rows = list(csv.DictReader(runs_before.splitlines()))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["training_commit"], training_commit)
            self.assertEqual(rows[0]["finalization_commit"], finalization_commit)

    def test_recovery_reconciles_known_post_training_finalization_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _training, finalization_commit, _broken = (
                make_recovery_fixture(root)
            )
            mark_known_post_training_finalization_failure(output)
            records = (
                output / "experiment_records" /
                "c2_l03_multi_granularity_part"
            )

            result = recover_existing_run(
                output, records, output, fixture_root=output
            )

            manifest = recording.read_json(output / "run_manifest.json")
            status = recording.read_json(output / "run_status.json")
            repair = recording.read_json(output / "config_repair_manifest.json")
            self.assertEqual(result["metrics"]["selected_epoch"], 120)
            self.assertEqual(manifest["exit_code"], 0)
            self.assertEqual(status["exit_code"], 0)
            self.assertEqual(status["errors"], [])
            self.assertEqual(
                manifest["training_exit_code_reconciliation"]
                ["manifest_exit_code_before_recovery"],
                1,
            )
            self.assertEqual(
                repair["training_exit_code_reconciliation"]
                ["reconciled_training_exit_code"],
                0,
            )
            self.assertEqual(
                repair["recovery_attempt_commits"],
                [finalization_commit],
            )

    def test_recovery_continues_after_partial_ancestor_repair_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _training, repair_commit, _broken = make_recovery_fixture(root)
            mark_known_post_training_finalization_failure(output)
            records = (
                output / "experiment_records" /
                "c2_l03_multi_granularity_part"
            )
            with mock.patch.object(
                    recording, "finalize_run",
                    side_effect=EvidenceIncompleteError(
                        "simulated post-repair finalization failure"
                    )):
                with self.assertRaisesRegex(
                        EvidenceIncompleteError, "simulated post-repair"):
                    recover_existing_run(
                        output, records, output, fixture_root=output
                    )

            mark_known_post_training_finalization_failure(output)
            marker = output / "second_recovery_code_marker.txt"
            marker.write_text("second recovery code\n", encoding="utf-8")
            run_git(output, "add", marker.name)
            run_git(output, "commit", "-m", "second recovery commit")
            current_commit = subprocess.check_output(
                ["git", "-C", str(output), "rev-parse", "HEAD"],
                text=True,
            ).strip()

            result = recover_existing_run(
                output, records, output, fixture_root=output
            )

            manifest = recording.read_json(output / "run_manifest.json")
            repair = recording.read_json(output / "config_repair_manifest.json")
            self.assertEqual(result["metrics"]["selected_epoch"], 120)
            self.assertEqual(manifest["finalization_commit"], current_commit)
            self.assertEqual(repair["repair_code_commit"], repair_commit)
            self.assertEqual(
                repair["recovery_attempt_commits"],
                [repair_commit, current_commit],
            )

    def test_recovery_rejects_nonzero_exit_without_completion_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _training, _finalization, _broken = make_recovery_fixture(root)
            manifest_path = output / "run_manifest.json"
            status_path = output / "run_status.json"
            manifest = recording.read_json(manifest_path)
            status = recording.read_json(status_path)
            manifest["exit_code"] = 1
            status["exit_code"] = 1
            status["errors"] = [recording.KNOWN_CONFIG_FINALIZATION_ERROR]
            atomic_write_json(manifest_path, manifest)
            atomic_write_json(status_path, status)

            with self.assertRaisesRegex(
                    EvidenceIncompleteError,
                    "exactly one recorder completion record"):
                recover_existing_run(
                    output,
                    output / "experiment_records" /
                    "c2_l03_multi_granularity_part",
                    output,
                    fixture_root=output,
                )

    def test_recovery_rejects_nonzero_exit_without_known_finalization_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _training, _finalization, _broken = make_recovery_fixture(root)
            mark_known_post_training_finalization_failure(output)
            status_path = output / "run_status.json"
            status = recording.read_json(status_path)
            status["errors"] = ["unrelated failure"]
            atomic_write_json(status_path, status)

            with self.assertRaisesRegex(
                    EvidenceIncompleteError, "known post-training"):
                recover_existing_run(
                    output,
                    output / "experiment_records" /
                    "c2_l03_multi_granularity_part",
                    output,
                    fixture_root=output,
                )

    def test_finalization_only_recovery_failure_never_registers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _training, _finalization, _broken = make_recovery_fixture(root)
            records = output / "experiment_records" / "c2_l03_multi_granularity_part"
            (output / "resnet50_checkpoint_22320.pt").unlink()
            with self.assertRaises(EvidenceIncompleteError):
                recover_existing_run(
                    output, records, output, fixture_root=output
                )
            self.assertFalse((records / "runs.csv").exists())
            self.assertFalse((records / "evidence_manifest.tsv").exists())

    def test_recovery_cli_never_routes_to_formal_training(self):
        with mock.patch.object(runner, "_run_recovery", return_value=0) as recovery, \
                mock.patch.object(runner, "_run_formal") as formal:
            result = runner_main([
                "--recover-existing-run",
                "--output-dir", "synthetic-completed-output",
            ])
        self.assertEqual(result, 0)
        self.assertEqual(recovery.call_count, 1)
        formal.assert_not_called()

    def test_post_training_finalization_failure_keeps_training_exit_zero(self):
        error = EvidenceIncompleteError("synthetic finalization failure")
        self.assertEqual(runner._failure_exit_code(error, 0), 0)
        self.assertEqual(runner._failure_exit_code(error, 7), 7)
        self.assertEqual(runner._failure_exit_code(error, None), 1)

    def test_formal_preflight_accepts_exact_clean_branch(self):
        self.assertEqual(
            EXPECTED_BRANCH,
            "exp/c2-l03-multi-granularity-local-feature",
        )
        with tempfile.TemporaryDirectory() as directory:
            repo, config_path, launch, output = make_preflight_repository(Path(directory))
            result = run_preflight(repo, config_path, launch, output)
            self.assertEqual(result["branch"], EXPECTED_BRANCH)
            self.assertRegex(result["training_commit"], r"^[0-9a-f]{40}$")
            self.assertEqual(result["training_seed"], 42)

    def test_every_formal_protocol_field_is_locked_for_source_and_resolved(self):
        source = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(validate_formal_protocol(source, "source"))
        self.assertTrue(validate_formal_protocol(source, "resolved"))
        for dotted_path, expected in FORMAL_PROTOCOL.items():
            if isinstance(expected, bool):
                drift = not expected
            elif isinstance(expected, int):
                drift = expected + 1
            elif isinstance(expected, float):
                drift = expected + 0.1
            elif isinstance(expected, list):
                drift = list(reversed(expected))
            else:
                drift = str(expected) + "-drift"
            for label in ("source", "resolved"):
                with self.subTest(field=dotted_path, label=label):
                    broken = copy.deepcopy(source)
                    set_dotted(broken, dotted_path, drift)
                    with self.assertRaisesRegex(PreflightError, dotted_path):
                        validate_formal_protocol(broken, label)

    def test_formal_protocol_rejects_any_uncovered_training_leaf(self):
        source = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
        resolved_cfg = cfg.clone()
        resolved_cfg.merge_from_file(str(EXPERIMENT_CONFIG))
        self.assertTrue(validate_formal_protocol(resolved_cfg, "resolved"))
        source["SOLVER"]["UNTRACKED_TRAINING_FIELD"] = 1
        with self.assertRaisesRegex(PreflightError, "without a protocol"):
            validate_formal_protocol(source, "source")
        for required_field in (
                "MODEL.NAME", "MODEL.LAST_STRIDE", "MODEL.NECK",
                "MODEL.METRIC_LOSS_TYPE", "MODEL.IF_LABELSMOOTH",
                "MODEL.CAMERA_AWARE_TRIPLET", "INPUT.PROB",
                "INPUT.RE_PROB", "INPUT.PADDING", "SOLVER.MARGIN",
                "SOLVER.WEIGHT_DECAY", "SOLVER.GAMMA",
                "SOLVER.WARMUP_FACTOR", "SOLVER.WARMUP_ITERS",
                "SOLVER.WARMUP_METHOD"):
            self.assertIn(required_field, FORMAL_PROTOCOL)

    def test_custom_config_cannot_use_formal_experiment_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, config_path, launch, output = make_preflight_repository(root)
            custom = repo / "custom.yml"
            custom.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
            resolved = yaml.safe_load(custom.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(PreflightError, "reserved for config"):
                formal_preflight(
                    repo, custom, launch, output,
                    EXPECTED_EXPERIMENT_FAMILY, EXPECTED_RUN_ID,
                    EXPECTED_EVIDENCE_ID, resolved_config=resolved,
                )

    def test_formal_mode_rejects_checkpoint_resume(self):
        source = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
        source["MODEL"]["PRETRAIN_CHOICE"] = "self"
        with self.assertRaisesRegex(PreflightError, "PRETRAIN_CHOICE"):
            validate_formal_protocol(source, "source")

    def test_formal_preflight_rejects_wrong_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, config_path, launch, output = make_preflight_repository(
                Path(directory), branch="wrong-branch"
            )
            with self.assertRaisesRegex(PreflightError, "requires branch"):
                run_preflight(repo, config_path, launch, output)

    def test_formal_preflight_rejects_dirty_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, config_path, launch, output = make_preflight_repository(
                Path(directory), dirty=True
            )
            with self.assertRaisesRegex(PreflightError, "completely clean"):
                run_preflight(repo, config_path, launch, output)

    def test_formal_preflight_rejects_nonempty_output(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, config_path, launch, output = make_preflight_repository(
                Path(directory), nonempty_output=True
            )
            with self.assertRaisesRegex(PreflightError, "must not exist or must be empty"):
                run_preflight(repo, config_path, launch, output)

    def test_formal_preflight_rejects_registered_run_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, config_path, launch, output = make_preflight_repository(root)
            record_dir = root / "records"
            record_dir.mkdir()
            atomic_write_text(
                record_dir / "runs.csv",
                "run_id,evidence_id\n{},{}\n".format(
                    EXPECTED_RUN_ID, EXPECTED_EVIDENCE_ID
                ),
            )
            with self.assertRaisesRegex(PreflightError, "already exists"):
                run_preflight(
                    repo, config_path, launch, output, record_dir=record_dir
                )

    def test_formal_preflight_requires_explicit_seed_42(self):
        for include_seed, seed, message in (
                (False, 42, "SEED"),
                (True, 7, "SEED expected 42")):
            with self.subTest(include_seed=include_seed, seed=seed):
                with tempfile.TemporaryDirectory() as directory:
                    repo, config_path, launch, output = make_preflight_repository(
                        Path(directory), seed=seed, include_seed=include_seed
                    )
                    with self.assertRaisesRegex((ValueError, PreflightError), message):
                        run_preflight(repo, config_path, launch, output)

    def test_projection_dim_one_and_mgp_center_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than or equal to 2"):
            MultiGranularityPartHead(8, [2, 4, 6], projection_dim=1)
        local_cfg = cfg.clone()
        local_cfg.merge_from_file(str(EXPERIMENT_CONFIG))
        local_cfg.defrost()
        local_cfg.MODEL.PRETRAIN_CHOICE = "none"
        local_cfg.MODEL.PRETRAIN_PATH = ""
        local_cfg.MODEL.IF_WITH_CENTER = "yes"
        local_cfg.freeze()
        with self.assertRaisesRegex(ValueError, "center loss still assumes"):
            build_model(local_cfg, num_classes=2)

    def test_formal_source_config_explicitly_declares_seed_42(self):
        source = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
        self.assertIn("SEED", source)
        self.assertEqual(source["SEED"], 42)

    def test_dataset_manifest_hash_is_stable_under_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [
                (root / "b.jpg", 2, 1),
                (root / "a.jpg", 1, 0),
            ]
            first = build_dataset_manifest(
                {"train": entries, "query": entries[:1], "gallery": entries[1:]},
                root, "market1501", "softmax_triplet", 64, 4, 0, 1, 42,
            )
            second = build_dataset_manifest(
                {"train": list(reversed(entries)), "query": entries[:1], "gallery": entries[1:]},
                root, "market1501", "softmax_triplet", 64, 4, 0, 1, 42,
            )
            self.assertEqual(first["combined_manifest_sha256"], second["combined_manifest_sha256"])
            self.assertEqual(first["split_manifest_sha256"], second["split_manifest_sha256"])

    def test_environment_schema_and_seed_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {
                    "PYTHONHASHSEED": "42",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }):
                environment = collect_environment(directory)
            self.assertTrue(validate_environment_schema(environment))
            self.assertEqual(environment["pythonhashseed"], "42")
            self.assertEqual(environment["cublas_workspace_config"], ":4096:8")
            self.assertEqual(len(environment["pip_freeze_sha256"]), 64)

    def test_cpu_profiler_emits_required_metadata_without_gpu_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.json"
            completed = subprocess.run(
                [
                    sys.executable, str(PROFILE_TOOL), "--device", "cpu",
                    "--mode", "fixture", "--measurement-seed", "42",
                    "--batch-size", "4", "--input-height", "96",
                    "--input-width", "32", "--num-classes", "2",
                    "--dtype", "float32", "--warmup", "0",
                    "--measurement-repeats", "1", "--output-file", str(output),
                ],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["mode"], "fixture")
            self.assertEqual(payload["measurement_seed"], 42)
            self.assertIsInstance(payload["argv"], list)
            measurement = payload["measurement"]
            self.assertEqual(measurement["device"], "cpu")
            self.assertEqual(measurement["gpu_name"], NOT_APPLICABLE)
            self.assertEqual(
                measurement["worker_isolation"],
                recording.PROFILER_WORKER_ISOLATION,
            )
            self.assertEqual(
                measurement["operation_count_convention"],
                recording.OPERATION_COUNT_CONVENTION,
            )
            self.assertEqual(
                [(item["name"], item["variant"]) for item in payload["variants"]],
                [
                    ("C2-L03", "legacy"),
                    (
                        "C2-L03 + Multi-Granularity K={2,4,6}",
                        "multi_granularity",
                    ),
                ],
            )
            self.assertEqual(payload["variants"][1]["feature_dim"], 2816)
            for variant in payload["variants"]:
                self.assertGreater(variant["flops"], 0)
                self.assertGreater(variant["macs"], 0)
                self.assertGreater(variant["inference_latency_median_ms"], 0)

    def test_validation_selection_keeps_metrics_from_one_block(self):
        selected = select_best_validation(validation_records())
        self.assertEqual(selected["epoch"], 120)
        self.assertEqual(selected["global_iteration"], 22320)
        self.assertEqual(selected["rank5_percent"], 95.0)
        self.assertEqual(selected["map_percent"], 75.0)

    def test_checkpoint_binding_uses_verified_global_iterations(self):
        with tempfile.TemporaryDirectory() as directory:
            output, records, selected = make_checkpoint_binding_fixture(
                Path(directory)
            )
            rows, selected_row = recording.build_checkpoint_manifest(
                output, records, selected
            )
            self.assertEqual(
                [(row["epoch"], row["global_iteration"]) for row in rows],
                [(40, 7440), (80, 14880), (120, 22320)],
            )
            self.assertEqual(
                [row["global_iteration"] // row["epoch"] for row in rows],
                [186, 186, 186],
            )
            self.assertEqual(selected_row["epoch"], 120)
            self.assertEqual(selected_row["global_iteration"], 22320)
            self.assertEqual(
                selected_row["filename"], "resnet50_checkpoint_22320.pt"
            )
            self.assertEqual(selected_row["selected"], "true")
            manifest_rows = list(csv.DictReader(
                (output / "checkpoint_manifest.tsv").read_text(
                    encoding="utf-8"
                ).splitlines(),
                delimiter="\t",
            ))
            self.assertEqual(
                set((
                    "epoch", "global_iteration", "filename", "path",
                    "size_bytes", "sha256", "selected",
                )).difference(manifest_rows[0]),
                set(),
            )
            self.assertEqual(
                [row["selected"] for row in manifest_rows],
                ["false", "false", "true"],
            )

    def test_checkpoint_binding_rejects_missing_selected_global_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            output, records, selected = make_checkpoint_binding_fixture(
                Path(directory), checkpoint_iterations=[7440, 14880]
            )
            with self.assertRaisesRegex(
                    EvidenceIncompleteError,
                    "epoch 120 maps to global iteration 22320.*found 0"):
                recording._checkpoint_manifest_rows(output, records, selected)

    def test_checkpoint_binding_rejects_duplicate_iteration_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            output, records, selected = make_checkpoint_binding_fixture(
                Path(directory)
            )
            (output / "resnet50_checkpoint_022320.pt").write_bytes(b"duplicate\n")
            with self.assertRaisesRegex(
                    EvidenceIncompleteError,
                    "epoch 120 maps to global iteration 22320.*found 2"):
                recording._checkpoint_manifest_rows(output, records, selected)

    def test_checkpoint_binding_rejects_inconsistent_epoch_iteration_ratios(self):
        records = validation_records()
        records[-1] = dict(records[-1], global_iteration=22319)
        with tempfile.TemporaryDirectory() as directory:
            output, records, selected = make_checkpoint_binding_fixture(
                Path(directory), records=records
            )
            with self.assertRaisesRegex(
                    EvidenceIncompleteError, "exact positive ratio|inconsistent"):
                recording._checkpoint_manifest_rows(output, records, selected)

    def test_checkpoint_binding_rejects_validation_log_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            output, records, selected = make_checkpoint_binding_fixture(
                Path(directory), log_iterations_per_epoch=185
            )
            with self.assertRaisesRegex(
                    EvidenceIncompleteError,
                    "Conflicting iterations_per_epoch evidence.*"
                    "validation_history.jsonl=186.*log.txt=185"):
                recording._checkpoint_manifest_rows(output, records, selected)

    def test_checkpoint_binding_ignores_inconsistent_progress_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            output, records, selected = make_checkpoint_binding_fixture(
                Path(directory), log_iterations_per_epoch=None
            )
            atomic_write_text(
                output / "log.txt",
                "Epoch[9] Iteration[180/183] fixture\n"
                "Epoch[9] Iteration[20/184] fixture\n",
            )

            self.assertEqual(
                recording._log_iteration_evidence(output),
                (None, 0),
            )
            rows, selected_row = recording.build_checkpoint_manifest(
                output, records, selected
            )

            self.assertEqual(
                [(row["epoch"], row["global_iteration"]) for row in rows],
                [(40, 7440), (80, 14880), (120, 22320)],
            )
            self.assertEqual(selected_row["global_iteration"], 22320)

    def test_checkpoint_binding_rejects_malformed_checkpoint_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            output, records, selected = make_checkpoint_binding_fixture(
                Path(directory)
            )
            (output / "resnet50_checkpoint_epoch120.pt").write_bytes(b"bad\n")
            with self.assertRaisesRegex(
                    EvidenceIncompleteError, "Checkpoint filename does not match"):
                recording._checkpoint_manifest_rows(output, records, selected)

    def test_checkpoint_binding_never_treats_epoch_suffix_as_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            output, records, selected = make_checkpoint_binding_fixture(
                Path(directory), checkpoint_iterations=[7440, 14880, 120]
            )
            with self.assertRaisesRegex(
                    EvidenceIncompleteError,
                    "epoch 120 maps to global iteration 22320.*found 0"):
                recording._checkpoint_manifest_rows(output, records, selected)

    def test_success_finalization_binds_checkpoint_hash_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = make_success_fixture(root)
            record_dir = root / "records"
            result = finalize_run(output, record_dir)
            metrics = result["metrics"]
            self.assertEqual(metrics["selected_epoch"], 120)
            self.assertEqual(metrics["training_runtime_seconds"], 12.5)
            self.assertEqual(metrics["total_run_runtime_seconds"], 20.0)
            finalized_manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(finalized_manifest["profiling_runtime_seconds"], 5.0)
            self.assertNotEqual(
                metrics["training_runtime_seconds"],
                finalized_manifest["profiling_runtime_seconds"],
            )
            self.assertEqual(metrics["training_seed"], 42)
            self.assertEqual(metrics["status"], TRAINING_COMPLETE)
            self.assertEqual(metrics["evidence_status"], LOCAL_EVIDENCE_PENDING)
            selected = output / metrics["selected_checkpoint"]
            self.assertEqual(metrics["selected_checkpoint_sha256"], sha256_file(selected))
            rows = list(csv.DictReader(
                (output / "checkpoint_manifest.tsv").read_text(encoding="utf-8").splitlines(),
                delimiter="\t"))
            selected_rows = [row for row in rows if row["selected"] == "true"]
            self.assertEqual(len(selected_rows), 1)
            self.assertEqual(selected_rows[0]["epoch"], "120")
            self.assertEqual(selected_rows[0]["global_iteration"], "22320")
            self.assertEqual(
                selected_rows[0]["filename"],
                "resnet50_checkpoint_22320.pt",
            )
            self.assertEqual(
                result["metrics"]["selected_global_iteration"], 22320
            )
            self.assertEqual(
                result["metrics"]["selected_checkpoint"],
                "resnet50_checkpoint_22320.pt",
            )
            run_row = result["run_row"]
            self.assertEqual(run_row["training_runtime_seconds"], 12.5)
            self.assertEqual(run_row["total_run_runtime_seconds"], 20.0)
            self.assertEqual(run_row["archive_status"], NOT_ARCHIVED)

    def test_finalization_runtime_covers_each_phase_one_boundary(self):
        delayed_helpers = (
            "validate_efficiency_profile",
            "build_checkpoint_manifest",
            "_build_metrics_summary",
            "_write_artifact_hashes",
            "_stage_registry_transaction",
        )
        for helper_name in delayed_helpers:
            with self.subTest(helper=helper_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = make_success_fixture(
                        root, finalization_complete=False
                    )
                    clock = FakeMonotonicClock()
                    original = getattr(recording, helper_name)

                    def delayed(*args, _original=original, **kwargs):
                        clock.advance(3.0)
                        return _original(*args, **kwargs)

                    with mock.patch.object(
                            recording.time, "monotonic", side_effect=clock), \
                            mock.patch.object(
                                recording, helper_name, side_effect=delayed
                            ):
                        result = finalize_run(output, root / "records")
                    manifest = result["manifest"]
                    self.assertEqual(manifest["finalization_runtime_seconds"], 3.0)
                    self.assertEqual(manifest["total_run_runtime_seconds"], 23.0)
                    self.assertEqual(
                        manifest["finalization_timing_boundary"],
                        recording.FINALIZATION_TIMING_BOUNDARY,
                    )

    def test_idempotent_finalize_does_not_accumulate_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = make_success_fixture(root, finalization_complete=False)
            clock = FakeMonotonicClock()
            original = recording._build_metrics_summary

            def delayed(*args, **kwargs):
                clock.advance(2.0)
                return original(*args, **kwargs)

            with mock.patch.object(
                    recording.time, "monotonic", side_effect=clock), \
                    mock.patch.object(
                        recording, "_build_metrics_summary", side_effect=delayed
                    ):
                first = finalize_run(output, root / "records")
                second = finalize_run(output, root / "records")
            self.assertEqual(first["manifest"]["finalization_runtime_seconds"], 2.0)
            self.assertEqual(second["manifest"]["finalization_runtime_seconds"], 2.0)
            self.assertEqual(first["manifest"]["total_run_runtime_seconds"], 22.0)
            self.assertEqual(second["manifest"]["total_run_runtime_seconds"], 22.0)

    def test_run_formal_total_starts_before_config_and_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "formal_output"
            local_cfg = cfg.clone()
            local_cfg.merge_from_file(str(EXPERIMENT_CONFIG))
            local_cfg.defrost()
            local_cfg.OUTPUT_DIR = str(output)
            local_cfg.freeze()
            args = SimpleNamespace(
                config=str(runner.DEFAULT_CONFIG),
                experiment_family=EXPECTED_EXPERIMENT_FAMILY,
                run_id=EXPECTED_RUN_ID,
                evidence_id=EXPECTED_EVIDENCE_ID,
            )
            clock = FakeMonotonicClock()
            captured_timing = {}

            def load_with_delay(_path):
                clock.advance(2.0)
                return local_cfg

            def preflight_with_delay(*_args, **_kwargs):
                clock.advance(3.0)
                return {
                    "branch": EXPECTED_BRANCH,
                    "training_commit": "a" * 40,
                    "dirty": False,
                    "source_config_path": str(EXPERIMENT_CONFIG),
                    "source_config_sha256": sha256_file(EXPERIMENT_CONFIG),
                    "launch_script_path": str(
                        REPO_ROOT / "scripts" /
                        "train_c2_l03_multi_granularity_part_autodl.sh"
                    ),
                    "launch_script_sha256": "b" * 64,
                    "pretrained_weight_path": str(root / "unused.pth"),
                    "pretrained_weight_sha256": "c" * 64,
                    "data_root": str(root / "synthetic_data"),
                    "output_dir": str(output),
                    "training_seed": 42,
                }

            subprocess_calls = {"count": 0}

            def subprocess_with_delay(*_args, **_kwargs):
                subprocess_calls["count"] += 1
                clock.advance(5.0 if subprocess_calls["count"] == 1 else 11.0)
                return SimpleNamespace(returncode=0)

            def capture_finish(_output, runtimes, _exit_code):
                captured_timing.update(runtimes)

            environment = {
                "hostname": "synthetic-host",
                "gpu_count": 1,
                "pythonhashseed": "42",
                "cublas_workspace_config": ":4096:8",
            }
            with mock.patch.object(
                    runner.time, "monotonic", side_effect=clock), \
                    mock.patch.object(
                        runner, "_load_config", side_effect=load_with_delay
                    ), \
                    mock.patch.object(
                        runner, "formal_preflight", side_effect=preflight_with_delay
                    ), \
                    mock.patch.object(runner, "initialize_run"), \
                    mock.patch.object(
                        runner, "collect_environment", return_value=environment
                    ), \
                    mock.patch(
                        "data.build.collect_dataset_protocol", return_value=({}, 2)
                    ), \
                    mock.patch.object(runner, "_model_manifest", return_value={}), \
                    mock.patch.object(
                        runner.subprocess, "run", side_effect=subprocess_with_delay
                    ), \
                    mock.patch.object(
                        runner, "finish_run_timing", side_effect=capture_finish
                    ), \
                    mock.patch.object(runner, "_append_run_log"), \
                    mock.patch.object(
                        runner, "finalize_run", return_value={"metrics": {}}
                    ) as finalize_mock, \
                    mock.patch.dict(os.environ, {}, clear=False):
                self.assertEqual(runner._run_formal(args), 0)
            self.assertEqual(captured_timing["profiling_runtime_seconds"], 5.0)
            self.assertEqual(captured_timing["training_runtime_seconds"], 11.0)
            self.assertEqual(captured_timing["total_run_runtime_seconds"], 21.0)
            self.assertEqual(
                finalize_mock.call_args.kwargs["total_runtime_started"], 100.0
            )
            self.assertEqual(finalize_mock.call_count, 1)

    def test_run_formal_profiler_failure_prevents_training_and_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "formal_output"
            local_cfg = cfg.clone()
            local_cfg.merge_from_file(str(EXPERIMENT_CONFIG))
            local_cfg.defrost()
            local_cfg.OUTPUT_DIR = str(output)
            local_cfg.freeze()
            args = SimpleNamespace(
                config=str(runner.DEFAULT_CONFIG),
                experiment_family=EXPECTED_EXPERIMENT_FAMILY,
                run_id=EXPECTED_RUN_ID,
                evidence_id=EXPECTED_EVIDENCE_ID,
            )
            preflight = {
                "branch": EXPECTED_BRANCH,
                "training_commit": "a" * 40,
                "dirty": False,
                "source_config_path": str(EXPERIMENT_CONFIG),
                "source_config_sha256": sha256_file(EXPERIMENT_CONFIG),
                "launch_script_path": str(
                    REPO_ROOT / "scripts" /
                    "train_c2_l03_multi_granularity_part_autodl.sh"
                ),
                "launch_script_sha256": "b" * 64,
                "pretrained_weight_path": str(root / "unused.pth"),
                "pretrained_weight_sha256": "c" * 64,
                "data_root": str(root / "synthetic_data"),
                "output_dir": str(output),
                "training_seed": 42,
            }
            environment = {
                "hostname": "synthetic-host",
                "gpu_count": 1,
                "pythonhashseed": "42",
                "cublas_workspace_config": ":4096:8",
            }
            with mock.patch.object(runner, "_load_config", return_value=local_cfg), \
                    mock.patch.object(
                        runner, "formal_preflight", return_value=preflight
                    ), \
                    mock.patch.object(runner, "initialize_run"), \
                    mock.patch.object(
                        runner, "collect_environment", return_value=environment
                    ), \
                    mock.patch(
                        "data.build.collect_dataset_protocol", return_value=({}, 2)
                    ), \
                    mock.patch.object(runner, "_model_manifest", return_value={}), \
                    mock.patch.object(
                        runner.subprocess, "run",
                        return_value=SimpleNamespace(returncode=9),
                    ) as subprocess_mock, \
                    mock.patch.object(runner, "finish_run_timing"), \
                    mock.patch.object(runner, "_append_run_log"), \
                    mock.patch.object(runner, "finalize_run") as finalize_mock, \
                    mock.patch.dict(os.environ, {}, clear=False):
                with self.assertRaisesRegex(
                        RuntimeError, "Mandatory formal efficiency profiler failed"):
                    runner._run_formal(args)
            self.assertEqual(subprocess_mock.call_count, 1)
            finalize_mock.assert_not_called()

    def test_formal_runner_closes_every_post_initialize_failure(self):
        cases = (
            "resolved_config_write",
            "environment_exception",
            "profiler_failure",
            "training_nonzero",
            "finalization_exception",
            "keyboard_interrupt",
            "system_exit",
        )
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output, local_cfg, args, preflight, environment = (
                        formal_runner_fixture(root)
                    )
                    subprocess_results = [SimpleNamespace(returncode=0)]
                    if case == "training_nonzero":
                        subprocess_results.append(SimpleNamespace(returncode=7))
                    else:
                        subprocess_results.append(SimpleNamespace(returncode=0))
                    collect_side_effect = None
                    collect_return = environment
                    if case == "environment_exception":
                        collect_side_effect = RuntimeError("environment failed")
                    elif case == "keyboard_interrupt":
                        collect_side_effect = KeyboardInterrupt()
                    elif case == "system_exit":
                        collect_side_effect = SystemExit(17)

                    lock_factory = lambda repo, run_output, run_id: (
                        recording.formal_run_lock(
                            repo, run_output, run_id, lock_root=root / "locks"
                        )
                    )
                    patches = [
                        mock.patch.object(
                            runner, "_load_config", return_value=local_cfg
                        ),
                        mock.patch.object(
                            runner, "formal_preflight", return_value=preflight
                        ),
                        mock.patch.object(
                            runner, "formal_run_lock", side_effect=lock_factory
                        ),
                        mock.patch.object(
                            runner, "collect_environment",
                            return_value=collect_return,
                            side_effect=collect_side_effect,
                        ),
                        mock.patch(
                            "data.build.collect_dataset_protocol",
                            return_value=({}, 2),
                        ),
                        mock.patch.object(
                            runner, "_model_manifest", return_value={}
                        ),
                        mock.patch.object(
                            runner.subprocess, "run",
                            side_effect=subprocess_results,
                        ),
                        mock.patch.object(
                            runner, "RECORD_DIR", root / "formal_records"
                        ),
                        mock.patch.dict(os.environ, {}, clear=False),
                    ]
                    if case == "resolved_config_write":
                        patches.append(mock.patch.object(
                            runner, "atomic_write_text",
                            side_effect=OSError("resolved config write failed"),
                        ))
                    if case == "profiler_failure":
                        patches[-3] = mock.patch.object(
                            runner.subprocess, "run",
                            return_value=SimpleNamespace(returncode=9),
                        )
                    if case == "finalization_exception":
                        patches.append(mock.patch.object(
                            runner, "finalize_run",
                            side_effect=RuntimeError("finalization failed"),
                        ))
                    else:
                        patches.append(mock.patch.object(runner, "finalize_run"))

                    entered = []
                    try:
                        for patcher in patches:
                            entered.append(patcher.start())
                        if case == "keyboard_interrupt":
                            with self.assertRaises(KeyboardInterrupt):
                                runner._run_formal(args)
                        elif case == "system_exit":
                            with self.assertRaises(SystemExit) as caught:
                                runner._run_formal(args)
                            self.assertEqual(caught.exception.code, 17)
                        else:
                            with self.assertRaises((RuntimeError, OSError)):
                                runner._run_formal(args)
                    finally:
                        for patcher in reversed(patches):
                            patcher.stop()

                    status = json.loads(
                        (output / "run_status.json").read_text(encoding="utf-8")
                    )
                    manifest = json.loads(
                        (output / "run_manifest.json").read_text(encoding="utf-8")
                    )
                    self.assertIn(status["status"], ("failed", "incomplete"))
                    self.assertNotEqual(status["status"], "running")
                    self.assertNotEqual(manifest["status"], "running")
                    for field in (
                            "total_run_runtime_seconds",
                            "environment_collection_runtime_seconds",
                            "profiling_runtime_seconds",
                            "training_runtime_seconds",
                            "finalization_runtime_seconds"):
                        self.assertNotEqual(status[field], recording.NOT_RECORDED)
                        self.assertNotEqual(manifest[field], recording.NOT_RECORDED)
                    finalize_mock = entered[-1]
                    if case in (
                            "resolved_config_write", "environment_exception",
                            "profiler_failure", "training_nonzero",
                            "keyboard_interrupt", "system_exit"):
                        finalize_mock.assert_not_called()
                    self.assertFalse(any((root / "locks").glob("*.lock")))

    def test_formal_run_lock_is_cross_process_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_root = root / "locks"
            output = root / "output"
            lock = recording.formal_run_lock(
                root, output, EXPECTED_RUN_ID, lock_root=lock_root
            )
            child_code = (
                "import sys; "
                "from utils.experiment_recording import formal_run_lock, "
                "FormalRunLockError; "
                "repo,out,run_id,lock_root=sys.argv[1:5]; "
                "lock=formal_run_lock(repo,out,run_id,lock_root=lock_root); "
                "\ntry:\n lock.acquire(); print('acquired'); lock.release()"
                "\nexcept FormalRunLockError as error:\n print(error); sys.exit(3)"
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(REPO_ROOT)
            command = [
                sys.executable, "-B", "-c", child_code,
                str(root), str(output), EXPECTED_RUN_ID, str(lock_root),
            ]
            lock.acquire()
            try:
                blocked = subprocess.run(
                    command, cwd=str(REPO_ROOT), env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                )
                self.assertEqual(blocked.returncode, 3)
                diagnostics = blocked.stdout.decode("utf-8", errors="replace")
                for field in ("pid", "hostname", "acquired_at_utc", "owner="):
                    self.assertIn(field, diagnostics)
            finally:
                self.assertTrue(lock.release())
            acquired = subprocess.run(
                command, cwd=str(REPO_ROOT), env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(
                acquired.returncode, 0,
                acquired.stderr.decode("utf-8", errors="replace"),
            )
            self.assertIn("acquired", acquired.stdout.decode("utf-8"))
            self.assertFalse(any(lock_root.glob("*.lock")))

    def test_formal_run_lock_stale_and_foreign_owner_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = recording.formal_run_lock(
                root, root / "output", EXPECTED_RUN_ID,
                lock_root=root / "locks",
            )
            stale = {
                "token": "stale-owner",
                "pid": 999999,
                "hostname": "stale-host",
                "acquired_at_utc": "2020-01-01T00:00:00Z",
            }
            lock.path.write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaisesRegex(
                    recording.FormalRunLockError, "stale-owner"):
                lock.acquire()
            self.assertTrue(lock.path.is_file())
            lock.path.unlink()

            owner = recording.formal_run_lock(
                root, root / "output", EXPECTED_RUN_ID,
                lock_root=root / "locks",
            ).acquire()
            foreign = dict(owner.owner)
            foreign["token"] = "replacement-owner"
            owner.path.write_text(json.dumps(foreign), encoding="utf-8")
            self.assertFalse(owner.release())
            self.assertTrue(owner.path.is_file())
            owner.path.unlink()

    def test_same_run_id_finalization_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = make_success_fixture(root)
            record_dir = root / "records"
            finalize_run(output, record_dir)
            finalize_run(output, record_dir)
            rows = list(csv.DictReader(
                (record_dir / "runs.csv").read_text(encoding="utf-8").splitlines()
            ))
            self.assertEqual(len(rows), 1)
            evidence = list(csv.DictReader(
                (record_dir / "evidence_manifest.tsv").read_text(
                    encoding="utf-8"
                ).splitlines(), delimiter="\t",
            ))
            self.assertEqual({row["run_id"] for row in evidence}, {EXPECTED_RUN_ID})
            self.assertEqual(
                len([row for row in evidence if row["artifact_type"] == "artifact_hashes"]),
                1,
            )
            original_runs = (record_dir / "runs.csv").read_bytes()
            atomic_write_text(output / "log.txt", "tampered synthetic log\n")
            with self.assertRaisesRegex(EvidenceIncompleteError, "conflict"):
                finalize_run(output, record_dir)
            self.assertEqual((record_dir / "runs.csv").read_bytes(), original_runs)

    def test_seed_missing_conflict_checkpoint_missing_and_failure_never_register(self):
        cases = (
            {"source_seed": False},
            {"metadata_seed": 7},
            {"selected_checkpoint": False},
            {"exit_code": 1},
        )
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = make_success_fixture(root, **case)
                    record_dir = root / "records"
                    with self.assertRaises(EvidenceIncompleteError):
                        finalize_run(output, record_dir)
                    runs = record_dir / "runs.csv"
                    self.assertFalse(runs.exists())
                    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
                    expected_status = "failed" if case.get("exit_code") == 1 else "incomplete"
                    self.assertEqual(status["status"], expected_status)

    def test_seed_finalizer_rejects_worker_and_generator_mutations(self):
        def mutate_seed_applied(reproducibility, _dataset):
            reproducibility["seed_applied_before_data_loading"] = False

        def mutate_worker_enabled(reproducibility, _dataset):
            reproducibility["data_loader_worker_seeding"]["enabled"] = False

        def mutate_worker_scheme(reproducibility, _dataset):
            reproducibility["data_loader_worker_seeding"]["scheme"] = "tampered"

        def mutate_worker_count(reproducibility, _dataset):
            reproducibility["data_loader_worker_seeding"]["num_workers"] = 7

        def mutate_both_generator_copies(reproducibility, dataset):
            reproducibility["data_loader_generators"]["stream_seeds"]["train"] = 7
            dataset["data_loader_generators"]["stream_seeds"]["train"] = 7

        mutations = (
            mutate_seed_applied,
            mutate_worker_enabled,
            mutate_worker_scheme,
            mutate_worker_count,
            mutate_both_generator_copies,
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = make_success_fixture(root)
                    reproducibility_path = output / "reproducibility.json"
                    dataset_path = output / "dataset_manifest.json"
                    reproducibility = json.loads(
                        reproducibility_path.read_text(encoding="utf-8")
                    )
                    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
                    mutation(reproducibility, dataset)
                    atomic_write_json(reproducibility_path, reproducibility)
                    atomic_write_json(dataset_path, dataset)
                    record_dir = root / "records"
                    with self.assertRaises(EvidenceIncompleteError):
                        finalize_run(output, record_dir)
                    self.assertFalse((record_dir / "runs.csv").exists())

    def test_formal_seed_evidence_rejects_missing_drift_and_wrong_types(self):
        fields = {
            "seed": (43, "42"),
            "python_random_seed": (43, "42"),
            "python_random_seeded": (False, 1),
            "numpy_seed": (43, "42"),
            "numpy_seeded": (False, 1),
            "torch_cpu_seed": (43, "42"),
            "torch_cpu_seeded": (False, 1),
            "cuda_available": (False, 1),
            "torch_cuda_manual_seed_all_called": (False, 1),
            "torch_cuda_manual_seed_all_seed": (43, "42"),
            "torch_cuda_all_seeded": (False, 1),
            "cudnn_deterministic": (False, 1),
            "cudnn_benchmark": (True, 0),
            "pythonhashseed": ("43", 42),
            "cublas_workspace_config": (":16:8", 4096),
        }
        for field, (drift, wrong_type) in fields.items():
            for action, value in (
                    ("missing", None), ("drift", drift),
                    ("wrong_type", wrong_type)):
                with self.subTest(field=field, action=action):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        output = make_success_fixture(
                            root, execution_mode="formal"
                        )
                        path = output / "reproducibility.json"
                        evidence = json.loads(path.read_text(encoding="utf-8"))
                        if action == "missing":
                            evidence["random_state"].pop(field)
                        else:
                            evidence["random_state"][field] = value
                        atomic_write_json(path, evidence)
                        record_dir = root / "records"
                        with self.assertRaises(EvidenceIncompleteError):
                            finalize_run(output, record_dir)
                        self.assertFalse((record_dir / "runs.csv").exists())

    def test_formal_seed_cross_evidence_rejects_mutations(self):
        mutations = {
            "top_seed": lambda reproducibility, _manifest, _environment: (
                reproducibility.update(seed=43)
            ),
            "seed_chain": lambda reproducibility, _manifest, _environment: (
                reproducibility["seed_chain"].update(applied_training_seed=43)
            ),
            "manifest_seed": lambda _reproducibility, manifest, _environment: (
                manifest.update(training_seed=43)
            ),
            "environment_gpu_count": lambda _reproducibility, _manifest, environment: (
                environment.update(gpu_count=0)
            ),
            "environment_pythonhashseed_type": lambda _reproducibility, _manifest, environment: (
                environment.update(pythonhashseed=42)
            ),
            "environment_cublas": lambda _reproducibility, _manifest, environment: (
                environment.update(cublas_workspace_config=":16:8")
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = make_success_fixture(root, execution_mode="formal")
                    paths = {
                        "reproducibility": output / "reproducibility.json",
                        "manifest": output / "run_manifest.json",
                        "environment": output / "environment.json",
                    }
                    payloads = {
                        name: json.loads(path.read_text(encoding="utf-8"))
                        for name, path in paths.items()
                    }
                    mutation(
                        payloads["reproducibility"], payloads["manifest"],
                        payloads["environment"],
                    )
                    for payload_name, path in paths.items():
                        atomic_write_json(path, payloads[payload_name])
                    record_dir = root / "records"
                    with self.assertRaises(EvidenceIncompleteError):
                        finalize_run(output, record_dir)
                    self.assertFalse((record_dir / "runs.csv").exists())

    def test_checkpoint_integrity_rejects_all_structural_mutations(self):
        def write_text(path):
            path.write_text("not a checkpoint\n", encoding="utf-8")

        def write_empty_file(path):
            path.write_bytes(b"")

        def write_empty_state(path):
            torch.save({}, path)

        def write_optimizer_only(path):
            torch.save({"state": {}, "param_groups": []}, path)

        def write_missing_key(path):
            state = synthetic_model_state_dict()
            state.pop("base.weight")
            torch.save(state, path)

        def write_extra_key(path):
            state = synthetic_model_state_dict()
            state["unexpected.weight"] = torch.zeros(1)
            torch.save(state, path)

        def write_wrong_shape(path):
            state = synthetic_model_state_dict()
            state["base.weight"] = torch.zeros(3, 3)
            torch.save(state, path)

        def write_wrong_dtype(path):
            state = synthetic_model_state_dict()
            state["base.weight"] = state["base.weight"].to(torch.float64)
            torch.save(state, path)

        mutations = {
            "text": write_text,
            "empty_file": write_empty_file,
            "empty_state_dict": write_empty_state,
            "optimizer_only": write_optimizer_only,
            "missing_key": write_missing_key,
            "extra_key": write_extra_key,
            "wrong_shape": write_wrong_shape,
            "wrong_dtype": write_wrong_dtype,
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = make_success_fixture(root)
                    selected = output / "resnet50_checkpoint_22320.pt"
                    mutation(selected)
                    record_dir = root / "records"
                    with self.assertRaises(EvidenceIncompleteError):
                        finalize_run(output, record_dir)
                    self.assertFalse((record_dir / "runs.csv").exists())

    def test_valid_synthetic_state_dict_checkpoint_finalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = make_success_fixture(root)
            selected = output / "resnet50_checkpoint_22320.pt"
            loaded = torch.load(selected, map_location="cpu", weights_only=True)
            self.assertEqual(
                recording.model_state_dict_schema(loaded),
                json.loads(
                    (output / "model_manifest.json").read_text(encoding="utf-8")
                )["state_dict_schema"],
            )
            result = finalize_run(output, root / "records")
            self.assertEqual(result["run_row"]["status"], TRAINING_COMPLETE)

    def test_missing_or_invalid_efficiency_profile_never_finalizes(self):
        mutations = {
            "missing": None,
            "not_recorded": lambda profile: profile.update(status="not_recorded"),
            "wrong_seed": lambda profile: profile.update(measurement_seed=7),
            "wrong_batch": lambda profile: profile["measurement"].update(batch_size=32),
            "not_recorded_numeric": lambda profile: profile["variants"][0].update(
                flops="not_recorded"
            ),
            "missing_field": lambda profile: profile.pop("variants"),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = make_success_fixture(root)
                    profile_path = output / "efficiency_profile.json"
                    if mutation is None:
                        profile_path.unlink()
                    else:
                        profile = json.loads(profile_path.read_text(encoding="utf-8"))
                        mutation(profile)
                        atomic_write_json(profile_path, profile)
                    record_dir = root / "records"
                    with self.assertRaises(EvidenceIncompleteError):
                        finalize_run(output, record_dir)
                    self.assertFalse((record_dir / "runs.csv").exists())

    def test_complete_formal_efficiency_fixture_finalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = make_success_fixture(root, execution_mode="formal")
            result = finalize_run(output, root / "records")
            self.assertEqual(result["run_row"]["status"], TRAINING_COMPLETE)

    def test_formal_efficiency_allows_cuda_driver_reserved_gpu_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = make_success_fixture(root, execution_mode="formal")
            profile_path = output / "efficiency_profile.json"
            profile = recording.read_json(profile_path)
            profile["measurement"]["gpu_total_memory_mib"] = (
                24576.0 - 481.125
            )
            atomic_write_json(profile_path, profile)

            result = finalize_run(output, root / "records")

            self.assertEqual(result["run_row"]["status"], TRAINING_COMPLETE)

    def test_formal_efficiency_mutations_never_finalize(self):
        def set_argv(profile, option, value):
            index = profile["argv"].index(option)
            profile["argv"][index + 1] = value
            profile["display_command"] = shlex.join(
                [profile["python_executable"]] + profile["argv"]
            )

        def append_argv(profile, option, value):
            profile["argv"].extend([option, value])
            profile["display_command"] = shlex.join(
                [profile["python_executable"]] + profile["argv"]
            )

        mutations = {
            "warmup": lambda profile: profile["measurement"].update(warmup=0),
            "repeats": lambda profile: profile["measurement"].update(
                measurement_repeats=1
            ),
            "schema": lambda profile: profile.update(schema_version=1),
            "profiler_sha256": lambda profile: profile.update(
                profiler_script_sha256="0" * 64
            ),
            "variant_name": lambda profile: profile["variants"][0].update(
                name="tampered baseline"
            ),
            "variant_order": lambda profile: profile["variants"].reverse(),
            "worker_isolation": lambda profile: profile["measurement"].update(
                worker_isolation="shared process"
            ),
            "operation_convention": lambda profile: profile["measurement"].update(
                operation_count_convention="estimated"
            ),
            "argv": lambda profile: set_argv(profile, "--batch-size", "32"),
            "argv_unknown_option": lambda profile: append_argv(
                profile, "--undeclared-formal-option", "1"
            ),
            "display_command": lambda profile: profile.update(
                display_command=profile["display_command"] + " --tampered"
            ),
            "flops_macs": lambda profile: profile["variants"][0].update(
                flops=profile["variants"][0]["flops"] + 1
            ),
            "throughput": lambda profile: profile["variants"][0].update(
                throughput_images_per_second=(
                    profile["variants"][0]["throughput_images_per_second"] + 1
                )
            ),
            "delta": lambda profile: profile["deltas"]["macs"].update(
                absolute=profile["deltas"]["macs"]["absolute"] + 1
            ),
            "gpu_environment": lambda profile: profile["measurement"].update(
                gpu_name="Tampered GPU"
            ),
            "driver_environment": lambda profile: profile["measurement"].update(
                nvidia_driver="tampered-driver"
            ),
            "pytorch_environment": lambda profile: profile["measurement"].update(
                pytorch_version="tampered-pytorch"
            ),
            "cuda_environment": lambda profile: profile["measurement"].update(
                cuda_runtime="tampered-cuda"
            ),
            "cudnn_environment": lambda profile: profile["measurement"].update(
                cudnn_version="tampered-cudnn"
            ),
            "gpu_memory_environment": lambda profile: profile["measurement"].update(
                gpu_total_memory_mib=(
                    profile["measurement"]["gpu_total_memory_mib"] + 1.0
                )
            ),
            "gpu_memory_reserved_gap": lambda profile: profile["measurement"].update(
                gpu_total_memory_mib=(
                    profile["measurement"]["gpu_total_memory_mib"] - 600.0
                )
            ),
            "model_descriptor": lambda profile: profile["variants"][1].update(
                feature_dim=2817
            ),
            "model_total_parameters": lambda profile: profile["variants"][1].update(
                total_parameters=(
                    profile["variants"][1]["total_parameters"] + 1
                )
            ),
            "model_trainable_parameters": lambda profile: profile["variants"][1].update(
                trainable_parameters=(
                    profile["variants"][1]["trainable_parameters"] + 1
                )
            ),
            "formal_cpu": lambda profile: profile["measurement"].update(
                device="cpu"
            ),
        }
        for delta_name in recording.REQUIRED_EFFICIENCY_DELTA_FIELDS:
            mutations["delta_{}".format(delta_name)] = (
                lambda profile, name=delta_name: profile["deltas"][name].update(
                    absolute=profile["deltas"][name]["absolute"] + 1
                )
            )
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = make_success_fixture(
                        root, execution_mode="formal"
                    )
                    profile_path = output / "efficiency_profile.json"
                    profile = json.loads(profile_path.read_text(encoding="utf-8"))
                    mutation(profile)
                    atomic_write_json(profile_path, profile)
                    record_dir = root / "records"
                    with self.assertRaises(EvidenceIncompleteError):
                        finalize_run(output, record_dir)
                    self.assertFalse((record_dir / "runs.csv").exists())

    def test_registry_fault_injection_never_leaves_success_row(self):
        cases = (
            "artifact_hashes.tsv",
            "evidence_manifest.tsv",
            "missing_evidence.md",
            "runs.csv",
        )
        for failed_target in cases:
            with self.subTest(failed_target=failed_target):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = make_success_fixture(root)
                    record_dir = root / "records"
                    if failed_target == "artifact_hashes.tsv":
                        patcher = mock.patch.object(
                            recording, "_write_artifact_hashes",
                            side_effect=OSError("injected artifact hash failure"),
                        )
                    else:
                        original_commit = recording._commit_registry_file

                        def injected_commit(staged, target, name=failed_target):
                            if Path(target).name == name:
                                raise OSError("injected {} failure".format(name))
                            return original_commit(staged, target)

                        patcher = mock.patch.object(
                            recording, "_commit_registry_file",
                            side_effect=injected_commit,
                        )
                    with patcher:
                        with self.assertRaises(EvidenceIncompleteError):
                            finalize_run(output, record_dir)
                    runs_path = record_dir / "runs.csv"
                    if runs_path.is_file():
                        rows = list(csv.DictReader(
                            runs_path.read_text(encoding="utf-8").splitlines()
                        ))
                        self.assertFalse(any(
                            row.get("run_id") == EXPECTED_RUN_ID
                            and row.get("status") == TRAINING_COMPLETE
                            for row in rows
                        ))
                        self.assertFalse(any(
                            row.get("run_id") == EXPECTED_RUN_ID
                            and row.get("evidence_status") == LOCAL_EVIDENCE_PENDING
                            for row in rows
                        ))
                    status = json.loads(
                        (output / "run_status.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(status["status"], "incomplete")

    def test_dry_run_fixture_uses_temp_and_updates_once(self):
        with tempfile.TemporaryDirectory() as directory:
            output = make_success_fixture(Path(directory))
            self.assertEqual(
                runner_main(["--dry-run", "--fixture-dir", str(output)]), 0
            )
            self.assertEqual(
                runner_main(["--dry-run", "--fixture-dir", str(output)]), 0
            )
            runs = output / "experiment_records" / "c2_l03_multi_granularity_part" / "runs.csv"
            rows = list(csv.DictReader(runs.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(len(rows), 1)

    def test_dry_run_rejects_parent_and_symlink_path_escapes(self):
        def create_junction(link, target):
            completed = subprocess.run(
                ["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(
                completed.returncode, 0,
                (completed.stdout + completed.stderr).decode(
                    "utf-8", errors="replace"
                ),
            )

        cases = (
            "source_parent",
            "launch_parent",
            "source_symlink",
            "launch_symlink",
            "checkpoint_symlink",
            "registry_symlink",
        )
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = make_success_fixture(root)
                    manifest_path = output / "run_manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    source = Path(manifest["source_config"]["path"])
                    launch = Path(manifest["launch_script"]["path"])
                    if case == "source_parent":
                        outside = root / "outside_source.yml"
                        outside.write_bytes(source.read_bytes())
                        manifest["source_config"]["path"] = str(outside)
                        manifest["source_config"]["sha256"] = sha256_file(outside)
                        atomic_write_json(manifest_path, manifest)
                    elif case == "launch_parent":
                        outside = root / "outside_launch.sh"
                        outside.write_bytes(launch.read_bytes())
                        manifest["launch_script"]["path"] = str(outside)
                        manifest["launch_script"]["sha256"] = sha256_file(outside)
                        atomic_write_json(manifest_path, manifest)
                    elif case in ("source_symlink", "launch_symlink"):
                        inside = source if case == "source_symlink" else launch
                        outside_dir = root / "outside_{}_dir".format(case)
                        outside_dir.mkdir()
                        outside = outside_dir / inside.name
                        outside.write_bytes(inside.read_bytes())
                        junction = output / "{}_junction".format(case)
                        create_junction(junction, outside_dir)
                        manifest_key = (
                            "source_config" if case == "source_symlink"
                            else "launch_script"
                        )
                        manifest[manifest_key]["path"] = str(
                            junction / inside.name
                        )
                        manifest[manifest_key]["sha256"] = sha256_file(outside)
                        atomic_write_json(manifest_path, manifest)
                    elif case == "checkpoint_symlink":
                        inside = output / "resnet50_checkpoint_22320.pt"
                        inside.unlink()
                        outside = root / "outside_checkpoint_dir"
                        outside.mkdir()
                        torch.save(
                            synthetic_model_state_dict(), outside / "model.pth"
                        )
                        create_junction(inside, outside)
                    elif case == "registry_symlink":
                        outside = root / "outside_registry"
                        outside.mkdir()
                        create_junction(output / "experiment_records", outside)
                    self.assertEqual(
                        runner_main(["--dry-run", "--fixture-dir", str(output)]),
                        1,
                    )
                    self.assertFalse(
                        (output / "experiment_records" /
                         "c2_l03_multi_granularity_part" / "runs.csv").is_file()
                    )

    def test_forbidden_path_guard_and_utf8_lf_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for forbidden in (
                    root / "paper_notes" / "x.md",
                    root / "thesis_evidence" / "x.json",
                    root / "Downloads" / "x.txt",
                    root / "EXPERIMENTS.md",
                    root / "scripts" / "append_experiment_result.py"):
                with self.subTest(forbidden=forbidden):
                    with self.assertRaisesRegex(ValueError, "Forbidden"):
                        assert_path_allowed(forbidden)
            output = root / "utf8.txt"
            atomic_write_text(output, "合成记录\r\n第二行\r\n")
            raw = output.read_bytes()
            self.assertIn("合成记录".encode("utf-8"), raw)
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(raw.count(b"\n"), 2)


if __name__ == "__main__":
    unittest.main()
