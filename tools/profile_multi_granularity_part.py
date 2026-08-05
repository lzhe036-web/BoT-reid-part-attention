#!/usr/bin/env python
"""Profile C2-L03 and C2-MGP-K246 without loading pretrained weights."""

from __future__ import absolute_import

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from modeling import build_model
from utils.config_serialization import deserialize_cfg_node_yaml
from utils.reproducibility import seed_everything, validate_seed


FORMAL_CONFIG = (
    REPO_ROOT / "configs" /
    "softmax_triplet_c2_l03_multi_granularity_part_autodl.yml"
)
NOT_APPLICABLE = "not_applicable"
SCHEMA_VERSION = 2
WORKER_ISOLATION = "each variant measured in an independent subprocess"
OPERATION_COUNT_CONVENTION = "Conv2d/Linear MACs; FLOPs=2×MACs"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Profile C2-L03 vs C2-L03 + multi-granularity K={2,4,6}"
    )
    parser.add_argument("--config", default=str(FORMAL_CONFIG))
    parser.add_argument("--resolved-config")
    parser.add_argument("--source-config-sha256")
    parser.add_argument("--resolved-config-sha256")
    parser.add_argument("--mode", choices=("formal", "fixture"), default="fixture")
    parser.add_argument("--measurement-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=751)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measurement-repeats", type=int, default=20)
    parser.add_argument("--output-file")
    parser.add_argument(
        "--worker-variant", choices=("legacy", "multi_granularity"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("{}.tmp.{}".format(target.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(str(temporary), str(target))


def profiling_config_from_source(path, legacy_baseline=False):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(path))
    local_cfg.defrost()
    # Profiling is intentionally random-initialized. Formal pretrained weights
    # are never opened by this process.
    local_cfg.MODEL.PRETRAIN_CHOICE = "none"
    local_cfg.MODEL.PRETRAIN_PATH = ""
    if legacy_baseline:
        local_cfg.MODEL.PART_ATTENTION = True
        local_cfg.MODEL.PART_ATTENTION_PARTS = 6
        local_cfg.MODEL.MULTI_GRANULARITY_PART = False
    local_cfg.freeze()
    return local_cfg


def declared_output_dir(path):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(path))
    return Path(str(local_cfg.OUTPUT_DIR)).resolve()


def declared_resolved_output_dir(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        resolved = deserialize_cfg_node_yaml(handle.read())
    if "OUTPUT_DIR" not in resolved:
        raise ValueError("Resolved configuration is missing OUTPUT_DIR")
    return Path(str(resolved["OUTPUT_DIR"])).resolve()


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return int(total), int(trainable)


def count_macs(model, device, dtype, height, width):
    """Count Conv2d/Linear multiply-accumulates for one descriptor forward."""
    counts = []

    def hook(module, _inputs, output):
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            per_output = (module.in_channels // module.groups) * kernel
            counts.append(int(output.numel() * per_output))
        elif isinstance(module, nn.Linear):
            counts.append(int(output.numel() * module.in_features))

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    model.eval()
    sample = torch.zeros(1, 3, height, width, device=device, dtype=dtype)
    try:
        with torch.inference_mode():
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
        del sample
    macs = int(sum(counts))
    return macs, int(2 * macs)


def percentile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Latency measurements are empty")
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_profile(model, inputs, device, warmup, repeats):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(repeats):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(inputs)
                end.record()
                torch.cuda.synchronize(device)
                timings.append(float(start.elapsed_time(end)))
            else:
                started = time.perf_counter()
                model(inputs)
                timings.append((time.perf_counter() - started) * 1000.0)
    median = float(statistics.median(timings))
    p95 = float(percentile(timings, 0.95))
    throughput = float(inputs.shape[0] / (median / 1000.0))
    return median, p95, throughput


def mib(value):
    return float(value / (1024.0 ** 2))


def cuda_memory_profile(model, local_cfg, inputs, device, num_classes):
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    model.eval()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model(inputs)
    torch.cuda.synchronize(device)
    forward = {
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved(device)),
    }

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    score, features = model(inputs)
    instances_per_identity = int(local_cfg.DATALOADER.NUM_INSTANCE)
    if inputs.shape[0] % instances_per_identity != 0:
        raise ValueError("Batch size must be divisible by NUM_INSTANCE")
    identities = inputs.shape[0] // instances_per_identity
    if identities < 2 or identities > num_classes:
        raise ValueError("Synthetic identity count is invalid")
    targets = torch.arange(identities, device=device).repeat_interleave(
        instances_per_identity
    )
    camids = torch.arange(inputs.shape[0], device=device).remainder(
        instances_per_identity
    )
    from layers import make_loss
    with contextlib.redirect_stdout(io.StringIO()):
        loss_function = make_loss(local_cfg, num_classes)
    loss = loss_function(score, features, targets, camids)["loss_total"]
    loss.backward()
    torch.cuda.synchronize(device)
    forward_backward = {
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved(device)),
    }
    return forward, forward_backward


def select_device(args):
    cuda_available = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is unavailable")
    use_cuda = cuda_available if args.device == "auto" else args.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.mode == "formal" and device.type != "cuda":
        raise RuntimeError("Formal profiling requires CUDA")
    return device


def profile_variant(args):
    legacy = args.worker_variant == "legacy"
    seed_everything(args.measurement_seed)
    device = select_device(args)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU profiling supports float32 only")
    local_cfg = profiling_config_from_source(
        args.config, legacy_baseline=legacy
    )
    model = build_model(local_cfg, num_classes=args.num_classes).to(
        device=device, dtype=dtype
    )
    total, trainable = count_parameters(model)
    macs, flops = count_macs(
        model, device, dtype, args.input_height, args.input_width
    )
    inputs = torch.randn(
        args.batch_size, 3, args.input_height, args.input_width,
        device=device, dtype=dtype,
    )
    median, p95, throughput = latency_profile(
        model, inputs, device, args.warmup, args.measurement_repeats
    )
    if device.type == "cuda":
        forward, forward_backward = cuda_memory_profile(
            model, local_cfg, inputs, device, args.num_classes
        )
    else:
        forward = {
            "peak_allocated_mib": NOT_APPLICABLE,
            "peak_reserved_mib": NOT_APPLICABLE,
        }
        forward_backward = dict(forward)
    result = {
        "name": "C2-L03" if legacy else "C2-L03 + Multi-Granularity K={2,4,6}",
        "variant": args.worker_variant,
        "measurement_seed": args.measurement_seed,
        "config_variant": (
            "legacy PART_ATTENTION=True,K=6,MGP=False"
            if legacy else "MGP=True,scales=[2,4,6]"
        ),
        "feature_dim": int(model.feature_dim),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "macs": macs,
        "flops": flops,
        "inference_latency_median_ms": median,
        "inference_latency_p95_ms": p95,
        "throughput_images_per_second": throughput,
        "forward_peak_memory": forward,
        "forward_backward_peak_memory": forward_backward,
    }
    del inputs
    del model
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return result


def worker_command(args, variant, output):
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--config", str(Path(args.config).resolve()),
        "--mode", args.mode,
        "--measurement-seed", str(args.measurement_seed),
        "--batch-size", str(args.batch_size),
        "--input-height", str(args.input_height),
        "--input-width", str(args.input_width),
        "--num-classes", str(args.num_classes),
        "--device", args.device,
        "--dtype", args.dtype,
        "--warmup", str(args.warmup),
        "--measurement-repeats", str(args.measurement_repeats),
        "--worker-variant", variant,
        "--worker-output", str(output),
    ]
    return command


def run_variant_workers(args):
    variants = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for variant in ("legacy", "multi_granularity"):
            output = root / "{}.json".format(variant)
            command = worker_command(args, variant, output)
            completed = subprocess.run(
                command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Profiler worker {} failed: {}".format(
                        variant,
                        completed.stderr.decode("utf-8", errors="replace"),
                    )
                )
            variants.append(json.loads(output.read_text(encoding="utf-8")))
    return variants


def delta_pair(baseline, experiment):
    if baseline in (NOT_APPLICABLE, None) or experiment in (NOT_APPLICABLE, None):
        return {"absolute": NOT_APPLICABLE, "percent": NOT_APPLICABLE}
    baseline = float(baseline)
    experiment = float(experiment)
    if baseline == 0:
        raise ValueError("Cannot compute percentage delta from zero baseline")
    return {
        "absolute": experiment - baseline,
        "percent": 100.0 * (experiment - baseline) / baseline,
    }


def build_deltas(baseline, experiment):
    values = {
        "total_parameters": (baseline["total_parameters"], experiment["total_parameters"]),
        "trainable_parameters": (
            baseline["trainable_parameters"], experiment["trainable_parameters"]
        ),
        "descriptor_dim": (baseline["feature_dim"], experiment["feature_dim"]),
        "flops": (baseline["flops"], experiment["flops"]),
        "macs": (baseline["macs"], experiment["macs"]),
        "latency_median_ms": (
            baseline["inference_latency_median_ms"],
            experiment["inference_latency_median_ms"],
        ),
        "latency_p95_ms": (
            baseline["inference_latency_p95_ms"],
            experiment["inference_latency_p95_ms"],
        ),
        "throughput_images_per_second": (
            baseline["throughput_images_per_second"],
            experiment["throughput_images_per_second"],
        ),
    }
    for prefix, key in (
            ("memory_forward", "forward_peak_memory"),
            ("memory_forward_backward", "forward_backward_peak_memory")):
        for memory_name in ("peak_allocated_mib", "peak_reserved_mib"):
            values["{}_{}".format(prefix, memory_name)] = (
                baseline[key][memory_name], experiment[key][memory_name]
            )
    return {name: delta_pair(*pair) for name, pair in values.items()}


def driver_version():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return NOT_APPLICABLE
    lines = output.decode("utf-8", errors="replace").splitlines()
    return lines[0].strip() if lines else NOT_APPLICABLE


def validate_arguments(args):
    validate_seed(args.measurement_seed)
    if args.measurement_seed != 42:
        raise ValueError("C2-MGP-K246 profiling seed must be 42")
    if min(args.batch_size, args.input_height, args.input_width, args.num_classes) <= 0:
        raise ValueError("Batch size, input size, and num_classes must be positive")
    if args.warmup < 0 or args.measurement_repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")
    if args.mode == "formal":
        expected = (args.batch_size, args.input_height, args.input_width, args.dtype)
        if expected != (64, 256, 128, "float32"):
            raise ValueError("Formal profiling protocol must be batch=64,size=256x128,float32")
        if args.device != "cuda":
            raise ValueError("Formal profiling must explicitly request --device cuda")
        if args.warmup != 5 or args.measurement_repeats != 20:
            raise ValueError("Formal profiling requires warmup=5 and repeats=20")
        if Path(args.config).resolve() != FORMAL_CONFIG.resolve():
            raise ValueError("Formal profiling requires the fixed source config")
        if not args.worker_variant:
            if not all((
                    args.resolved_config, args.source_config_sha256,
                    args.resolved_config_sha256, args.output_file)):
                raise ValueError(
                    "Formal profiling requires resolved config, both config hashes, "
                    "and an output file"
                )
            resolved_path = Path(args.resolved_config).resolve()
            output_path = Path(args.output_file).resolve()
            source_output = declared_output_dir(args.config)
            resolved_output = declared_resolved_output_dir(resolved_path)
            if (resolved_path != source_output / "config_resolved.yml"
                    or resolved_output != source_output
                    or output_path != source_output / "efficiency_profile.json"):
                raise ValueError(
                    "Formal profiler resolved/output paths must match the source and "
                    "resolved config OUTPUT_DIR"
                )


def main(argv=None):
    args = parse_args(argv)
    validate_arguments(args)
    if args.worker_variant:
        if not args.worker_output:
            raise ValueError("Profiler worker requires --worker-output")
        atomic_json(args.worker_output, profile_variant(args))
        return 0

    source_path = Path(args.config).resolve()
    resolved_path = Path(args.resolved_config or args.config).resolve()
    source_hash = file_sha256(source_path)
    resolved_hash = file_sha256(resolved_path)
    if args.source_config_sha256 and args.source_config_sha256 != source_hash:
        raise ValueError("Provided source config SHA256 does not match")
    if args.resolved_config_sha256 and args.resolved_config_sha256 != resolved_hash:
        raise ValueError("Provided resolved config SHA256 does not match")

    variants = run_variant_workers(args)
    device = select_device(args)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = float(
            torch.cuda.get_device_properties(device).total_memory / (1024.0 ** 2)
        )
        driver = driver_version()
    else:
        gpu_name = NOT_APPLICABLE
        gpu_memory = NOT_APPLICABLE
        driver = NOT_APPLICABLE
    recorded_argv = list(
        sys.argv if argv is None
        else [str(Path(__file__).resolve())] + list(argv)
    )
    recorded_argv[0] = str(Path(__file__).resolve())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "mode": args.mode,
        "measurement_timestamp_utc": (
            dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z")
        ),
        "measurement_seed": args.measurement_seed,
        "seed_source": "explicit --measurement-seed",
        "python_executable": sys.executable,
        "argv": recorded_argv,
        "display_command": shlex.join([sys.executable] + recorded_argv),
        "profiler_script_sha256": file_sha256(Path(__file__).resolve()),
        "source_config_sha256": source_hash,
        "resolved_config_sha256": resolved_hash,
        "config_overrides": {
            "MODEL.PRETRAIN_CHOICE": "none",
            "MODEL.PRETRAIN_PATH": "",
        },
        "measurement": {
            "num_classes": args.num_classes,
            "batch_size": args.batch_size,
            "input_size": [args.input_height, args.input_width],
            "dtype": args.dtype,
            "device": str(device),
            "gpu_name": gpu_name,
            "gpu_total_memory_mib": gpu_memory,
            "nvidia_driver": driver,
            "pytorch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda or NOT_APPLICABLE,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.version() is not None else NOT_APPLICABLE
            ),
            "warmup": args.warmup,
            "measurement_repeats": args.measurement_repeats,
            "operation_count_convention": OPERATION_COUNT_CONVENTION,
            "worker_isolation": WORKER_ISOLATION,
        },
        "variants": variants,
        "deltas": build_deltas(variants[0], variants[1]),
    }
    if args.output_file:
        atomic_json(args.output_file, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
