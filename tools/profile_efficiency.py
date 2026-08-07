#!/usr/bin/env python
# encoding: utf-8
"""Measure model efficiency without modifying or training the model."""

from __future__ import absolute_import

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from tools.analyze_distance_distributions import build_checkpoint_model
from utils.experiment_recording import (
    NOT_RECORDED,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from utils.reproducibility import seed_analysis_process, validate_seed


def _forward_model(model, sample, camids=None):
    if camids is None:
        return model(sample)
    return model(sample, camids=camids)


def _profile_camids(model, batch_size, device):
    if not getattr(model, "camera_conditional_part_attention", False):
        return None
    return torch.zeros(int(batch_size), dtype=torch.long, device=device)


def _profile_macs(model, sample, camids=None):
    totals = {"macs": 0}
    handles = []

    def conv_hook(module, _inputs, output):
        output_elements = int(output.numel())
        kernel = int(module.kernel_size[0] * module.kernel_size[1])
        totals["macs"] += output_elements * kernel * int(
            module.in_channels // module.groups
        )

    def linear_hook(module, _inputs, output):
        output_rows = int(output.numel() // module.out_features)
        totals["macs"] += output_rows * int(module.in_features * module.out_features)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            _forward_model(model, sample, camids=camids)
    finally:
        for handle in handles:
            handle.remove()
    return int(totals["macs"])


def profile(config_file, checkpoint, output, seed, repeats=20):
    seed = validate_seed(seed)
    seed_analysis_process(seed)
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(config_file))
    local_cfg.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_checkpoint_model(local_cfg, checkpoint, device)
    height, width = (int(value) for value in local_cfg.INPUT.SIZE_TRAIN)
    sample = torch.zeros(1, 3, height, width, device=device)
    sample_camids = _profile_camids(model, sample.size(0), device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    macs = _profile_macs(model, sample, camids=sample_camids)
    flops = 2 * macs
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_forward = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_forward = NOT_RECORDED
    for _ in range(5):
        with torch.no_grad():
            _forward_model(model, sample, camids=sample_camids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        with torch.no_grad():
            _forward_model(model, sample, camids=sample_camids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)
    latency = statistics.median(timings)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "measurement_time": utc_now(),
        "measurement_seed": seed,
        "device": str(device),
        "GPU": torch.cuda.get_device_name(device) if device.type == "cuda" else NOT_RECORDED,
        "input_shape": [1, 3, height, width],
        "inference_uses_camid": sample_camids is not None,
        "profile_camid": 0 if sample_camids is not None else NOT_RECORDED,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "MACs": int(macs),
        "FLOPs": int(flops),
        "operation_count_rule": "Conv2d/Linear MACs; FLOPs = 2 * MACs",
        "peak_forward_memory": peak_forward,
        "peak_train_memory": NOT_RECORDED,
        "inference_latency": float(latency),
        "inference_latency_unit": "ms_per_image",
        "throughput": float(1000.0 / latency),
        "throughput_unit": "images_per_second",
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    atomic_write_json(output, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Profile checkpoint efficiency")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        profile(
            args.config_file, args.checkpoint, args.output, args.seed,
            repeats=args.repeats,
        )
    except BaseException as error:
        print("Efficiency profiling failed: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
