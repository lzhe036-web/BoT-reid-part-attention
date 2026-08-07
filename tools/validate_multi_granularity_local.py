#!/usr/bin/env python
# encoding: utf-8
"""Synthetic-only validation for the C2-L03 multi-granularity model."""

from __future__ import absolute_import

import argparse
import gc
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from layers.triplet_loss import (
    CrossCameraPositiveLoss,
    CrossEntropyLabelSmooth,
    TripletLoss,
)
from modeling.baseline import Baseline
from utils.experiment_recording import NOT_RECORDED, atomic_write_json


def build_model(local_cfg, num_classes, enabled):
    return Baseline(
        num_classes=num_classes,
        last_stride=local_cfg.MODEL.LAST_STRIDE,
        model_path="",
        neck=local_cfg.MODEL.NECK,
        neck_feat=local_cfg.TEST.NECK_FEAT,
        model_name=local_cfg.MODEL.NAME,
        pretrain_choice="none",
        part_attention=local_cfg.MODEL.PART_ATTENTION,
        part_attention_parts=local_cfg.MODEL.PART_ATTENTION_PARTS,
        multi_granularity_local=enabled,
        multi_granularity_scales=local_cfg.MODEL.MULTI_GRANULARITY_SCALES,
        multi_granularity_dim=local_cfg.MODEL.MULTI_GRANULARITY_DIM,
        multi_granularity_aggregation=(
            local_cfg.MODEL.MULTI_GRANULARITY_AGGREGATION
        ),
    )


def c2_loss(scores, descriptor, labels, camids, num_classes, device):
    id_value = CrossEntropyLabelSmooth(
        num_classes=num_classes, use_gpu=device.type == "cuda"
    )(scores, labels)
    triplet_value = TripletLoss(margin=0.3)(descriptor, labels)[0]
    cross_value = CrossCameraPositiveLoss(mode="mean")(
        descriptor, labels, camids
    )
    return id_value + triplet_value + 0.3 * cross_value


def parameter_counts(model):
    return {
        "total": int(sum(item.numel() for item in model.parameters())),
        "trainable": int(sum(
            item.numel() for item in model.parameters() if item.requires_grad
        )),
    }


def measure_cuda_memory(local_cfg, enabled, batch_size, backward):
    if not torch.cuda.is_available():
        return {"status": NOT_RECORDED, "reason": "CUDA unavailable"}
    device = torch.device("cuda")
    model = None
    inputs = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = build_model(local_cfg, num_classes=751, enabled=enabled)
        model.to(device)
        model.train(mode=backward)
        height, width = [int(value) for value in local_cfg.INPUT.SIZE_TRAIN]
        inputs = torch.randn(batch_size, 3, height, width, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        if backward:
            scores, descriptor = model(inputs)
            if batch_size >= 8 and batch_size % 4 == 0:
                identities = batch_size // 4
                labels = torch.arange(
                    identities, device=device
                ).repeat_interleave(4)
                camids = torch.arange(batch_size, device=device) % 2
            else:
                labels = torch.arange(batch_size, device=device) % 2
                camids = (torch.arange(batch_size, device=device) // 2) % 2
            loss = c2_loss(scores, descriptor, labels, camids, 751, device)
            loss.backward()
        else:
            model.eval()
            with torch.no_grad():
                model(inputs)
        torch.cuda.synchronize(device)
        return {
            "status": "complete",
            "batch_size": int(batch_size),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        return {
            "status": "missing_evidence",
            "batch_size": int(batch_size),
            "reason": "CUDA out of memory: {}".format(error),
        }
    finally:
        del inputs
        del model
        gc.collect()
        torch.cuda.empty_cache()


def validate(config_file, memory_batch_size):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(config_file))
    local_cfg.freeze()
    baseline = build_model(local_cfg, num_classes=751, enabled=False)
    enhanced = build_model(local_cfg, num_classes=751, enabled=True)
    baseline_counts = parameter_counts(baseline)
    enhanced_counts = parameter_counts(enhanced)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enhanced.to(device).eval()
    height, width = [int(value) for value in local_cfg.INPUT.SIZE_TRAIN]
    with torch.no_grad():
        inference, trace = enhanced.forward_with_shape_trace(
            torch.randn(1, 3, height, width, device=device)
        )
    trace["inference_feature"] = tuple(inference.shape)

    enhanced.train()
    small_inputs = torch.randn(4, 3, 96, 32, device=device)
    scores, descriptor = enhanced(small_inputs)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    camids = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    loss = c2_loss(scores, descriptor, labels, camids, 751, device)
    loss.backward()
    backward_complete = any(
        parameter.grad is not None
        for parameter in enhanced.multi_granularity_head.parameters()
    )
    enhanced.cpu()
    del small_inputs, scores, descriptor, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "validation_type": "synthetic_random_initialization_no_checkpoint",
        "baseline_commit": "dca6dc1dd890d47dbbbaf192de14c9ab5402afb0",
        "baseline_experiment": "C2-L03",
        "baseline_existing_attention": True,
        "new_module_attention": False,
        "scales": [2, 4, 6],
        "aggregation": "mean",
        "projection_dim": 256,
        "descriptor_dim": int(enhanced.descriptor_dim),
        "shape_trace": trace,
        "synthetic_backward_complete": bool(backward_complete),
        "baseline_parameters": baseline_counts,
        "enhanced_parameters": enhanced_counts,
        "parameter_increase": {
            "total": enhanced_counts["total"] - baseline_counts["total"],
            "trainable": (
                enhanced_counts["trainable"] - baseline_counts["trainable"]
            ),
        },
    }
    del baseline, enhanced
    gc.collect()
    result["cuda_memory"] = {
        "batch_size": int(memory_batch_size),
        "baseline_forward": measure_cuda_memory(
            local_cfg, False, memory_batch_size, backward=False
        ),
        "enhanced_forward": measure_cuda_memory(
            local_cfg, True, memory_batch_size, backward=False
        ),
        "baseline_forward_backward": measure_cuda_memory(
            local_cfg, False, memory_batch_size, backward=True
        ),
        "enhanced_forward_backward": measure_cuda_memory(
            local_cfg, True, memory_batch_size, backward=True
        ),
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        default=str(
            REPO_ROOT / "configs" /
            "softmax_triplet_c2l03_multi_granularity_local_feature_autodl.yml"
        ),
    )
    parser.add_argument("--memory-batch-size", type=int, default=64)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = validate(args.config_file, args.memory_batch_size)
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
