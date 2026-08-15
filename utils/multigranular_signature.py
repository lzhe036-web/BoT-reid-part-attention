# encoding: utf-8
"""Stable signatures for the shared global/local feature pipeline."""

from __future__ import absolute_import

import hashlib
import inspect
import json

import torch


NOT_RECORDED = "not_recorded"


def _source_sha256(callable_object):
    source = inspect.getsource(callable_object)
    normalized = source.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _feature_implementation_hashes():
    from layers.part_correspondence_consistency import (
        build_cross_camera_positive_pairs,
        build_local_part_descriptors,
        horizontal_part_bounds,
        pairwise_local_distance_matrix,
    )
    from modeling.baseline import Baseline, PartAttentionHead

    callables = {
        "baseline_forward": Baseline.forward,
        "part_attention_forward": PartAttentionHead.forward,
        "horizontal_part_bounds": horizontal_part_bounds,
        "build_local_part_descriptors": build_local_part_descriptors,
        "build_cross_camera_positive_pairs": (
            build_cross_camera_positive_pairs
        ),
        "pairwise_local_distance_matrix": pairwise_local_distance_matrix,
    }
    return {
        name: _source_sha256(callable_object)
        for name, callable_object in sorted(callables.items())
    }


def _nested(mapping, dotted_path, default=NOT_RECORDED):
    current = mapping
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _feature_parameter_schema(configuration):
    """Build deterministic parameter names/shapes without loading weights."""
    from modeling.baseline import Baseline

    model_name = str(_nested(configuration, "MODEL.NAME"))
    part_attention = bool(_nested(configuration, "MODEL.PART_ATTENTION", False))
    part_attention_parts = int(_nested(
        configuration, "MODEL.PART_ATTENTION_PARTS", 6
    ))
    pcc_enabled = bool(_nested(
        configuration, "MODEL.PART_CORRESPONDENCE_CONSISTENCY", False
    ))
    pcc_parts = int(_nested(configuration, "MODEL.PCC_PARTS", 6))
    pcc_mode = str(_nested(configuration, "MODEL.PCC_MODE", "fixed_index"))
    pcc_tau = _nested(configuration, "MODEL.PCC_SOFTMIN_TAU", 0.1)
    neck = str(_nested(configuration, "MODEL.NECK", "bnneck"))
    neck_feat = str(_nested(configuration, "TEST.NECK_FEAT", "after"))
    last_stride = int(_nested(configuration, "MODEL.LAST_STRIDE", 1))
    with torch.random.fork_rng(devices=[]):
        model = Baseline(
            1,
            last_stride,
            "",
            neck,
            neck_feat,
            model_name,
            "none",
            part_attention=part_attention,
            part_attention_parts=part_attention_parts,
            part_correspondence_consistency=pcc_enabled,
            pcc_parts=pcc_parts,
            pcc_mode=pcc_mode,
            pcc_softmin_tau=pcc_tau,
        )
    prefixes = ("base.", "part_attention_head.", "bottleneck.")
    return [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    ], int(model.in_planes)


def multigranular_feature_payload(configuration):
    """Return the mode-independent canonical feature definition."""
    parameters, descriptor_dim = _feature_parameter_schema(configuration)
    parts = int(_nested(configuration, "MODEL.PCC_PARTS", 6))
    attention_parts = int(_nested(
        configuration, "MODEL.PART_ATTENTION_PARTS", parts
    ))
    return {
        "signature_schema_version": 1,
        "backbone": {
            "name": str(_nested(configuration, "MODEL.NAME")),
            "last_stride": int(_nested(
                configuration, "MODEL.LAST_STRIDE", 1
            )),
            "shared_forward": "Baseline.base(x) exactly once",
        },
        "input": {
            "size_train": list(_nested(configuration, "INPUT.SIZE_TRAIN", [])),
            "size_test": list(_nested(configuration, "INPUT.SIZE_TEST", [])),
        },
        "part_attention": {
            "enabled": bool(_nested(
                configuration, "MODEL.PART_ATTENTION", False
            )),
            "parts": attention_parts,
            "part_order": "top_to_bottom",
            "fusion": "global_average_feature_plus_attention_part_feature",
        },
        "global_descriptor": {
            "source": "shared_backbone_feature_map",
            "pooling": "torch.nn.AdaptiveAvgPool2d(output_size=1)",
            "shape": ["B", descriptor_dim],
        },
        "local_descriptor": {
            "builder": "layers.part_correspondence_consistency.build_local_part_descriptors",
            "source": "same_shared_backbone_feature_map",
            "partition": "horizontal_part_bounds_integer_nonoverlap_full_height",
            "part_order": "top_to_bottom",
            "parts": parts,
            "pooling": "torch.nn.functional.adaptive_avg_pool2d(output_size=1)",
            "normalization": "none",
            "shape": ["B", parts, descriptor_dim],
        },
        "pair_selection": {
            "builder": "build_cross_camera_positive_pairs",
            "rule": "same_pid_and_different_camera_and_i_less_than_j",
        },
        "local_distance": {
            "builder": "pairwise_local_distance_matrix",
            "metric": "raw_euclidean_l2",
            "shape": ["N_pairs", parts, parts],
        },
        "inference_descriptor": {
            "neck": str(_nested(configuration, "MODEL.NECK", "bnneck")),
            "neck_feature": str(_nested(
                configuration, "TEST.NECK_FEAT", "after"
            )),
            "shape": ["B", descriptor_dim],
        },
        "feature_parameter_schema": parameters,
        "feature_implementation_sha256": _feature_implementation_hashes(),
    }


def canonical_multigranular_feature_signature(configuration):
    payload = multigranular_feature_payload(configuration)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return canonical, digest
