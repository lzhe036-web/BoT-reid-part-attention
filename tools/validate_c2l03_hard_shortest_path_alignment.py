#!/usr/bin/env python
# encoding: utf-8
"""Synthetic fail-closed validation for hard shortest-path part alignment."""

from __future__ import absolute_import

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from layers import make_loss
from layers.part_correspondence_consistency import (
    build_cross_camera_positive_pairs,
    fixed_index_distances,
    hard_shortest_path_alignment_loss,
    hard_shortest_path_costs,
    pairwise_local_distance_matrix,
    select_pair_local_features,
)
from layers.triplet_loss import CrossCameraPositiveLoss, TripletLoss
from modeling.baseline import Baseline
from utils.experiment_recording import NOT_APPLICABLE, experiment_identity


FIXED_CONFIG = (
    REPO_ROOT / "configs" /
    "softmax_triplet_c2l03_fixed_index_pcc_autodl.yml"
)
HARD_CONFIG = (
    REPO_ROOT / "configs" /
    "softmax_triplet_c2l03_hard_shortest_path_alignment_autodl.yml"
)


def changed_leaf_paths(left, right, prefix=""):
    paths = set()
    for key in set(left) | set(right):
        path = "{}.{}".format(prefix, key) if prefix else str(key)
        left_value = left.get(key, object())
        right_value = right.get(key, object())
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            paths.update(changed_leaf_paths(left_value, right_value, path))
        elif left_value != right_value:
            paths.add(path)
    return paths


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    fixed_config = yaml.safe_load(FIXED_CONFIG.read_text(encoding="utf-8"))
    hard_config = yaml.safe_load(HARD_CONFIG.read_text(encoding="utf-8"))
    differences = changed_leaf_paths(fixed_config, hard_config)
    require(
        differences == {"MODEL.PCC_MODE", "OUTPUT_DIR"},
        "formal config isolation failed: {}".format(sorted(differences)),
    )
    require(hard_config["SEED"] == 42, "formal seed must be 42")
    require(hard_config["MODEL"]["PCC_PARTS"] == 6, "alignment K must be 6")
    require(hard_config["MODEL"]["PCC_LAMBDA"] == 0.1, "lambda must be 0.1")

    diagonal_trap = torch.tensor([[[
        0.0, 10.0, 10.0,
    ], [
        10.0, 0.0, 10.0,
    ], [
        10.0, 10.0, 0.0,
    ]]])
    trap_cost = hard_shortest_path_costs(diagonal_trap)
    require(trap_cost.item() == 20.0, "diagonal move entered hard DP")

    tie_matrix = torch.zeros(1, 3, 3, requires_grad=True)
    tie_cost = hard_shortest_path_costs(tie_matrix)
    tie_cost.backward()
    expected_tie_gradient = torch.tensor([[[
        1.0, 1.0, 1.0,
    ], [
        0.0, 0.0, 1.0,
    ], [
        0.0, 0.0, 1.0,
    ]]])
    require(
        torch.equal(tie_matrix.grad, expected_tie_gradient),
        "tie gradient did not follow the up-first path",
    )

    torch.manual_seed(20260815)
    local = torch.randn(4, 6, 16, requires_grad=True)
    pids = torch.tensor([0, 0, 1, 1])
    camids = torch.tensor([0, 1, 0, 1])
    pairs = build_cross_camera_positive_pairs(pids, camids)
    local_a, local_b = select_pair_local_features(local, pairs)
    distance_matrix = pairwise_local_distance_matrix(local_a, local_b)
    require(tuple(distance_matrix.shape) == (2, 6, 6), "distance shape failed")
    manual_distance = torch.linalg.vector_norm(
        local_a.unsqueeze(2) - local_b.unsqueeze(1), dim=-1
    )
    require(
        torch.allclose(distance_matrix, manual_distance),
        "distance is not raw Euclidean L2",
    )
    hard_loss, pair_count, raw_cost, path_offset = (
        hard_shortest_path_alignment_loss(local, pids, camids)
    )
    require(pair_count == 2, "valid unordered pair selection failed")
    require(
        torch.allclose(hard_loss.detach(), raw_cost / 11.0),
        "raw path cost and normalized loss are inconsistent",
    )
    hard_loss.backward(retain_graph=True)
    require(
        local.grad is not None and torch.isfinite(local.grad).all(),
        "hard alignment gradient is missing or non-finite",
    )

    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(HARD_CONFIG))
    local_cfg.defrost()
    local_cfg.MODEL.IF_LABELSMOOTH = "off"
    local_cfg.freeze()
    score = torch.randn(4, 2, requires_grad=True)
    feature = torch.randn(4, 16, requires_grad=True)
    local_for_formula = local.detach().clone().requires_grad_(True)
    loss_output = make_loss(local_cfg, 2)(
        score, feature, pids, camids, local_for_formula
    )
    expected_total = (
        F.cross_entropy(score, pids)
        + TripletLoss(local_cfg.SOLVER.MARGIN)(feature, pids)[0]
        + 0.3 * CrossCameraPositiveLoss("mean")(feature, pids, camids)
        + 0.1 * hard_shortest_path_alignment_loss(
            local_for_formula, pids, camids
        )[0]
    )
    require(
        torch.allclose(loss_output["loss_total"], expected_total),
        "total loss formula failed",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Baseline(
        2, 1, "", "bnneck", "after", "resnet50", "none",
        part_attention=True, part_attention_parts=6,
        part_correspondence_consistency=True, pcc_parts=6,
        pcc_mode="fixed_index",
    ).to(device)
    parameter_signature = {
        name: list(parameter.shape) for name, parameter in model.named_parameters()
    }
    model.pcc_mode = "hard_shortest_path"
    require(
        parameter_signature == {
            name: list(parameter.shape)
            for name, parameter in model.named_parameters()
        },
        "alignment mode changed model parameters",
    )
    model.eval()
    images = torch.randn(1, 3, 256, 128, device=device)
    with torch.no_grad():
        inference = model(images)
    require(tuple(inference.shape) == (1, 2048), "inference descriptor changed")

    identity = experiment_identity(hard_config)
    require(identity["method_family"] == "part_alignment", "identity family failed")
    require(
        identity["method_variant"] == "hard_shortest_path",
        "identity variant failed",
    )
    require(
        identity["alignment_temperature"] == NOT_APPLICABLE
        and identity["gating_mode"] == NOT_APPLICABLE,
        "non-applicable future fields failed",
    )
    report = {
        "config_differences": sorted(differences),
        "device": str(device),
        "distance_matrix_shape": list(distance_matrix.shape),
        "fixed_index_diagonal_mean_shape": list(
            fixed_index_distances(distance_matrix).shape
        ),
        "hard_path_cost_shape": list(
            hard_shortest_path_costs(distance_matrix).shape
        ),
        "hard_alignment_loss": float(hard_loss.detach().item()),
        "mean_hard_path_cost": float(raw_cost.item()),
        "mean_path_absolute_offset": float(path_offset.item()),
        "finite_gradient": True,
        "inference_descriptor_shape": list(inference.shape),
        "parameter_count": len(parameter_signature),
        "identity": {
            "method_family": identity["method_family"],
            "method_variant": identity["method_variant"],
            "alignment_mode": identity["alignment_mode"],
            "alignment_temperature": identity["alignment_temperature"],
            "gating_mode": identity["gating_mode"],
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
