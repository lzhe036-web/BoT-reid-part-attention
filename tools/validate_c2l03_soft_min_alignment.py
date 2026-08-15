#!/usr/bin/env python
# encoding: utf-8
"""Synthetic fail-closed validation for the Soft-Min alignment protocol."""

from __future__ import absolute_import

import itertools
import json
import subprocess
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
    hard_shortest_path_costs,
    pairwise_local_distance_matrix,
    soft_min_alignment_loss,
    soft_min_path_costs,
)
from layers.triplet_loss import CrossCameraPositiveLoss, TripletLoss
from modeling.baseline import Baseline
from utils.experiment_recording import (
    NOT_APPLICABLE,
    experiment_identity,
    validate_parent_lineage,
)
from utils.multigranular_signature import (
    canonical_multigranular_feature_signature,
)


HARD_SHA = "6b46f2c3747124b97d59ed5cf987f33efb82282b"
HARD_BRANCH = "exp/c2l03-hard-shortest-path-alignment"
HARD_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2l03_hard_shortest_path_alignment_autodl.yml"
)
SOFT_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2l03_soft_min_alignment_autodl.yml"
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def changed_leaf_paths(left, right, prefix=""):
    paths = set()
    for key in set(left) | set(right):
        path = "{}.{}".format(prefix, key) if prefix else str(key)
        if key not in left or key not in right:
            paths.add(path)
            continue
        if isinstance(left[key], dict) and isinstance(right[key], dict):
            paths.update(changed_leaf_paths(left[key], right[key], path))
        elif left[key] != right[key]:
            paths.add(path)
    return paths


def right_down_oracle(matrix, tau):
    parts = int(matrix.size(0))
    costs = []
    for down_positions in itertools.combinations(
            range(2 * parts - 2), parts - 1):
        down_positions = set(down_positions)
        row = column = 0
        cost = matrix[0, 0]
        for move_index in range(2 * parts - 2):
            if move_index in down_positions:
                row += 1
            else:
                column += 1
            cost = cost + matrix[row, column]
        costs.append(cost)
    path_costs = torch.stack(costs)
    return -tau * torch.logsumexp(-path_costs / tau, dim=0)


def main():
    hard_text = subprocess.check_output(
        [
            "git", "show", "{}:{}".format(
                HARD_SHA,
                "configs/softmax_triplet_c2l03_hard_shortest_path_alignment_autodl.yml",
            ),
        ],
        cwd=str(REPO_ROOT), text=True,
    )
    require(
        HARD_CONFIG.read_text(encoding="utf-8").replace("\r\n", "\n")
        == hard_text.replace("\r\n", "\n"),
        "working-tree Hard formal config differs from fixed Hard commit",
    )
    hard = yaml.safe_load(hard_text)
    soft = yaml.safe_load(SOFT_CONFIG.read_text(encoding="utf-8"))
    differences = changed_leaf_paths(hard, soft)
    require(
        differences == {
            "MODEL.PCC_MODE", "MODEL.PCC_SOFTMIN_TAU", "OUTPUT_DIR"
        },
        "formal config isolation failed: {}".format(sorted(differences)),
    )
    require(soft["SEED"] == 42, "Soft formal seed must be 42")
    require(soft["MODEL"]["PCC_PARTS"] == 6, "Soft K must be 6")
    require(soft["MODEL"]["PCC_LAMBDA"] == 0.1, "Soft lambda must be 0.1")
    tau = float(soft["MODEL"]["PCC_SOFTMIN_TAU"])
    require(tau == 0.1, "Soft candidate tau must be explicit 0.1")

    lineage = validate_parent_lineage(
        REPO_ROOT, HARD_BRANCH, HARD_SHA, child_commit="HEAD"
    )
    origin_hard = subprocess.check_output(
        ["git", "rev-parse", "origin/{}".format(HARD_BRANCH)],
        cwd=str(REPO_ROOT), text=True,
    ).strip()
    require(origin_hard == HARD_SHA, "remote-tracking Hard tip changed")

    hard_signature, hard_signature_sha = (
        canonical_multigranular_feature_signature(hard)
    )
    soft_signature, soft_signature_sha = (
        canonical_multigranular_feature_signature(soft)
    )
    require(hard_signature == soft_signature, "feature definitions differ")
    require(
        hard_signature_sha == soft_signature_sha,
        "feature signature SHA256 differs",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260815)
    matrices = torch.randn(4, 4, 4, dtype=torch.float64, device=device)
    actual = soft_min_path_costs(matrices, tau)
    expected = torch.stack([right_down_oracle(item, tau) for item in matrices])
    require(torch.allclose(actual, expected, atol=1e-11), "Soft DP/oracle failed")
    diagonal_trap = torch.tensor([[[
        0.0, 10.0, 10.0,
    ], [
        10.0, 0.0, 10.0,
    ], [
        10.0, 10.0, 0.0,
    ]]], device=device)
    require(
        soft_min_path_costs(diagonal_trap, tau).item() > 19.0,
        "diagonal entered Soft DP",
    )
    require(
        torch.allclose(
            soft_min_path_costs(matrices, 1.0e-6),
            hard_shortest_path_costs(matrices),
            atol=1e-5,
        ),
        "tau-to-zero Hard limit failed",
    )

    local = torch.randn(4, 6, 16, device=device, requires_grad=True)
    pids = torch.tensor([0, 0, 1, 1], device=device)
    camids = torch.tensor([0, 1, 0, 1], device=device)
    pairs = build_cross_camera_positive_pairs(pids, camids)
    require(pairs.tolist() == [[0, 1], [2, 3]], "pair selection changed")
    distance = pairwise_local_distance_matrix(
        local.index_select(0, pairs[:, 0]),
        local.index_select(0, pairs[:, 1]),
    )
    manual_distance = torch.linalg.vector_norm(
        local.index_select(0, pairs[:, 0]).unsqueeze(2)
        - local.index_select(0, pairs[:, 1]).unsqueeze(1),
        dim=-1,
    )
    require(tuple(distance.shape) == (2, 6, 6), "distance shape changed")
    require(torch.allclose(distance, manual_distance), "distance is not raw L2")
    soft_loss, pair_count, raw_cost = soft_min_alignment_loss(
        local, pids, camids, tau
    )
    require(pair_count == 2, "pair count failed")
    require(
        torch.allclose(soft_loss.detach(), raw_cost / 11.0),
        "Soft cost normalization failed",
    )
    soft_loss.backward(retain_graph=True)
    require(
        local.grad is not None and torch.isfinite(local.grad).all(),
        "Soft local gradient is missing/non-finite",
    )

    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(SOFT_CONFIG))
    local_cfg.defrost()
    local_cfg.MODEL.IF_LABELSMOOTH = "off"
    local_cfg.freeze()
    score = torch.randn(4, 2, device=device, requires_grad=True)
    feature = torch.randn(4, 16, device=device, requires_grad=True)
    formula_local = local.detach().clone().requires_grad_(True)
    loss_output = make_loss(local_cfg, 2)(
        score, feature, pids, camids, formula_local
    )
    expected_total = (
        F.cross_entropy(score, pids)
        + TripletLoss(local_cfg.SOLVER.MARGIN)(feature, pids)[0]
        + 0.3 * CrossCameraPositiveLoss("mean")(feature, pids, camids)
        + 0.1 * soft_min_alignment_loss(
            formula_local, pids, camids, tau
        )[0]
    )
    require(
        torch.allclose(loss_output["loss_total"], expected_total),
        "total loss formula failed",
    )

    model = Baseline(
        2, 1, "", "bnneck", "after", "resnet50", "none",
        part_attention=True, part_attention_parts=6,
        part_correspondence_consistency=True, pcc_parts=6,
        pcc_mode="hard_shortest_path", pcc_softmin_tau=tau,
    ).to(device)
    parameter_schema = {
        name: list(parameter.shape) for name, parameter in model.named_parameters()
    }
    model.pcc_mode = "soft_min"
    require(parameter_schema == {
        name: list(parameter.shape) for name, parameter in model.named_parameters()
    }, "alignment mode changed parameters")
    model.eval()
    with torch.no_grad():
        inference = model(torch.randn(1, 3, 256, 128, device=device))
    require(tuple(inference.shape) == (1, 2048), "inference shape changed")

    identity = experiment_identity(soft)
    require(identity["method_variant"] == "soft_min", "identity failed")
    require(identity["alignment_temperature"] == tau, "identity tau failed")
    require(identity["gating_mode"] == NOT_APPLICABLE, "gating sentinel failed")
    report = {
        "device": str(device),
        "parent_lineage": lineage,
        "config_differences": sorted(differences),
        "seed": soft["SEED"],
        "tau": tau,
        "distance_matrix_shape": list(distance.shape),
        "soft_alignment_loss": float(soft_loss.detach()),
        "mean_soft_path_cost": float(raw_cost),
        "finite_gradient": True,
        "inference_descriptor_shape": list(inference.shape),
        "feature_signature_sha256": soft_signature_sha,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
