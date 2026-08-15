# encoding: utf-8
"""Algorithm, integration, and fairness tests for hard part alignment."""

from __future__ import absolute_import

import itertools
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from config import cfg
from layers import make_loss
from layers.part_correspondence_consistency import (
    fixed_index_distances,
    hard_shortest_path_alignment_loss,
    hard_shortest_path_cost,
    hard_shortest_path_costs,
    hard_shortest_path_costs_and_offsets,
    pairwise_local_distance_matrix,
    part_alignment_loss,
)
from layers.triplet_loss import CrossCameraPositiveLoss, TripletLoss
from modeling.baseline import Baseline


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_CONFIG = (
    REPO_ROOT / "configs" /
    "softmax_triplet_c2l03_fixed_index_pcc_autodl.yml"
)
HARD_CONFIG = (
    REPO_ROOT / "configs" /
    "softmax_triplet_c2l03_hard_shortest_path_alignment_autodl.yml"
)


def _brute_force_cost(matrix):
    parts = matrix.size(0)
    moves = ["down"] * (parts - 1) + ["right"] * (parts - 1)
    costs = []
    for permutation in set(itertools.permutations(moves)):
        row = column = 0
        cost = matrix[row, column]
        for move in permutation:
            if move == "down":
                row += 1
            else:
                column += 1
            cost = cost + matrix[row, column]
        costs.append(cost)
    return torch.stack(costs).min()


def _changed_leaf_paths(left, right, prefix=""):
    paths = set()
    keys = set(left) | set(right)
    for key in keys:
        path = "{}.{}".format(prefix, key) if prefix else str(key)
        left_value = left.get(key, object())
        right_value = right.get(key, object())
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            paths.update(_changed_leaf_paths(left_value, right_value, path))
        elif left_value != right_value:
            paths.add(path)
    return paths


class HardShortestPathAlignmentTest(unittest.TestCase):
    def test_batched_and_single_distance_matrix_shapes(self):
        first = torch.randn(2, 6, 8)
        second = torch.randn(2, 6, 8)
        batched = pairwise_local_distance_matrix(first, second)
        single = pairwise_local_distance_matrix(first[:1], second[:1])[0]
        self.assertEqual(tuple(batched.shape), (2, 6, 6))
        self.assertEqual(tuple(single.shape), (6, 6))

    def test_known_shortest_path_cost(self):
        matrix = torch.tensor([
            [1.0, 2.0, 9.0],
            [9.0, 3.0, 4.0],
            [9.0, 9.0, 5.0],
        ])
        self.assertEqual(hard_shortest_path_cost(matrix).item(), 15.0)

    def test_dp_matches_brute_force_for_small_matrices(self):
        torch.manual_seed(29)
        matrices = torch.rand(7, 4, 4)
        actual = hard_shortest_path_costs(matrices)
        expected = torch.stack([_brute_force_cost(item) for item in matrices])
        self.assertTrue(torch.allclose(actual, expected))

    def test_diagonal_move_is_not_available(self):
        matrix = torch.tensor([
            [0.0, 10.0, 10.0],
            [10.0, 0.0, 10.0],
            [10.0, 10.0, 0.0],
        ])
        self.assertEqual(hard_shortest_path_cost(matrix).item(), 20.0)

    def test_tie_break_is_up_first_for_statistics_and_gradient(self):
        matrix = torch.zeros(1, 3, 3, requires_grad=True)
        costs, offsets = hard_shortest_path_costs_and_offsets(matrix)
        costs.sum().backward()
        expected_gradient = torch.tensor([[[
            1.0, 1.0, 1.0,
        ], [
            0.0, 0.0, 1.0,
        ], [
            0.0, 0.0, 1.0,
        ]]])
        self.assertTrue(torch.equal(matrix.grad, expected_gradient))
        self.assertAlmostEqual(offsets.item(), 4.0 / 5.0)

    def test_hard_and_fixed_differ_on_monotonic_shift(self):
        matrix = torch.tensor([[[
            0.0, 0.0, 0.0,
        ], [
            100.0, 100.0, 0.0,
        ], [
            100.0, 100.0, 0.0,
        ]]])
        self.assertEqual(hard_shortest_path_costs(matrix).item(), 0.0)
        self.assertGreater(fixed_index_distances(matrix).item(), 30.0)

    def test_zero_valid_pair_is_graph_safe(self):
        local = torch.randn(4, 6, 12, requires_grad=True)
        pids = torch.tensor([0, 0, 1, 1])
        camids = torch.tensor([0, 0, 1, 1])
        loss, count, cost, offset = hard_shortest_path_alignment_loss(
            local, pids, camids
        )
        self.assertEqual(count, 0)
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual((loss.item(), cost.item(), offset.item()), (0.0, 0.0, 0.0))
        loss.backward()
        self.assertTrue(torch.equal(local.grad, torch.zeros_like(local)))

    def test_valid_hard_loss_requires_grad_and_backward_is_finite(self):
        local = torch.randn(4, 6, 12, requires_grad=True)
        pids = torch.tensor([0, 0, 1, 1])
        camids = torch.tensor([0, 1, 0, 1])
        loss, count, cost, offset = hard_shortest_path_alignment_loss(
            local, pids, camids
        )
        self.assertEqual(count, 2)
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(cost))
        self.assertTrue(torch.isfinite(offset))
        loss.backward()
        self.assertIsNotNone(local.grad)
        self.assertTrue(torch.isfinite(local.grad).all())

    def test_hard_loss_uses_raw_euclidean_matrix_and_divides_by_eleven(self):
        local = torch.randn(2, 6, 5, requires_grad=True)
        pids = torch.tensor([3, 3])
        camids = torch.tensor([0, 1])
        distances = pairwise_local_distance_matrix(local[:1], local[1:])
        expected_cost = hard_shortest_path_costs(distances)[0]
        loss, count, mean_cost, _offset = hard_shortest_path_alignment_loss(
            local, pids, camids
        )
        self.assertEqual(count, 1)
        self.assertTrue(torch.allclose(mean_cost, expected_cost.detach()))
        self.assertTrue(torch.allclose(loss, expected_cost / 11.0))

    def test_hard_mode_total_loss_formula(self):
        local_cfg = cfg.clone()
        local_cfg.defrost()
        local_cfg.MODEL.IF_LABELSMOOTH = "off"
        local_cfg.MODEL.CAMERA_AWARE_TRIPLET = False
        local_cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY = True
        local_cfg.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA = 0.3
        local_cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY = True
        local_cfg.MODEL.PCC_PARTS = 6
        local_cfg.MODEL.PCC_LAMBDA = 0.1
        local_cfg.MODEL.PCC_MODE = "hard_shortest_path"
        local_cfg.DATALOADER.SAMPLER = "softmax_triplet"
        local_cfg.freeze()
        torch.manual_seed(31)
        score = torch.randn(4, 2, requires_grad=True)
        feature = torch.randn(4, 16, requires_grad=True)
        local = torch.randn(4, 6, 16, requires_grad=True)
        target = torch.tensor([0, 0, 1, 1])
        camids = torch.tensor([0, 1, 0, 1])
        output = make_loss(local_cfg, 2)(score, feature, target, camids, local)
        hard_loss = hard_shortest_path_alignment_loss(
            local, target, camids
        )[0]
        expected = (
            F.cross_entropy(score, target)
            + TripletLoss(local_cfg.SOLVER.MARGIN)(feature, target)[0]
            + 0.3 * CrossCameraPositiveLoss("mean")(feature, target, camids)
            + 0.1 * hard_loss
        )
        self.assertTrue(torch.allclose(output["loss_total"], expected))
        self.assertTrue(torch.equal(output["loss_pcc"], output["hard_alignment_loss"]))
        self.assertEqual(output["valid_alignment_pair_count"], 2)

    def test_dispatch_reuses_descriptors_and_euclidean_distance(self):
        local = torch.randn(4, 6, 7, requires_grad=True)
        pids = torch.tensor([0, 0, 1, 1])
        camids = torch.tensor([0, 1, 0, 1])
        fixed = part_alignment_loss(local, pids, camids, "fixed_index")
        hard = part_alignment_loss(local, pids, camids, "hard_shortest_path")
        self.assertEqual(fixed["valid_pcc_pair_count"], 2)
        self.assertEqual(hard["valid_pcc_pair_count"], 2)
        manual = torch.linalg.vector_norm(
            local[0].unsqueeze(1) - local[1].unsqueeze(0), dim=-1
        )
        self.assertTrue(torch.allclose(
            fixed["mean_fixed_index_part_distance"],
            torch.stack([
                manual.diagonal().mean(),
                torch.linalg.vector_norm(
                    local[2].unsqueeze(1) - local[3].unsqueeze(0), dim=-1
                ).diagonal().mean(),
            ]).mean(),
        ))

    def test_unknown_mode_fails_closed(self):
        local = torch.randn(2, 6, 4)
        with self.assertRaisesRegex(ValueError, "Unsupported PCC_MODE"):
            part_alignment_loss(
                local, torch.tensor([0, 0]), torch.tensor([0, 1]), "unknown"
            )

    def test_formal_config_deep_diff_is_only_mode_and_output(self):
        fixed = yaml.safe_load(FIXED_CONFIG.read_text(encoding="utf-8"))
        hard = yaml.safe_load(HARD_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            _changed_leaf_paths(fixed, hard),
            {"MODEL.PCC_MODE", "OUTPUT_DIR"},
        )
        self.assertEqual(hard["SEED"], 42)
        self.assertEqual(hard["SOLVER"]["MAX_EPOCHS"], 120)
        self.assertEqual(hard["MODEL"]["PCC_PARTS"], 6)
        self.assertEqual(hard["MODEL"]["PCC_LAMBDA"], 0.1)
        self.assertEqual(hard["TEST"]["NECK_FEAT"], "after")
        self.assertEqual(hard["TEST"]["FEAT_NORM"], "yes")
        self.assertEqual(hard["TEST"]["RE_RANKING"], "no")

    def test_mode_does_not_change_parameters_or_inference_descriptor(self):
        torch.manual_seed(37)
        model = Baseline(
            2, 1, "", "bnneck", "after", "resnet50", "none",
            part_attention=True, part_attention_parts=6,
            part_correspondence_consistency=True, pcc_parts=6,
            pcc_mode="fixed_index",
        )
        signature = {
            name: tuple(parameter.shape)
            for name, parameter in model.named_parameters()
        }
        model.pcc_mode = "hard_shortest_path"
        self.assertEqual(signature, {
            name: tuple(parameter.shape)
            for name, parameter in model.named_parameters()
        })
        model.eval()
        with torch.no_grad():
            descriptor = model(torch.randn(1, 3, 256, 128))
        self.assertEqual(tuple(descriptor.shape), (1, 2048))


if __name__ == "__main__":
    unittest.main()
