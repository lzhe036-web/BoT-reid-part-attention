# encoding: utf-8
"""Soft-Min algorithm, feature-equivalence, and protocol tests."""

from __future__ import absolute_import

import itertools
import json
import math
import subprocess
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from config import cfg
from layers import make_loss
from layers.part_correspondence_consistency import (
    SUPPORTED_ALIGNMENT_MODES,
    build_cross_camera_positive_pairs,
    hard_shortest_path_costs,
    pairwise_local_distance_matrix,
    part_alignment_loss,
    soft_min_alignment_loss,
    soft_min_path_cost,
    soft_min_path_costs,
    softmin_two_predecessors,
    validate_softmin_tau,
)
from layers.triplet_loss import CrossCameraPositiveLoss, TripletLoss
from modeling.baseline import Baseline
from utils.experiment_recording import NOT_APPLICABLE, experiment_identity
from utils.multigranular_signature import (
    canonical_multigranular_feature_signature,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HARD_SHA = "6b46f2c3747124b97d59ed5cf987f33efb82282b"
HARD_BRANCH = "exp/c2l03-hard-shortest-path-alignment"
HARD_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2l03_hard_shortest_path_alignment_autodl.yml"
)
SOFT_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2l03_soft_min_alignment_autodl.yml"
)


def _changed_leaf_paths(left, right, prefix=""):
    paths = set()
    for key in set(left) | set(right):
        path = "{}.{}".format(prefix, key) if prefix else str(key)
        if key not in left or key not in right:
            paths.add(path)
            continue
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            paths.update(_changed_leaf_paths(left_value, right_value, path))
        elif left_value != right_value:
            paths.add(path)
    return paths


def _right_down_path_costs(matrix):
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
    return torch.stack(costs)


def _soft_path_oracle(matrix, tau):
    path_costs = _right_down_path_costs(matrix)
    return -tau * torch.logsumexp(-path_costs / tau, dim=0)


class SoftMinAlignmentTest(unittest.TestCase):
    def test_branch_lineage_uses_fixed_hard_commit(self):
        hard_tip = subprocess.check_output(
            ["git", "rev-parse", HARD_BRANCH], cwd=str(REPO_ROOT), text=True
        ).strip()
        merge_base = subprocess.check_output(
            ["git", "merge-base", HARD_BRANCH, "HEAD"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
        self.assertEqual(hard_tip, HARD_SHA)
        self.assertEqual(merge_base, HARD_SHA)

    def test_known_soft_min_value(self):
        tau = 0.1
        matrix = torch.zeros(2, 2)
        expected = -tau * math.log(2.0)
        self.assertAlmostEqual(
            soft_min_path_cost(matrix, tau).item(), expected, places=6
        )

    def test_dp_matches_all_right_down_paths(self):
        torch.manual_seed(41)
        matrices = torch.randn(5, 4, 4, dtype=torch.float64)
        tau = 0.37
        expected = torch.stack([
            _soft_path_oracle(matrix, tau) for matrix in matrices
        ])
        actual = soft_min_path_costs(matrices, tau)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-11))

    def test_diagonal_move_is_not_available(self):
        matrix = torch.tensor([
            [0.0, 10.0, 10.0],
            [10.0, 0.0, 10.0],
            [10.0, 10.0, 0.0],
        ])
        actual = soft_min_path_cost(matrix, 0.1)
        expected = _soft_path_oracle(matrix, 0.1)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertGreater(actual.item(), 19.0)

    def test_temperature_validation(self):
        self.assertEqual(validate_softmin_tau(0.1), 0.1)
        for invalid in (0.0, -0.1, float("nan"), float("inf"),
                        -float("inf"), True, "bad"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_softmin_tau(invalid)

    def test_large_values_are_numerically_stable(self):
        matrix = torch.full((3, 4, 4), 1.0e20, dtype=torch.float64)
        costs = soft_min_path_costs(matrix, 0.1)
        self.assertTrue(torch.isfinite(costs).all())

    def test_tau_tends_to_hard_min(self):
        torch.manual_seed(43)
        matrix = torch.rand(4, 4, dtype=torch.float64)
        soft = soft_min_path_cost(matrix, 1.0e-6)
        hard = hard_shortest_path_costs(matrix.unsqueeze(0))[0]
        self.assertTrue(torch.allclose(soft, hard, atol=1e-5))

    def test_both_predecessors_receive_gradient(self):
        up = torch.tensor([1.0], requires_grad=True)
        left = torch.tensor([2.0], requires_grad=True)
        softmin_two_predecessors(up, left, 0.5).backward()
        self.assertGreater(up.grad.item(), 0.0)
        self.assertGreater(left.grad.item(), 0.0)
        self.assertAlmostEqual(
            up.grad.item() + left.grad.item(), 1.0, delta=1e-6
        )

    def test_local_gradient_and_pair_selection_are_finite(self):
        local = torch.randn(5, 6, 8, requires_grad=True)
        pids = torch.tensor([0, 0, 0, 1, 1])
        camids = torch.tensor([0, 1, 1, 0, 1])
        pairs = build_cross_camera_positive_pairs(pids, camids)
        self.assertEqual(pairs.tolist(), [[0, 1], [0, 2], [3, 4]])
        loss, count, raw_cost = soft_min_alignment_loss(
            local, pids, camids, 0.1
        )
        self.assertEqual(count, 3)
        self.assertTrue(torch.allclose(loss.detach(), raw_cost / 11.0))
        loss.backward()
        self.assertIsNotNone(local.grad)
        self.assertTrue(torch.isfinite(local.grad).all())

    def test_zero_pair_backward_is_graph_safe(self):
        local = torch.randn(4, 6, 8, requires_grad=True)
        loss, count, cost = soft_min_alignment_loss(
            local,
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 0, 1, 1]),
            0.1,
        )
        self.assertEqual(count, 0)
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual((loss.item(), cost.item()), (0.0, 0.0))
        loss.backward()
        self.assertTrue(torch.equal(local.grad, torch.zeros_like(local)))

    def test_distance_and_dispatch_reuse_shared_pipeline(self):
        local = torch.randn(4, 6, 9, requires_grad=True)
        pids = torch.tensor([0, 0, 1, 1])
        camids = torch.tensor([0, 1, 0, 1])
        distances = pairwise_local_distance_matrix(local[:2], local[2:])
        manual = torch.linalg.vector_norm(
            local[:2].unsqueeze(2) - local[2:].unsqueeze(1), dim=-1
        )
        self.assertEqual(tuple(distances.shape), (2, 6, 6))
        self.assertTrue(torch.allclose(distances, manual))
        hard = part_alignment_loss(
            local, pids, camids, "hard_shortest_path"
        )
        soft = part_alignment_loss(local, pids, camids, "soft_min", 0.1)
        self.assertEqual(hard["valid_pcc_pair_count"], 2)
        self.assertEqual(soft["valid_pcc_pair_count"], 2)

    def test_soft_total_loss_formula(self):
        local_cfg = cfg.clone()
        local_cfg.merge_from_file(str(SOFT_CONFIG))
        local_cfg.defrost()
        local_cfg.MODEL.IF_LABELSMOOTH = "off"
        local_cfg.freeze()
        torch.manual_seed(47)
        score = torch.randn(4, 2, requires_grad=True)
        feature = torch.randn(4, 16, requires_grad=True)
        local = torch.randn(4, 6, 16, requires_grad=True)
        target = torch.tensor([0, 0, 1, 1])
        camids = torch.tensor([0, 1, 0, 1])
        output = make_loss(local_cfg, 2)(
            score, feature, target, camids, local
        )
        soft_loss = soft_min_alignment_loss(
            local, target, camids, local_cfg.MODEL.PCC_SOFTMIN_TAU
        )[0]
        expected = (
            F.cross_entropy(score, target)
            + TripletLoss(local_cfg.SOLVER.MARGIN)(feature, target)[0]
            + 0.3 * CrossCameraPositiveLoss("mean")(
                feature, target, camids
            )
            + 0.1 * soft_loss
        )
        self.assertTrue(torch.allclose(output["loss_total"], expected))
        self.assertTrue(torch.equal(
            output["loss_pcc"], output["soft_alignment_loss"]
        ))

    def test_formal_config_diff_has_only_three_allowed_paths(self):
        hard = yaml.safe_load(HARD_CONFIG.read_text(encoding="utf-8"))
        soft = yaml.safe_load(SOFT_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            _changed_leaf_paths(hard, soft),
            {"MODEL.PCC_MODE", "MODEL.PCC_SOFTMIN_TAU", "OUTPUT_DIR"},
        )
        self.assertEqual(soft["SEED"], 42)
        self.assertEqual(soft["MODEL"]["PCC_SOFTMIN_TAU"], 0.1)

    def test_feature_signatures_are_identical(self):
        hard_text = subprocess.check_output(
            [
                "git", "show", "{}:{}".format(
                    HARD_SHA,
                    "configs/softmax_triplet_c2l03_hard_shortest_path_alignment_autodl.yml",
                ),
            ],
            cwd=str(REPO_ROOT), text=True,
        )
        self.assertEqual(
            HARD_CONFIG.read_text(encoding="utf-8").replace("\r\n", "\n"),
            hard_text.replace("\r\n", "\n"),
        )
        hard = yaml.safe_load(hard_text)
        soft = yaml.safe_load(SOFT_CONFIG.read_text(encoding="utf-8"))
        hard_signature, hard_sha = (
            canonical_multigranular_feature_signature(hard)
        )
        soft_signature, soft_sha = (
            canonical_multigranular_feature_signature(soft)
        )
        self.assertEqual(hard_signature, soft_signature)
        self.assertEqual(hard_sha, soft_sha)
        self.assertEqual(len(soft_sha), 64)
        payload = json.loads(soft_signature)
        self.assertEqual(payload["global_descriptor"]["shape"], ["B", 2048])
        self.assertEqual(
            payload["local_descriptor"]["shape"], ["B", 6, 2048]
        )
        self.assertEqual(
            payload["inference_descriptor"]["shape"], ["B", 2048]
        )
        parameter_names = {
            item["name"] for item in payload["feature_parameter_schema"]
        }
        self.assertIn("base.conv1.weight", parameter_names)
        self.assertIn("part_attention_head.attention.weight", parameter_names)
        self.assertIn("bottleneck.weight", parameter_names)
        self.assertTrue(all(
            len(value) == 64
            for value in payload["feature_implementation_sha256"].values()
        ))

    def test_mode_does_not_change_parameters_or_inference_shape(self):
        model = Baseline(
            2, 1, "", "bnneck", "after", "resnet50", "none",
            part_attention=True, part_attention_parts=6,
            part_correspondence_consistency=True, pcc_parts=6,
            pcc_mode="hard_shortest_path", pcc_softmin_tau=0.1,
        )
        parameters = {
            name: tuple(parameter.shape)
            for name, parameter in model.named_parameters()
        }
        model.pcc_mode = "soft_min"
        self.assertEqual(parameters, {
            name: tuple(parameter.shape)
            for name, parameter in model.named_parameters()
        })
        model.eval()
        with torch.no_grad():
            descriptor = model(torch.randn(1, 3, 256, 128))
        self.assertEqual(tuple(descriptor.shape), (1, 2048))

    def test_identity_and_future_fields(self):
        soft = yaml.safe_load(SOFT_CONFIG.read_text(encoding="utf-8"))
        identity = experiment_identity(soft)
        self.assertEqual(identity["method_family"], "part_alignment")
        self.assertEqual(identity["method_variant"], "soft_min")
        self.assertEqual(identity["alignment_mode"], "soft_min")
        self.assertEqual(identity["alignment_temperature"], 0.1)
        self.assertEqual(identity["gating_mode"], NOT_APPLICABLE)
        self.assertEqual(identity["gating_temperature"], NOT_APPLICABLE)
        self.assertNotIn("dynamic_gating", SUPPORTED_ALIGNMENT_MODES)

    def test_runners_use_unified_recorder_and_fixed_parent_evidence(self):
        runners = {
            "formal": REPO_ROOT / "scripts" /
            "train_c2l03_soft_min_alignment_autodl.sh",
            "smoke": REPO_ROOT / "scripts" /
            "test_c2l03_soft_min_alignment_1epoch.sh",
        }
        for run_kind, path in runners.items():
            text = path.read_text(encoding="utf-8")
            with self.subTest(run_kind=run_kind):
                self.assertIn("python tools/run_experiment.py", text)
                self.assertNotIn("python tools/train.py", text)
                self.assertIn("--run-kind {}".format(run_kind), text)
                self.assertIn(HARD_SHA, text)
                self.assertIn('--parent-branch "${PARENT_BRANCH}"', text)
                self.assertIn('--parent-commit "${PARENT_COMMIT}"', text)
        smoke_text = runners["smoke"].read_text(encoding="utf-8")
        self.assertIn("SOLVER.MAX_EPOCHS 1", smoke_text)
        self.assertIn(
            "c2l03_soft_min_alignment_tau0p1_seed42_market1501_smoke",
            smoke_text,
        )


if __name__ == "__main__":
    unittest.main()
