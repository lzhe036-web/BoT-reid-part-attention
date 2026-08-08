# encoding: utf-8
"""Unit and compatibility tests for Fixed-Index PCC."""

from __future__ import absolute_import

import copy
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from config import cfg
from layers import make_loss
from layers.part_correspondence_consistency import (
    build_cross_camera_positive_pairs,
    build_local_part_descriptors,
    fixed_index_distances,
    fixed_index_pcc_loss,
    horizontal_part_bounds,
    pairwise_local_distance_matrix,
    select_pair_local_features,
)
from layers.triplet_loss import CrossCameraPositiveLoss, TripletLoss
from modeling.baseline import Baseline


REPO_ROOT = Path(__file__).resolve().parents[1]


class PartCorrespondenceConsistencyTest(unittest.TestCase):
    def test_horizontal_bounds_cover_height_without_gap_overlap_or_empty_part(self):
        bounds = horizontal_part_bounds(16, 6)
        self.assertEqual(bounds, [(0, 2), (2, 5), (5, 8), (8, 10), (10, 13), (13, 16)])
        covered = [row for start, end in bounds for row in range(start, end)]
        self.assertEqual(covered, list(range(16)))
        self.assertTrue(all(start < end for start, end in bounds))

    def test_local_descriptor_shape_is_b_k_c(self):
        feature_map = torch.randn(3, 2048, 16, 8)
        local = build_local_part_descriptors(feature_map, 6)
        self.assertEqual(tuple(local.shape), (3, 6, 2048))

    def test_pair_definition_is_unique_same_pid_different_camera(self):
        pids = torch.tensor([1, 1, 1, 2, 2, 3])
        camids = torch.tensor([0, 1, 0, 0, 0, 1])
        pairs = build_cross_camera_positive_pairs(pids, camids)
        self.assertEqual(pairs.tolist(), [[0, 1], [1, 2]])
        self.assertTrue(torch.all(pairs[:, 0] < pairs[:, 1]))

    def test_pair_local_features_and_full_distance_matrix_shapes(self):
        local = torch.randn(4, 6, 2048)
        pairs = torch.tensor([[0, 1], [2, 3]])
        local_a, local_b = select_pair_local_features(local, pairs)
        distances = pairwise_local_distance_matrix(local_a, local_b)
        self.assertEqual(tuple(local_a.shape), (2, 6, 2048))
        self.assertEqual(tuple(local_b.shape), (2, 6, 2048))
        self.assertEqual(tuple(distances.shape), (2, 6, 6))

    def test_fixed_loss_ignores_every_non_diagonal_element(self):
        matrix = torch.arange(72, dtype=torch.float32).reshape(2, 6, 6)
        first = fixed_index_distances(matrix)
        changed = matrix.clone()
        off_diagonal = ~torch.eye(6, dtype=torch.bool).unsqueeze(0).expand_as(changed)
        changed[off_diagonal] = changed[off_diagonal] + 10000.0
        second = fixed_index_distances(changed)
        self.assertTrue(torch.equal(first, second))

    def test_fixed_loss_changes_when_diagonal_changes(self):
        matrix = torch.zeros(1, 6, 6)
        first = fixed_index_distances(matrix)
        matrix[:, 3, 3] = 6.0
        second = fixed_index_distances(matrix)
        self.assertFalse(torch.equal(first, second))
        self.assertAlmostEqual(second.item(), 1.0)

    def test_no_valid_pair_returns_graph_safe_zero(self):
        local = torch.randn(4, 6, 32, requires_grad=True)
        pids = torch.tensor([0, 0, 1, 1])
        camids = torch.tensor([0, 0, 1, 1])
        loss, pair_count, mean_distance = fixed_index_pcc_loss(local, pids, camids)
        self.assertEqual(pair_count, 0)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(mean_distance.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(local.grad, torch.zeros_like(local)))

    def test_valid_pair_loss_and_backward_are_finite(self):
        local = torch.randn(4, 6, 32, requires_grad=True)
        pids = torch.tensor([0, 0, 1, 1])
        camids = torch.tensor([0, 1, 0, 1])
        loss, pair_count, _ = fixed_index_pcc_loss(local, pids, camids)
        self.assertEqual(pair_count, 2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(local.grad).all())

    def test_loss_integration_keeps_two_lambdas_independent(self):
        local_cfg = cfg.clone()
        local_cfg.defrost()
        local_cfg.MODEL.IF_LABELSMOOTH = 'off'
        local_cfg.MODEL.CAMERA_AWARE_TRIPLET = False
        local_cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY = True
        local_cfg.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA = 0.3
        local_cfg.MODEL.CROSS_CAMERA_POSITIVE_MODE = 'mean'
        local_cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY = True
        local_cfg.MODEL.PCC_PARTS = 6
        local_cfg.MODEL.PCC_LAMBDA = 0.1
        local_cfg.MODEL.PCC_MODE = 'fixed_index'
        local_cfg.DATALOADER.SAMPLER = 'softmax_triplet'
        local_cfg.freeze()

        torch.manual_seed(7)
        score = torch.randn(4, 2, requires_grad=True)
        feat = torch.randn(4, 16, requires_grad=True)
        local = torch.randn(4, 6, 16, requires_grad=True)
        target = torch.tensor([0, 0, 1, 1])
        camids = torch.tensor([0, 1, 0, 1])
        output = make_loss(local_cfg, 2)(score, feat, target, camids, local)

        expected = (
            F.cross_entropy(score, target)
            + TripletLoss(local_cfg.SOLVER.MARGIN)(feat, target)[0]
            + 0.3 * CrossCameraPositiveLoss('mean')(feat, target, camids)
            + 0.1 * fixed_index_pcc_loss(local, target, camids)[0]
        )
        self.assertTrue(torch.allclose(output['loss_total'], expected))
        self.assertEqual(output['valid_pcc_pair_count'], 2)
        output['loss_total'].backward()
        self.assertTrue(torch.isfinite(local.grad).all())

    def test_pcc_off_preserves_c2l03_loss_formula(self):
        local_cfg = cfg.clone()
        local_cfg.defrost()
        local_cfg.MODEL.IF_LABELSMOOTH = 'off'
        local_cfg.MODEL.CAMERA_AWARE_TRIPLET = False
        local_cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY = True
        local_cfg.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA = 0.3
        local_cfg.MODEL.CROSS_CAMERA_POSITIVE_MODE = 'mean'
        local_cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY = False
        local_cfg.DATALOADER.SAMPLER = 'softmax_triplet'
        local_cfg.freeze()

        torch.manual_seed(13)
        score = torch.randn(4, 2, requires_grad=True)
        feat = torch.randn(4, 16, requires_grad=True)
        target = torch.tensor([0, 0, 1, 1])
        camids = torch.tensor([0, 1, 0, 1])
        output = make_loss(local_cfg, 2)(score, feat, target, camids)
        expected = (
            F.cross_entropy(score, target)
            + TripletLoss(local_cfg.SOLVER.MARGIN)(feat, target)[0]
            + 0.3 * CrossCameraPositiveLoss('mean')(feat, target, camids)
        )
        self.assertTrue(torch.allclose(output['loss_total'], expected))
        self.assertEqual(output['loss_pcc'].item(), 0.0)
        self.assertEqual(output['valid_pcc_pair_count'], 0)

    def test_formal_config_changes_only_pcc_keys_and_output_dir(self):
        with (REPO_ROOT / 'configs' / 'softmax_triplet_c2l03_seed42_autodl.yml').open(encoding='utf-8') as handle:
            baseline = yaml.safe_load(handle)
        with (REPO_ROOT / 'configs' / 'softmax_triplet_c2l03_fixed_index_pcc_autodl.yml').open(encoding='utf-8') as handle:
            pcc = yaml.safe_load(handle)
        stripped = copy.deepcopy(pcc)
        for key in (
            'PART_CORRESPONDENCE_CONSISTENCY', 'PCC_PARTS',
            'PCC_LAMBDA', 'PCC_MODE',
        ):
            stripped['MODEL'].pop(key)
        stripped['OUTPUT_DIR'] = baseline['OUTPUT_DIR']
        self.assertEqual(stripped, baseline)
        self.assertEqual(pcc['SEED'], 42)
        self.assertEqual(pcc['MODEL']['NAME'], 'resnet50')
        self.assertTrue(pcc['MODEL']['PART_ATTENTION'])
        self.assertEqual(pcc['MODEL']['PART_ATTENTION_PARTS'], 6)
        self.assertEqual(pcc['MODEL']['CROSS_CAMERA_POSITIVE_LAMBDA'], 0.3)
        self.assertEqual(pcc['MODEL']['PCC_LAMBDA'], 0.1)
        self.assertNotIn('MULTI_GRANULARITY_LOCAL', pcc['MODEL'])

    def test_pcc_off_keeps_legacy_training_return_and_inference_descriptor(self):
        torch.manual_seed(11)
        model = Baseline(
            2, 1, '', 'bnneck', 'after', 'resnet50', 'none',
            part_attention=True, part_attention_parts=6,
            part_correspondence_consistency=False, pcc_parts=6,
        )
        images = torch.randn(2, 3, 256, 128)
        model.train()
        legacy_output = model(images)
        self.assertEqual(len(legacy_output), 2)

        model.part_correspondence_consistency = True
        pcc_output = model(images)
        self.assertEqual(len(pcc_output), 3)
        self.assertEqual(tuple(pcc_output[2].shape), (2, 6, 2048))
        self.assertTrue(torch.allclose(legacy_output[0], pcc_output[0], atol=1e-6))
        self.assertTrue(torch.allclose(legacy_output[1], pcc_output[1], atol=1e-6))

        model.eval()
        bottleneck_values = {}

        def capture_bottleneck(_module, inputs, output):
            bottleneck_values['before'] = inputs[0].detach().clone()
            bottleneck_values['after'] = output.detach().clone()

        hook = model.bottleneck.register_forward_hook(capture_bottleneck)
        with torch.no_grad():
            inference = model(images)
            model.neck_feat = 'before'
            inference_before = model(images)
        hook.remove()
        self.assertEqual(tuple(inference.shape), (2, 2048))
        self.assertTrue(torch.allclose(inference, bottleneck_values['after']))
        self.assertTrue(torch.allclose(
            inference_before, bottleneck_values['before']
        ))


if __name__ == '__main__':
    unittest.main()
