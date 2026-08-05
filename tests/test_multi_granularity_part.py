import unittest
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from config import cfg
from layers import make_loss
from modeling import build_model
from modeling.baseline import Baseline, MultiGranularityPartHead


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "softmax_triplet_c2_l03_multi_granularity_part_autodl.yml"
)


class CountingBackbone(nn.Module):
    """Small differentiable backbone stub with a ResNet50-compatible output."""

    def __init__(self, output_height=7, output_width=3):
        super().__init__()
        self.projection = nn.Conv2d(3, 2048, kernel_size=1, bias=False)
        self.output_height = output_height
        self.output_width = output_width
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        features = self.projection(inputs)
        return F.adaptive_avg_pool2d(
            features, (self.output_height, self.output_width)
        )


def multi_granularity_cfg():
    local_cfg = cfg.clone()
    local_cfg.defrost()
    local_cfg.MODEL.PRETRAIN_CHOICE = "none"
    local_cfg.MODEL.PRETRAIN_PATH = ""
    local_cfg.MODEL.PART_ATTENTION = False
    local_cfg.MODEL.MULTI_GRANULARITY_PART = True
    local_cfg.MODEL.MULTI_GRANULARITY_PART_SCALES = [2, 4, 6]
    local_cfg.MODEL.MULTI_GRANULARITY_PART_DIM = 256
    local_cfg.MODEL.MULTI_GRANULARITY_PART_AGGREGATION = "mean"
    local_cfg.MODEL.MULTI_GRANULARITY_PART_FUSION = "concat"
    local_cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY = True
    local_cfg.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA = 0.3
    local_cfg.MODEL.CROSS_CAMERA_POSITIVE_MODE = "mean"
    local_cfg.MODEL.IF_LABELSMOOTH = "off"
    local_cfg.DATALOADER.SAMPLER = "softmax_triplet"
    local_cfg.freeze()
    return local_cfg


def model_with_counting_backbone(num_classes=2, output_height=7):
    model = build_model(multi_granularity_cfg(), num_classes=num_classes)
    model.base = CountingBackbone(output_height=output_height)
    return model


class MultiGranularityPartHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_region_counts_and_exact_coverage_for_all_scales(self):
        for scale in (2, 4, 6):
            bounds = MultiGranularityPartHead.region_bounds(12, scale)
            self.assertEqual(len(bounds), scale)
            covered = [index for start, end in bounds for index in range(start, end)]
            self.assertEqual(covered, list(range(12)))
            self.assertTrue(all(start < end for start, end in bounds))

    def test_non_divisible_height_has_no_overlap_omission_or_empty_region(self):
        bounds = MultiGranularityPartHead.region_bounds(7, 4)
        self.assertEqual(bounds, ((0, 1), (1, 3), (3, 5), (5, 7)))
        covered = [index for start, end in bounds for index in range(start, end)]
        self.assertEqual(covered, list(range(7)))
        self.assertEqual(len(covered), len(set(covered)))

    def test_each_scale_returns_projected_256_dimensional_feature(self):
        head = MultiGranularityPartHead(8, [2, 4, 6], projection_dim=256)
        outputs = head(torch.randn(2, 8, 7, 3))
        self.assertEqual(len(outputs), 3)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (2, 256))
        self.assertIsNot(head.projections["2"], head.projections["4"])
        self.assertIsNot(head.projections["4"], head.projections["6"])

    def test_each_scale_is_exact_arithmetic_mean_of_projected_parts(self):
        head = MultiGranularityPartHead(3, [2, 4, 6], projection_dim=3)
        fixed_weights = {
            "2": torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            "4": torch.tensor([
                [0.5, 0.25, -0.25],
                [-0.5, 1.0, 0.5],
                [0.75, -0.25, 1.0],
            ]),
            "6": torch.tensor([
                [1.0, -0.5, 0.25],
                [0.25, 0.75, -0.5],
                [-0.25, 0.5, 1.0],
            ]),
        }
        with torch.no_grad():
            for scale in head.scales:
                projection = head.projections[str(scale)]
                projection[0].weight.copy_(fixed_weights[str(scale)])
                projection[1].weight.copy_(torch.tensor([1.0, 1.5, 0.5]))
                projection[1].bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

        rows = torch.tensor([
            [1.0, 3.0, 2.0], [2.0, 1.0, 4.0],
            [4.0, 2.0, 1.0], [3.0, 5.0, 2.0],
            [6.0, 1.0, 3.0], [2.0, 7.0, 5.0],
            [5.0, 4.0, 8.0], [7.0, 2.0, 6.0],
            [8.0, 6.0, 1.0], [4.0, 9.0, 7.0],
            [9.0, 3.0, 5.0], [6.0, 8.0, 4.0],
        ])
        feature_map = rows.transpose(0, 1).reshape(1, 3, 12, 1)
        actual_by_scale = dict(zip(head.scales, head(feature_map)))

        for scale in head.scales:
            pooled_parts = head.pool_parts(feature_map, scale)
            projected_parts = head.projections[str(scale)](
                pooled_parts.reshape(scale, 3)
            ).reshape(1, scale, 3)
            expected = projected_parts.sum(dim=1) / scale
            self.assertTrue(
                torch.equal(actual_by_scale[scale], expected),
                msg="scale {} must use sum(parts) / K exactly".format(scale),
            )

    def test_fused_dimensions_train_and_eval_interfaces(self):
        model = model_with_counting_backbone(num_classes=3)
        self.assertEqual(model.feature_dim, 2816)
        self.assertEqual(model.bottleneck.num_features, 2816)
        self.assertEqual(model.classifier.in_features, 2816)

        inputs = torch.randn(2, 3, 8, 4)
        model.train()
        score, fused_pre_bn = model(inputs)
        self.assertEqual(tuple(score.shape), (2, 3))
        self.assertEqual(tuple(fused_pre_bn.shape), (2, 2816))

        model.eval()
        with torch.no_grad():
            eval_feature = model(inputs)
        self.assertIsInstance(eval_feature, torch.Tensor)
        self.assertEqual(tuple(eval_feature.shape), (2, 2816))

    def test_before_and_after_neck_semantics_are_preserved(self):
        model = model_with_counting_backbone()
        inputs = torch.randn(2, 3, 8, 4)
        model.eval()
        with torch.no_grad():
            model.neck_feat = "before"
            before = model(inputs)
            model.neck_feat = "after"
            after = model(inputs)
        self.assertEqual(tuple(before.shape), (2, 2816))
        self.assertEqual(tuple(after.shape), (2, 2816))
        self.assertFalse(torch.equal(before, after))

    def test_actual_loss_backward_gradients_and_single_backbone_call(self):
        model = model_with_counting_backbone(num_classes=2)
        model.train()
        inputs = torch.randn(4, 3, 8, 4)
        targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        camids = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        score, fused_pre_bn = model(inputs)
        loss_values = make_loss(multi_granularity_cfg(), num_classes=2)(
            score, fused_pre_bn, targets, camids
        )
        loss = loss_values["loss_total"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertEqual(model.base.forward_calls, 1)
        gradients = [model.base.projection.weight.grad]
        gradients.extend(
            model.multi_granularity_part_head.projections[str(scale)][0].weight.grad
            for scale in (2, 4, 6)
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_legacy_fixed_k6_behavior_is_unchanged_when_new_switch_is_off(self):
        model = Baseline(
            num_classes=3,
            last_stride=1,
            model_path="",
            neck="bnneck",
            neck_feat="after",
            model_name="resnet50",
            pretrain_choice="none",
            part_attention=True,
            part_attention_parts=6,
            multi_granularity_part=False,
        )
        model.base = CountingBackbone(output_height=7)
        self.assertEqual(model.feature_dim, 2048)
        self.assertEqual(model.bottleneck.num_features, 2048)
        self.assertEqual(model.classifier.in_features, 2048)

        model.train()
        score, feature = model(torch.randn(2, 3, 8, 4))
        self.assertEqual(tuple(score.shape), (2, 3))
        self.assertEqual(tuple(feature.shape), (2, 2048))
        self.assertEqual(model.base.forward_calls, 1)

    def test_part_attention_switches_are_mutually_exclusive(self):
        local_cfg = multi_granularity_cfg().clone()
        local_cfg.defrost()
        local_cfg.MODEL.PART_ATTENTION = True
        local_cfg.freeze()
        with self.assertRaisesRegex(ValueError, "cannot both be enabled"):
            build_model(local_cfg, num_classes=2)

    def test_invalid_scales_raise_clear_errors(self):
        invalid_cases = (
            ([2, 2, 6], "duplicates"),
            ([4, 2, 6], "ascending"),
            ([0, 2, 6], "positive integers"),
            ([2, 4.0, 6], "positive integers"),
        )
        for scales, message in invalid_cases:
            with self.subTest(scales=scales):
                with self.assertRaisesRegex(ValueError, message):
                    MultiGranularityPartHead(8, scales, projection_dim=256)

    def test_projection_dimension_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than or equal to 2"):
            MultiGranularityPartHead(8, [2, 4, 6], projection_dim=1)

    def test_multi_granularity_center_loss_is_rejected_before_model_build(self):
        local_cfg = multi_granularity_cfg().clone()
        local_cfg.defrost()
        local_cfg.MODEL.IF_WITH_CENTER = "yes"
        local_cfg.freeze()
        with self.assertRaisesRegex(ValueError, "center loss still assumes"):
            build_model(local_cfg, num_classes=2)

    def test_feature_map_height_smaller_than_largest_scale_is_rejected(self):
        head = MultiGranularityPartHead(8, [2, 4, 6], projection_dim=256)
        with self.assertRaisesRegex(ValueError, "height 5.*maximum part scale 6"):
            head(torch.randn(2, 8, 5, 3))

    def test_experiment_config_is_yacs_parseable_and_matches_c2_protocol(self):
        parsed = cfg.clone()
        parsed.merge_from_file(str(EXPERIMENT_CONFIG))
        self.assertFalse(parsed.MODEL.PART_ATTENTION)
        self.assertTrue(parsed.MODEL.MULTI_GRANULARITY_PART)
        self.assertEqual(list(parsed.MODEL.MULTI_GRANULARITY_PART_SCALES), [2, 4, 6])
        self.assertEqual(parsed.MODEL.MULTI_GRANULARITY_PART_DIM, 256)
        self.assertEqual(parsed.MODEL.MULTI_GRANULARITY_PART_AGGREGATION, "mean")
        self.assertEqual(parsed.MODEL.MULTI_GRANULARITY_PART_FUSION, "concat")
        self.assertTrue(parsed.MODEL.CROSS_CAMERA_POSITIVE_ONLY)
        self.assertEqual(parsed.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA, 0.3)
        self.assertEqual(parsed.MODEL.CROSS_CAMERA_POSITIVE_MODE, "mean")
        self.assertEqual(parsed.MODEL.IF_WITH_CENTER, "no")
        self.assertEqual(parsed.SOLVER.MAX_EPOCHS, 120)
        self.assertEqual(parsed.SOLVER.IMS_PER_BATCH, 64)
        self.assertEqual(parsed.SEED, 42)


if __name__ == "__main__":
    unittest.main()
