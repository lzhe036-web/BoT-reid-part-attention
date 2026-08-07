# encoding: utf-8
"""Synthetic tests for the shared-map multi-granularity representation."""

from __future__ import absolute_import

import unittest
from pathlib import Path

import torch
from torch import nn
import yaml

from config import cfg
from layers.triplet_loss import (
    CrossCameraPositiveLoss,
    CrossEntropyLabelSmooth,
    TripletLoss,
)
from modeling.baseline import Baseline
from modeling.multi_granularity_local import (
    MultiGranularityLocalFeature,
    horizontal_part_bounds,
)
from utils.experiment_recording import experiment_identity


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "softmax_triplet_cross_camera_positive_lambda03_autodl.yml"
NEW_CONFIG = REPO_ROOT / "configs" / "softmax_triplet_c2l03_multi_granularity_local_feature_autodl.yml"


def build_synthetic_model(num_classes=8, enabled=True, neck_feat="after"):
    return Baseline(
        num_classes=num_classes,
        last_stride=1,
        model_path="",
        neck="bnneck",
        neck_feat=neck_feat,
        model_name="resnet50",
        pretrain_choice="none",
        part_attention=True,
        part_attention_parts=6,
        multi_granularity_local=enabled,
        multi_granularity_scales=(2, 4, 6),
        multi_granularity_dim=256,
        multi_granularity_aggregation="mean",
    )


class CountingBackbone(nn.Module):
    def __init__(self, height=13, width=5):
        super(CountingBackbone, self).__init__()
        self.height = height
        self.width = width
        self.calls = 0
        self.projection = nn.Conv2d(3, 2048, kernel_size=1, bias=False)

    def forward(self, inputs):
        self.calls += 1
        features = self.projection(inputs)
        return torch.nn.functional.adaptive_avg_pool2d(
            features, (self.height, self.width)
        )


class MultiGranularityLocalTest(unittest.TestCase):
    def test_non_divisible_partitions_cover_every_row_once(self):
        for scale in (2, 4, 6):
            bounds = horizontal_part_bounds(13, scale)
            covered = []
            for start, end in bounds:
                covered.extend(range(start, end))
            self.assertEqual(covered, list(range(13)))
            self.assertEqual(len(bounds), scale)

    def test_scale_projection_is_shared_and_aggregation_is_exact_mean(self):
        module = MultiGranularityLocalFeature(
            8, scales=(2, 4, 6), projection_dim=3, aggregation="mean"
        )
        outputs, details = module(
            torch.randn(2, 8, 13, 5), return_details=True
        )
        self.assertEqual(len(module.projections), 3)
        for output, scale in zip(outputs, module.scales):
            stacked = torch.stack(details[scale]["part_features"], dim=1)
            self.assertTrue(torch.equal(output, stacked.mean(dim=1)))
            self.assertEqual(tuple(output.shape), (2, 3))

    def test_new_module_contains_no_attention_or_gating(self):
        module = MultiGranularityLocalFeature(8)
        module_names = [name.lower() for name, _item in module.named_modules()]
        parameter_names = [name.lower() for name, _item in module.named_parameters()]
        for forbidden in ("attention", "gate", "alpha", "alignment"):
            self.assertFalse(any(forbidden in name for name in module_names))
            self.assertFalse(any(forbidden in name for name in parameter_names))

    def test_backbone_runs_once_and_shapes_are_2816(self):
        model = build_synthetic_model()
        counting_backbone = CountingBackbone()
        model.base = counting_backbone
        model.eval()
        with torch.no_grad():
            feature, trace = model.forward_with_shape_trace(
                torch.randn(2, 3, 20, 8)
            )
        self.assertEqual(counting_backbone.calls, 1)
        self.assertEqual(tuple(feature.shape), (2, 2816))
        self.assertEqual(trace["backbone_feature_map"], (2, 2048, 13, 5))
        self.assertEqual(trace["global_feature"], (2, 2048))
        self.assertEqual(trace["k2_aggregated"], (2, 256))
        self.assertEqual(trace["k4_aggregated"], (2, 256))
        self.assertEqual(trace["k6_aggregated"], (2, 256))
        self.assertEqual(trace["concat_feature"], (2, 2816))
        self.assertEqual(trace["bnneck_feature"], (2, 2816))
        self.assertTrue(trace["baseline_existing_attention"])
        self.assertFalse(trace["new_module_attention"])

    def test_train_forward_c2_losses_and_backward(self):
        model = build_synthetic_model(num_classes=8)
        counting_backbone = CountingBackbone(height=13, width=5)
        model.base = counting_backbone
        model.train()
        scores, descriptor = model(torch.randn(4, 3, 20, 8))
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        camids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        id_loss = CrossEntropyLabelSmooth(
            num_classes=8, use_gpu=False
        )(scores, labels)
        triplet_loss = TripletLoss(margin=0.3)(descriptor, labels)[0]
        cross_loss = CrossCameraPositiveLoss(mode="mean")(
            descriptor, labels, camids
        )
        total = id_loss + triplet_loss + 0.3 * cross_loss
        total.backward()
        self.assertEqual(tuple(scores.shape), (4, 8))
        self.assertEqual(tuple(descriptor.shape), (4, 2816))
        self.assertIsNotNone(
            model.multi_granularity_head.projections["2"].weight.grad
        )
        self.assertIsNotNone(counting_backbone.projection.weight.grad)
        self.assertEqual(counting_backbone.calls, 1)

    def test_neck_feat_before_and_after_keep_existing_semantics(self):
        after = build_synthetic_model(neck_feat="after")
        after.base = CountingBackbone()
        before = build_synthetic_model(neck_feat="before")
        before.base = CountingBackbone()
        after.eval()
        before.eval()
        inputs = torch.randn(2, 3, 20, 8)
        with torch.no_grad():
            self.assertEqual(tuple(after(inputs).shape), (2, 2816))
            self.assertEqual(tuple(before(inputs).shape), (2, 2816))

    def test_disabled_path_preserves_c2l03_parameter_shapes(self):
        baseline = build_synthetic_model(enabled=False)
        self.assertTrue(baseline.part_attention)
        self.assertFalse(baseline.multi_granularity_local)
        self.assertEqual(baseline.descriptor_dim, 2048)
        self.assertEqual(baseline.bottleneck.num_features, 2048)
        self.assertEqual(baseline.classifier.in_features, 2048)

    def test_parameter_increase_matches_only_new_representation(self):
        baseline = build_synthetic_model(enabled=False)
        enhanced = build_synthetic_model(enabled=True)
        baseline_params = sum(item.numel() for item in baseline.parameters())
        enhanced_params = sum(item.numel() for item in enhanced.parameters())
        projection_params = 3 * (2048 * 256 + 256)
        bnneck_growth = 2 * (2816 - 2048)
        classifier_growth = 8 * (2816 - 2048)
        self.assertEqual(
            enhanced_params - baseline_params,
            projection_params + bnneck_growth + classifier_growth,
        )

    def test_formal_config_diff_is_only_module_and_output_dir(self):
        with BASE_CONFIG.open("r", encoding="utf-8") as handle:
            baseline = yaml.safe_load(handle)
        with NEW_CONFIG.open("r", encoding="utf-8") as handle:
            enhanced = yaml.safe_load(handle)
        extra_keys = (
            "MULTI_GRANULARITY_LOCAL", "MULTI_GRANULARITY_SCALES",
            "MULTI_GRANULARITY_DIM", "MULTI_GRANULARITY_AGGREGATION",
        )
        for key in extra_keys:
            enhanced["MODEL"].pop(key)
        baseline.pop("OUTPUT_DIR")
        enhanced.pop("OUTPUT_DIR")
        self.assertEqual(enhanced, baseline)
        self.assertTrue(
            yaml.safe_load(NEW_CONFIG.read_text(encoding="utf-8"))["MODEL"]
            ["PART_ATTENTION"]
        )

    def test_resolved_formal_config_has_expected_single_variable(self):
        local_cfg = cfg.clone()
        local_cfg.merge_from_file(str(NEW_CONFIG))
        self.assertTrue(local_cfg.MODEL.PART_ATTENTION)
        self.assertTrue(local_cfg.MODEL.MULTI_GRANULARITY_LOCAL)
        self.assertEqual(list(local_cfg.MODEL.MULTI_GRANULARITY_SCALES), [2, 4, 6])
        self.assertEqual(local_cfg.MODEL.MULTI_GRANULARITY_DIM, 256)
        self.assertEqual(local_cfg.MODEL.MULTI_GRANULARITY_AGGREGATION, "mean")
        self.assertTrue(local_cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY)
        self.assertEqual(local_cfg.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA, 0.3)
        self.assertEqual(local_cfg.OUTPUT_DIR,
                         "/root/autodl-tmp/experiments/BoT/"
                         "c2l03_multi_granularity_local_market1501")

    def test_experiment_recording_identifies_method_and_module(self):
        configuration = yaml.safe_load(NEW_CONFIG.read_text(encoding="utf-8"))
        identity = experiment_identity(configuration)
        self.assertEqual(
            identity["method"],
            "C2-L03 + Multi-Granularity Local Feature "
            "(Global + K2 + K4 + K6, mean aggregation)",
        )
        self.assertTrue(identity["modules"]["multi_granularity"])
        self.assertTrue(identity["modules"]["cross_camera_positive"])
        self.assertEqual(identity["lambda"], 0.3)


if __name__ == "__main__":
    unittest.main()
