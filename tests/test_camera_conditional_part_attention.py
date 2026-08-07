# encoding: utf-8
"""Synthetic tests for Parallel A camera-conditional part attention."""

from __future__ import absolute_import

import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
import yaml

from config import cfg
from engine.inference import create_supervised_evaluator
from modeling import build_model
from modeling.baseline import Baseline, PartAttentionHead
from tools.analyze_distance_distributions import forward_model_with_camids
from tools.profile_efficiency import _forward_model, _profile_camids
from tools.run_experiment import _model_manifest
from utils.experiment_recording import config_modules, experiment_identity


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_CONFIG = (
    REPO_ROOT / "configs" /
    "softmax_triplet_c2l03_camera_conditional_part_attention_control_autodl.yml"
)
CONDPA_CONFIG = (
    REPO_ROOT / "configs" /
    "softmax_triplet_c2l03_camera_conditional_part_attention_autodl.yml"
)


class CountingBackbone(nn.Module):
    def __init__(self, height=12, width=4):
        super(CountingBackbone, self).__init__()
        self.projection = nn.Conv2d(3, 2048, kernel_size=1, bias=False)
        self.height = height
        self.width = width

    def forward(self, inputs):
        features = self.projection(inputs)
        return torch.nn.functional.adaptive_avg_pool2d(
            features, (self.height, self.width)
        )


def build_synthetic_baseline(enabled=True):
    model = Baseline(
        num_classes=8,
        last_stride=1,
        model_path="",
        neck="bnneck",
        neck_feat="after",
        model_name="resnet50",
        pretrain_choice="none",
        part_attention=True,
        part_attention_parts=6,
        camera_conditional_part_attention=enabled,
        num_cameras=6,
        multi_granularity_local=False,
    )
    model.base = CountingBackbone()
    return model


class CaptureCamidsModel(nn.Module):
    def __init__(self, conditional=True):
        super(CaptureCamidsModel, self).__init__()
        self.camera_conditional_part_attention = conditional
        self.last_camids = None

    def forward(self, images, camids=None):
        self.last_camids = camids
        return images.flatten(1)


class CameraConditionalPartAttentionTest(unittest.TestCase):
    def test_attention_weights_sum_to_one_along_parts(self):
        head = PartAttentionHead(
            8, num_parts=6, camera_conditional=True, num_cameras=6
        )
        _feature, weights = head(
            torch.randn(4, 8, 12, 3),
            camids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            return_attention=True,
        )
        torch.testing.assert_close(
            weights.sum(dim=1), torch.ones(4), rtol=0, atol=1e-6
        )

    def test_zero_camera_embedding_matches_original_attention(self):
        original = PartAttentionHead(8, num_parts=6)
        conditional = PartAttentionHead(
            8, num_parts=6, camera_conditional=True, num_cameras=6
        )
        conditional.attention.load_state_dict(original.attention.state_dict())
        inputs = torch.randn(4, 8, 12, 3)
        expected, expected_weights = original(inputs, return_attention=True)
        actual, actual_weights = conditional(
            inputs,
            camids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            return_attention=True,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)

    def test_nonuniform_embeddings_change_weights_for_same_image(self):
        head = PartAttentionHead(
            8, num_parts=6, camera_conditional=True, num_cameras=6
        )
        with torch.no_grad():
            head.camera_embedding.weight[0, 0] = 2.0
            head.camera_embedding.weight[1, 1] = 2.0
        image = torch.randn(1, 8, 12, 3).repeat(2, 1, 1, 1)
        _feature, weights = head(
            image,
            camids=torch.tensor([0, 1], dtype=torch.long),
            return_attention=True,
        )
        self.assertFalse(torch.equal(weights[0], weights[1]))
        self.assertGreater(weights[0, 0], weights[0, 1])
        self.assertGreater(weights[1, 1], weights[1, 0])

    def test_embedding_shape_and_market_parameter_count_are_exact(self):
        head = PartAttentionHead(
            8, num_parts=6, camera_conditional=True, num_cameras=6
        )
        self.assertIsInstance(head.camera_embedding, nn.Embedding)
        self.assertEqual(tuple(head.camera_embedding.weight.shape), (6, 6))
        self.assertEqual(head.camera_embedding.weight.numel(), 36)
        self.assertTrue(torch.equal(
            head.camera_embedding.weight,
            torch.zeros_like(head.camera_embedding.weight),
        ))

    def test_gradients_reach_embedding_and_content_attention(self):
        head = PartAttentionHead(
            8, num_parts=6, camera_conditional=True, num_cameras=6
        )
        outputs = head(
            torch.randn(4, 8, 12, 3, requires_grad=True),
            camids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        )
        outputs.square().mean().backward()
        self.assertIsNotNone(head.camera_embedding.weight.grad)
        self.assertIsNotNone(head.attention.weight.grad)
        self.assertGreater(head.camera_embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.attention.weight.grad.abs().sum().item(), 0)

    def test_disabled_path_needs_no_camids_and_has_no_embedding(self):
        head = PartAttentionHead(8, num_parts=6, camera_conditional=False)
        outputs = head(torch.randn(2, 8, 12, 3))
        self.assertEqual(tuple(outputs.shape), (2, 8))
        self.assertFalse(hasattr(head, "camera_embedding"))
        model = build_synthetic_baseline(enabled=False).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.randn(2, 3, 12, 4)).shape), (2, 2048))

    def test_full_model_parameter_increase_is_only_36(self):
        baseline = build_synthetic_baseline(enabled=False)
        condpa = build_synthetic_baseline(enabled=True)
        baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
        condpa_params = sum(parameter.numel() for parameter in condpa.parameters())
        self.assertEqual(condpa_params - baseline_params, 36)

    def test_invalid_camids_fail_clearly(self):
        head = PartAttentionHead(
            8, num_parts=6, camera_conditional=True, num_cameras=6
        )
        inputs = torch.randn(2, 8, 12, 3)
        invalid = (
            (None, ValueError, "required"),
            (torch.tensor([0, 1], dtype=torch.int32), TypeError, "torch.long"),
            (torch.tensor([0], dtype=torch.long), ValueError, "batch size"),
            (torch.tensor([0, 6], dtype=torch.long), ValueError, "\[0, 6\)"),
            (torch.tensor([-1, 0], dtype=torch.long), ValueError, "\[0, 6\)"),
        )
        for camids, error_type, message in invalid:
            with self.subTest(camids=camids):
                with self.assertRaisesRegex(error_type, message):
                    head(inputs, camids=camids)

    def test_train_eval_cpu_and_dataparallel_keyword_call(self):
        model = build_synthetic_baseline(enabled=True)
        inputs = torch.randn(4, 3, 12, 4)
        camids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        model.train()
        scores, descriptor = model(inputs, camids=camids)
        self.assertEqual(tuple(scores.shape), (4, 8))
        self.assertEqual(tuple(descriptor.shape), (4, 2048))
        model.eval()
        with torch.no_grad():
            feature = nn.DataParallel(model)(inputs, camids=camids)
        self.assertEqual(tuple(feature.shape), (4, 2048))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_call_uses_long_camids_on_model_device(self):
        head = PartAttentionHead(
            8, num_parts=6, camera_conditional=True, num_cameras=6
        ).cuda()
        output = head(
            torch.randn(2, 8, 12, 3, device="cuda"),
            camids=torch.tensor([0, 1], dtype=torch.long, device="cuda"),
        )
        self.assertEqual(output.device.type, "cuda")

    def test_condpa_checkpoint_strict_loads(self):
        source = build_synthetic_baseline(enabled=True)
        target = build_synthetic_baseline(enabled=True)
        incompatibility = target.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(incompatibility.missing_keys, [])
        self.assertEqual(incompatibility.unexpected_keys, [])

    def test_inference_keeps_raw_camids_for_metric_and_passes_long_tensor(self):
        model = CaptureCamidsModel()
        evaluator = create_supervised_evaluator(model, metrics={}, device=None)
        raw_camids = (0, 1)
        output = evaluator._process_function(
            evaluator,
            (torch.randn(2, 3, 4, 2), (10, 11), raw_camids),
        )
        self.assertEqual(output[2], raw_camids)
        self.assertEqual(model.last_camids.dtype, torch.long)
        self.assertEqual(model.last_camids.tolist(), [0, 1])

    def test_distance_and_efficiency_helpers_supply_camids(self):
        model = CaptureCamidsModel()
        images = torch.randn(2, 3, 4, 2)
        forward_model_with_camids(model, images, (1, 2), torch.device("cpu"))
        self.assertEqual(model.last_camids.dtype, torch.long)
        self.assertEqual(model.last_camids.tolist(), [1, 2])
        profile_camids = _profile_camids(model, 2, torch.device("cpu"))
        _forward_model(model, images, camids=profile_camids)
        self.assertEqual(model.last_camids.tolist(), [0, 0])

    def test_control_and_condpa_configs_have_exactly_two_differences(self):
        control = yaml.safe_load(CONTROL_CONFIG.read_text(encoding="utf-8"))
        condpa = yaml.safe_load(CONDPA_CONFIG.read_text(encoding="utf-8"))
        self.assertFalse(control["MODEL"]["CAMERA_CONDITIONAL_PART_ATTENTION"])
        self.assertTrue(condpa["MODEL"]["CAMERA_CONDITIONAL_PART_ATTENTION"])
        self.assertFalse(control["MODEL"]["MULTI_GRANULARITY_LOCAL"])
        self.assertFalse(condpa["MODEL"]["MULTI_GRANULARITY_LOCAL"])
        control["MODEL"].pop("CAMERA_CONDITIONAL_PART_ATTENTION")
        condpa["MODEL"].pop("CAMERA_CONDITIONAL_PART_ATTENTION")
        control.pop("OUTPUT_DIR")
        condpa.pop("OUTPUT_DIR")
        self.assertEqual(control, condpa)

    def test_model_builder_receives_dataset_camera_count(self):
        local_cfg = cfg.clone()
        local_cfg.merge_from_file(str(CONDPA_CONFIG))
        with mock.patch("modeling.Baseline") as baseline:
            build_model(local_cfg, num_classes=751, num_cameras=6)
        self.assertEqual(baseline.call_args.kwargs["num_cameras"], 6)

    def test_recording_identity_and_manifest_are_condpa_specific(self):
        configuration = yaml.safe_load(CONDPA_CONFIG.read_text(encoding="utf-8"))
        identity = experiment_identity(configuration)
        self.assertEqual(
            identity["method"],
            "C2-L03 + Camera-Conditional Part Attention",
        )
        self.assertTrue(identity["modules"]["camera_conditional_part_attention"])
        self.assertFalse(identity["modules"]["multi_granularity"])
        manifest = _model_manifest(configuration, num_cameras=6)
        self.assertEqual(manifest["camera_count"], 6)
        self.assertEqual(manifest["part_count"], 6)
        self.assertEqual(manifest["camera_embedding_params"], 36)
        self.assertTrue(manifest["inference_uses_camid"])

    def test_legacy_config_defaults_condpa_to_false(self):
        modules = config_modules({"MODEL": {"NAME": "resnet50"}})
        self.assertFalse(modules["camera_conditional_part_attention"])


if __name__ == "__main__":
    unittest.main()
