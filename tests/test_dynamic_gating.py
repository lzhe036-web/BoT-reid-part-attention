import math
import unittest
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import yaml

from config import cfg
from layers import make_loss
from modeling import build_model
from modeling.baseline import MultiGranularityDynamicGate
from utils.experiment_recording import validate_dynamic_configuration


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_CONFIG = REPO_ROOT / "configs" / "softmax_triplet_c2_l03_multi_granularity_part_autodl.yml"
DYNAMIC_CONFIG = REPO_ROOT / "configs" / "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_autodl.yml"


class CountingBackbone(nn.Module):
    def __init__(self):
        super(CountingBackbone, self).__init__()
        self.projection = nn.Conv2d(3, 2048, 1, bias=False)
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        return F.adaptive_avg_pool2d(self.projection(inputs), (7, 3))


def configuration(dynamic):
    result = cfg.clone()
    result.merge_from_file(str(DYNAMIC_CONFIG if dynamic else STATIC_CONFIG))
    result.defrost()
    result.MODEL.PRETRAIN_CHOICE = "none"
    result.MODEL.PRETRAIN_PATH = ""
    result.MODEL.IF_LABELSMOOTH = "off"
    result.freeze()
    return result


def model(dynamic, num_classes=3):
    result = build_model(configuration(dynamic), num_classes)
    result.base = CountingBackbone()
    return result


def copy_shared_state(static, dynamic):
    dynamic_state = dynamic.state_dict()
    for key, value in static.state_dict().items():
        if key in dynamic_state:
            dynamic_state[key].copy_(value)


class DynamicGatingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)

    def test_static_and_dynamic_descriptor_shapes_are_2816(self):
        values = torch.randn(2, 3, 8, 4)
        for enabled in (False, True):
            network = model(enabled)
            network.eval()
            with torch.no_grad():
                descriptor = network(values)
            self.assertEqual(tuple(descriptor.shape), (2, 2816))

    def test_logits_probabilities_and_weights_are_per_sample(self):
        gate = MultiGranularityDynamicGate(2048, 3, temperature=1.0)
        logits, probabilities, weights = gate(torch.randn(5, 2048))
        self.assertEqual(tuple(logits.shape), (5, 3))
        self.assertEqual(tuple(probabilities.shape), (5, 3))
        self.assertEqual(tuple(weights.shape), (5, 3))
        self.assertTrue(torch.all(probabilities >= 0))
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(5)))
        self.assertTrue(torch.allclose(weights, 3.0 * probabilities))
        self.assertTrue(torch.allclose(weights.sum(1), torch.full((5,), 3.0)))

    def test_controller_is_zero_initialized(self):
        gate = MultiGranularityDynamicGate(2048, 3)
        self.assertEqual(gate.controller.weight.count_nonzero().item(), 0)
        self.assertEqual(gate.controller.bias.count_nonzero().item(), 0)

    def test_initial_dynamic_descriptor_exactly_matches_static(self):
        static = model(False)
        dynamic = model(True)
        copy_shared_state(static, dynamic)
        values = torch.randn(3, 3, 8, 4)
        static.eval()
        dynamic.eval()
        with torch.no_grad():
            expected = static(values)
            actual = dynamic(values)
        self.assertTrue(torch.equal(actual, expected))

    def test_static_switch_has_parent_state_dict_and_behavior(self):
        static = model(False)
        self.assertFalse(any("dynamic_gate" in key for key in static.state_dict()))
        self.assertFalse(hasattr(static, "multi_granularity_dynamic_gate"))
        static(torch.randn(2, 3, 8, 4))
        self.assertEqual(static.base.forward_calls, 1)

    def test_different_samples_can_receive_different_gates(self):
        gate = MultiGranularityDynamicGate(2048, 3)
        with torch.no_grad():
            gate.controller.weight[0, 0] = 2.0
            gate.controller.weight[1, 0] = -2.0
        inputs = torch.zeros(2, 2048)
        inputs[0, 0] = 1.0
        inputs[1, 0] = -1.0
        probabilities = gate(inputs)[1]
        self.assertFalse(torch.allclose(probabilities[0], probabilities[1]))

    def test_batch_permutation_equivariance_has_no_cross_sample_leakage(self):
        gate = MultiGranularityDynamicGate(2048, 3)
        nn.init.normal_(gate.controller.weight)
        inputs = torch.randn(7, 2048)
        permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
        original = gate(inputs)[1]
        permuted = gate(inputs[permutation])[1]
        self.assertTrue(torch.allclose(permuted, original[permutation]))

    def test_gate_weight_bias_and_shared_features_receive_finite_gradients(self):
        network = model(True, num_classes=2)
        with torch.no_grad():
            nn.init.normal_(network.multi_granularity_dynamic_gate.controller.weight, std=0.01)
        network.train()
        score, descriptor = network(torch.randn(4, 3, 8, 4))
        loss = score.square().mean() + descriptor.square().mean()
        loss.backward()
        for parameter in (
                network.multi_granularity_dynamic_gate.controller.weight,
                network.multi_granularity_dynamic_gate.controller.bias,
                network.base.projection.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(network.base.forward_calls, 1)

    def test_invalid_temperature_fails_closed(self):
        for value in (0.0, -1.0, float("nan"), float("inf"), -float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and greater than zero"):
                    MultiGranularityDynamicGate(2048, 3, temperature=value)

    def test_inference_neck_protocol_and_dimensions_are_unchanged(self):
        network = model(True)
        network.eval()
        values = torch.randn(2, 3, 8, 4)
        with torch.no_grad():
            network.neck_feat = "before"
            before = network(values)
            network.neck_feat = "after"
            after = network(values)
        self.assertEqual(tuple(before.shape), (2, 2816))
        self.assertEqual(tuple(after.shape), (2, 2816))
        self.assertFalse(torch.equal(before, after))

    def test_state_dict_round_trip_preserves_gate(self):
        source = model(True)
        with torch.no_grad():
            nn.init.normal_(source.multi_granularity_dynamic_gate.controller.weight)
            nn.init.normal_(source.multi_granularity_dynamic_gate.controller.bias)
        target = model(True)
        target.load_state_dict(source.state_dict(), strict=True)
        self.assertTrue(torch.equal(
            source.multi_granularity_dynamic_gate.controller.weight,
            target.multi_granularity_dynamic_gate.controller.weight,
        ))

    def test_checkpoint_model_and_optimizer_resume_preserves_controller_state(self):
        source = model(True, num_classes=2)
        optimizer = torch.optim.Adam(source.parameters(), lr=1e-3)
        source.train()
        score, descriptor = source(torch.randn(4, 3, 8, 4))
        (score.square().mean() + descriptor.square().mean()).backward()
        optimizer.step()
        checkpoint = {
            "model": source.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        target = model(True, num_classes=2)
        target_optimizer = torch.optim.Adam(target.parameters(), lr=1e-3)
        target.load_state_dict(checkpoint["model"], strict=True)
        target_optimizer.load_state_dict(checkpoint["optimizer"])
        self.assertTrue(torch.equal(
            source.multi_granularity_dynamic_gate.controller.weight,
            target.multi_granularity_dynamic_gate.controller.weight,
        ))
        self.assertEqual(
            len(optimizer.state_dict()["state"]),
            len(target_optimizer.state_dict()["state"]),
        )

    def test_formal_config_diff_is_only_declared_gating_fields_and_output(self):
        static = yaml.safe_load(STATIC_CONFIG.read_text(encoding="utf-8"))
        dynamic = yaml.safe_load(DYNAMIC_CONFIG.read_text(encoding="utf-8"))
        differences = validate_dynamic_configuration(dynamic, static)
        self.assertEqual(set(differences), {
            "MODEL.MULTI_GRANULARITY_DYNAMIC_GATING",
            "MODEL.MULTI_GRANULARITY_GATING_INPUT",
            "MODEL.MULTI_GRANULARITY_GATING_TAU",
            "MODEL.MULTI_GRANULARITY_GATING_NORMALIZATION",
            "OUTPUT_DIR",
        })

    def test_total_loss_formula_has_no_gating_regularizer(self):
        network = model(True, num_classes=2)
        network.train()
        score, descriptor = network(torch.randn(4, 3, 8, 4))
        targets = torch.tensor([0, 0, 1, 1])
        cameras = torch.tensor([0, 1, 0, 1])
        result = make_loss(configuration(True), 2)(score, descriptor, targets, cameras)
        expected = (
            result["loss_id"] + result["loss_triplet"]
            + 0.3 * result["loss_cross_camera_positive"]
        )
        self.assertTrue(torch.allclose(result["loss_total"], expected))

    def test_gate_configuration_requires_multi_granularity_features(self):
        local = configuration(True).clone()
        local.defrost()
        local.MODEL.MULTI_GRANULARITY_PART = False
        local.freeze()
        with self.assertRaisesRegex(ValueError, "requires"):
            build_model(local, 2)


if __name__ == "__main__":
    unittest.main()
