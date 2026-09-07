# encoding: utf-8
"""Regression tests for the G2-C MLP controller experiment variable."""

from __future__ import absolute_import

import unittest

import torch
from torch import nn

from config import cfg
from modeling import build_model
from modeling.baseline import MultiGranularityDynamicGate
from tools.analyze_g2_global_local_gating import _block_rows
from tools.verify_g2_c_hidden256_tau0p5_protocol import verify


class G2CHidden256Tau05Test(unittest.TestCase):
    def _mlp_gate(self):
        return MultiGranularityDynamicGate(
            2048, 3, temperature=0.5, gating_input="concat_global_local",
            local_feature_dim=256, controller="mlp", controller_hidden_dim=256,
        )

    def test_default_linear_controller_preserves_historical_keys(self):
        gate = MultiGranularityDynamicGate(
            2048, 3, gating_input="concat_global_local", local_feature_dim=256,
        )
        self.assertIsInstance(gate.controller, nn.Linear)
        self.assertEqual(tuple(gate.controller.weight.shape), (3, 2816))
        self.assertEqual(gate.controller_architecture, "linear")
        self.assertEqual(gate.controller_hidden_dim, 0)
        self.assertEqual(
            set(gate.state_dict()), {"controller.weight", "controller.bias"}
        )

    def test_mlp_shape_input_and_neutral_initial_gate(self):
        gate = self._mlp_gate()
        self.assertIsInstance(gate.controller, nn.Sequential)
        self.assertEqual(tuple(gate.controller[0].weight.shape), (256, 2816))
        self.assertIsInstance(gate.controller[1], nn.ReLU)
        self.assertEqual(tuple(gate.controller[2].weight.shape), (3, 256))
        self.assertFalse(torch.equal(gate.controller[0].weight, torch.zeros_like(gate.controller[0].weight)))
        self.assertEqual(gate.controller[0].bias.count_nonzero().item(), 0)
        self.assertEqual(gate.controller[2].weight.count_nonzero().item(), 0)
        self.assertEqual(gate.controller[2].bias.count_nonzero().item(), 0)

        g = torch.randn(4, 2048)
        scales = tuple(torch.randn(4, 256) for _ in range(3))
        controller_input = gate.controller_input(g, scales)
        logits, probabilities, weights = gate(g, scales)
        self.assertEqual(tuple(controller_input.shape), (4, 2816))
        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertTrue(torch.allclose(logits, torch.zeros_like(logits)))
        self.assertTrue(torch.allclose(probabilities, torch.full_like(probabilities, 1.0 / 3.0)))
        self.assertTrue(torch.allclose(weights, torch.ones_like(weights)))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(4)))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.full((4,), 3.0)))

    def test_mlp_hidden_layer_receives_gradient_after_output_update(self):
        torch.manual_seed(7)
        gate = self._mlp_gate()
        optimizer = torch.optim.SGD(gate.parameters(), lr=0.1)
        g = torch.randn(3, 2048)
        scales = tuple(torch.randn(3, 256) for _ in range(3))

        optimizer.zero_grad()
        logits, _probabilities, _weights = gate(g, scales)
        logits[:, 0].sum().backward()
        optimizer.step()
        self.assertGreater(gate.controller[2].weight.abs().sum().item(), 0.0)

        optimizer.zero_grad()
        logits, probabilities, weights = gate(g, scales)
        (logits.square().mean() + probabilities.square().mean() + weights.square().mean()).backward()
        self.assertIsNotNone(gate.controller[0].weight.grad)
        self.assertTrue(torch.isfinite(gate.controller[0].weight.grad).all())
        self.assertGreater(gate.controller[0].weight.grad.abs().sum().item(), 0.0)

    def test_fusion_contract_and_z6_input_are_unchanged(self):
        gate = self._mlp_gate()
        g = torch.randn(2, 2048)
        z2, z4, z6 = (torch.randn(2, 256) for _ in range(3))
        _logits, p, w = gate(g, (z2, z4, z6))
        descriptor = torch.cat((g, w[:, 0:1] * z2, w[:, 1:2] * z4, w[:, 2:3] * z6), dim=1)
        self.assertEqual(tuple(descriptor.shape), (2, 2816))
        self.assertTrue(torch.allclose(p.sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(w.sum(dim=1), torch.full((2,), 3.0), atol=1e-6))

    def test_mlp_controller_block_proxy_is_explicitly_not_applicable(self):
        configuration = cfg.clone()
        configuration.merge_from_file(
            "configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_c_hidden256_tau0p5_autodl.yml"
        )
        gate = self._mlp_gate()
        state = {
            "multi_granularity_dynamic_gate.{}".format(key): value
            for key, value in gate.state_dict().items()
        }
        rows, boundaries = _block_rows(state, configuration, "a" * 64)
        self.assertIsNone(boundaries)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["applicability"], "not_applicable")
        self.assertEqual(rows[0]["controller_architecture"], "mlp")
        self.assertEqual(rows[0]["rms_weight"], "not_applicable")

    def test_mlp_rng_isolation_preserves_shared_initialization(self):
        common = dict(
            num_classes=7, last_stride=1, model_path="", neck="bnneck",
            neck_feat="after", model_name="resnet50", pretrain_choice="none",
            multi_granularity_part=True, multi_granularity_dynamic_gating=True,
            multi_granularity_gating_input="concat_global_local",
            multi_granularity_part_scales=(2, 4, 6), multi_granularity_part_dim=256,
        )
        torch.manual_seed(42)
        original = build_model_from_kwargs(common)
        torch.manual_seed(42)
        candidate = build_model_from_kwargs(dict(
            common, multi_granularity_gating_tau=0.5,
            multi_granularity_gating_controller="mlp",
            multi_granularity_gating_hidden_dim=256,
        ))
        original_state = original.state_dict()
        candidate_state = candidate.state_dict()
        shared = [
            key for key in original_state
            if "multi_granularity_dynamic_gate" not in key
        ]
        self.assertTrue(shared)
        for key in shared:
            self.assertTrue(torch.equal(original_state[key], candidate_state[key]), key)

    def test_typed_config_builds_mlp_and_protocol_is_exact(self):
        configuration = cfg.clone()
        configuration.merge_from_file(
            "configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_c_hidden256_tau0p5_autodl.yml"
        )
        configuration.defrost()
        configuration.MODEL.PRETRAIN_CHOICE = "none"
        configuration.MODEL.PRETRAIN_PATH = ""
        configuration.freeze()
        model = build_model(configuration, num_classes=7)
        gate = model.multi_granularity_dynamic_gate
        self.assertEqual(model.feature_dim, 2816)
        self.assertEqual(gate.controller_architecture, "mlp")
        self.assertEqual(gate.controller_hidden_dim, 256)
        self.assertEqual(gate.temperature, 0.5)
        result = verify(
            "configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_autodl.yml",
            "configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_c_hidden256_tau0p5_autodl.yml",
        )
        self.assertEqual(result["status"], "verified")


def build_model_from_kwargs(kwargs):
    from modeling.baseline import Baseline
    return Baseline(**kwargs)


if __name__ == "__main__":
    unittest.main()
