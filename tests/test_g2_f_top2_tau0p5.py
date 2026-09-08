import math
import unittest
from pathlib import Path

import torch

from config import cfg
from modeling import build_model
from modeling.baseline import MultiGranularityDynamicGate
from tools.verify_g2_f_top2_tau0p5_protocol import ProtocolError, verify


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau0p5_autodl.yml"
TOP2_CONFIG = REPO_ROOT / "configs" / "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_f_top2_tau0p5_autodl.yml"


def gate():
    return MultiGranularityDynamicGate(
        4, 3, temperature=.5, gating_input="global", sparsification="topk", topk=2,
        tie_break="scale_order",
    )


class G2FTop2Test(unittest.TestCase):
    def _with_bias(self, values):
        result = gate()
        with torch.no_grad():
            result.controller.bias.copy_(torch.tensor(values, dtype=torch.float32))
        return result

    def test_config_protocol_allows_only_top2_identity_and_output(self):
        report = verify(BASE_CONFIG, TOP2_CONFIG)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(set(report["allowed_differences"]), {
            "MODEL.MULTI_GRANULARITY_GATING_SPARSIFICATION",
            "MODEL.MULTI_GRANULARITY_GATING_TOPK",
            "MODEL.MULTI_GRANULARITY_GATING_TIE_BREAK", "OUTPUT_DIR",
        })

    def test_model_shape_and_controller_contract_are_unchanged(self):
        configuration = cfg.clone()
        configuration.merge_from_file(str(TOP2_CONFIG))
        configuration.defrost(); configuration.MODEL.PRETRAIN_CHOICE = "none"; configuration.MODEL.PRETRAIN_PATH = ""; configuration.freeze()
        network = build_model(configuration, 751)
        self.assertEqual(network.feature_dim, 2816)
        controller = network.multi_granularity_dynamic_gate.controller
        self.assertEqual((controller.in_features, controller.out_features), (2816, 3))
        self.assertEqual(sum(parameter.numel() for parameter in controller.parameters()), 8451)

    def test_top2_controller_does_not_change_shared_initialization_sequence(self):
        def build(path):
            configuration = cfg.clone()
            configuration.merge_from_file(str(path))
            configuration.defrost(); configuration.MODEL.PRETRAIN_CHOICE = "none"; configuration.MODEL.PRETRAIN_PATH = ""; configuration.freeze()
            return build_model(configuration, 751)
        torch.manual_seed(42)
        baseline = build(BASE_CONFIG)
        torch.manual_seed(42)
        candidate = build(TOP2_CONFIG)
        baseline_state, candidate_state = baseline.state_dict(), candidate.state_dict()
        for name, value in baseline_state.items():
            if "multi_granularity_dynamic_gate" not in name:
                self.assertTrue(torch.equal(value, candidate_state[name]), name)

    def test_known_distinct_logits_keep_two_largest_and_match_formula(self):
        network = self._with_bias((2.0, 1.0, -3.0))
        logits, sparse, weights = network(torch.zeros(1, 4))
        dense = torch.softmax(logits / .5, dim=1)
        expected = dense * torch.tensor([[1., 1., 0.]])
        expected = expected / expected.sum(1, keepdim=True)
        self.assertTrue(torch.allclose(sparse, expected))
        self.assertTrue(torch.equal(network._last_sparsification["selection_mask"], torch.tensor([[True, True, False]])))
        self.assertTrue(torch.equal(sparse[:, 2], torch.zeros(1)))
        self.assertTrue(torch.allclose(weights, 3.0 * sparse))

    def test_zero_initialized_tie_selects_k2_k4_and_boundary_is_tied(self):
        network = gate()
        _logits, sparse, weights = network(torch.zeros(2, 4))
        self.assertTrue(torch.equal(sparse, torch.tensor([[.5, .5, 0.], [.5, .5, 0.]])))
        self.assertTrue(torch.equal(weights, torch.tensor([[1.5, 1.5, 0.], [1.5, 1.5, 0.]])))
        self.assertTrue(torch.equal(network._last_sparsification["selection_mask"], torch.tensor([[True, True, False], [True, True, False]])))
        self.assertTrue(network._last_sparsification["selection_boundary_tie"].all())

    def test_second_third_tie_uses_scale_order(self):
        network = self._with_bias((3.0, 1.0, 1.0))
        _logits, sparse, _weights = network(torch.zeros(1, 4))
        self.assertTrue(torch.equal(network._last_sparsification["selection_mask"], torch.tensor([[True, True, False]])))
        self.assertEqual(float(sparse[0, 2].detach()), 0.0)
        self.assertTrue(bool(network._last_sparsification["selection_boundary_tie"][0]))

    def test_mask_and_sums_are_exactly_the_declared_contract(self):
        network = self._with_bias((.1, -.3, 2.1))
        _logits, sparse, weights = network(torch.randn(7, 4))
        mask = network._last_sparsification["selection_mask"]
        self.assertTrue(torch.equal(mask.sum(1), torch.full((7,), 2)))
        self.assertTrue(torch.equal(sparse.masked_select(~mask), torch.zeros_like(sparse.masked_select(~mask))))
        self.assertTrue(torch.allclose(sparse.sum(1), torch.ones(7)))
        self.assertTrue(torch.allclose(weights.sum(1), torch.full((7,), 3.0)))

    def test_selected_logits_retain_finite_gradient_for_asymmetric_loss(self):
        network = self._with_bias((1.5, .2, -1.0))
        network.controller.bias.requires_grad_(True)
        _logits, sparse, weights = network(torch.zeros(1, 4))
        loss = 1.7 * sparse[0, 0].square() + .23 * weights[0, 1]
        loss.backward()
        gradient = network.controller.bias.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(abs(float(gradient[0])), 0.0)
        self.assertGreater(abs(float(gradient[1])), 0.0)
        self.assertEqual(float(gradient[2]), 0.0)

    def test_selected_gate_path_keeps_local_input_gradient_connected(self):
        network = MultiGranularityDynamicGate(
            4, 3, temperature=.5, gating_input="concat_global_local", local_feature_dim=2,
            sparsification="topk", topk=2,
        )
        with torch.no_grad():
            network.controller.weight[0, 4] = 1.0
            network.controller.weight[1, 6] = .5
            network.controller.bias.copy_(torch.tensor((.8, .1, -1.0)))
        global_feature = torch.zeros(1, 4, requires_grad=True)
        local = tuple(torch.randn(1, 2, requires_grad=True) for _ in range(3))
        _logits, probabilities, _weights = network(global_feature, local)
        (1.3 * probabilities[0, 0] + .4 * probabilities[0, 1]).backward()
        self.assertIsNotNone(local[0].grad)
        self.assertTrue(torch.isfinite(local[0].grad).all())

    def test_extreme_finite_logits_keep_finite_values_and_gradients(self):
        network = self._with_bias((1000.0, -1000.0, 0.0))
        _logits, sparse, weights = network(torch.zeros(1, 4))
        mask = network._last_sparsification["selection_mask"]
        self.assertTrue(torch.isfinite(sparse).all() and torch.isfinite(weights).all())
        self.assertTrue(torch.equal(mask.sum(1), torch.tensor([2])))
        (weights[0, 0] * .71 + weights[0, 2] * .19).backward()
        self.assertTrue(torch.isfinite(network.controller.bias.grad).all())

    def test_dense_mode_recovers_original_scaled_softmax_behaviour(self):
        dense = MultiGranularityDynamicGate(4, 3, temperature=.5, sparsification="none", topk=0)
        with torch.no_grad():
            dense.controller.bias.copy_(torch.tensor((1.0, -.5, .3)))
        logits, probabilities, weights = dense(torch.zeros(3, 4))
        self.assertTrue(torch.allclose(probabilities, torch.softmax(logits / .5, dim=1)))
        self.assertTrue(dense._last_sparsification["selection_mask"].all())
        self.assertFalse(dense._last_sparsification["selection_boundary_tie"].any())

    def test_invalid_sparse_configuration_fails_closed(self):
        with self.assertRaises(ValueError):
            MultiGranularityDynamicGate(4, 3, sparsification="topk", topk=3)
        with self.assertRaises(ValueError):
            MultiGranularityDynamicGate(4, 3, sparsification="none", topk=2)
        with self.assertRaises(ValueError):
            MultiGranularityDynamicGate(4, 3, sparsification="topk", topk=2, tie_break="random")


if __name__ == "__main__":
    unittest.main()
