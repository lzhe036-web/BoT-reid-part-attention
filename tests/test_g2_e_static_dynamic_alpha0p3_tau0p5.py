import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import yaml

from config import cfg
from modeling import build_model
from tools.verify_g2_e_static_dynamic_alpha0p3_tau0p5_protocol import ProtocolError, verify


REPO_ROOT = Path(__file__).resolve().parents[1]
G2_A_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_global_local_tau0p5_autodl.yml"
)
E_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_e_static_dynamic_alpha0p3_tau0p5_autodl.yml"
)


class CountingBackbone(nn.Module):
    def __init__(self):
        super(CountingBackbone, self).__init__()
        self.projection = nn.Conv2d(3, 2048, 1, bias=False)
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        return F.adaptive_avg_pool2d(self.projection(inputs), (7, 3))


def configuration(e_variant=False, alpha=.3, residual=True):
    result = cfg.clone()
    result.merge_from_file(str(E_CONFIG if e_variant else G2_A_CONFIG))
    result.defrost()
    result.MODEL.PRETRAIN_CHOICE = "none"
    result.MODEL.PRETRAIN_PATH = ""
    result.MODEL.IF_LABELSMOOTH = "off"
    if e_variant:
        result.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL = residual
        result.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA = alpha
    result.freeze()
    return result


def model(e_variant=False, alpha=.3, residual=True):
    network = build_model(configuration(e_variant, alpha, residual), 3)
    network.base = CountingBackbone()
    return network


def copy_state(source, target):
    target_state = target.state_dict()
    for key, value in source.state_dict().items():
        if key in target_state:
            target_state[key].copy_(value)


class G2EStaticDynamicTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_protocol_allows_only_residual_switch_alpha_and_output(self):
        record = verify(G2_A_CONFIG, E_CONFIG)
        self.assertEqual(record["status"], "verified")
        self.assertEqual(record["allowed_differences"], [
            "MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA",
            "MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL", "OUTPUT_DIR",
        ])

    def test_protocol_rejects_tau_or_training_change(self):
        candidate = copy.deepcopy(yaml.safe_load(E_CONFIG.read_text(encoding="utf-8")))
        candidate["MODEL"]["MULTI_GRANULARITY_GATING_TAU"] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.yml"
            path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "GATING_TAU"):
                verify(G2_A_CONFIG, path)

    def test_disabled_residual_restores_exact_g2a_behavior(self):
        baseline = model(False)
        disabled = model(True, residual=False)
        copy_state(baseline, disabled)
        baseline.eval(); disabled.eval()
        values = torch.randn(2, 3, 8, 4)
        with torch.no_grad():
            self.assertTrue(torch.equal(baseline(values), disabled(values)))

    def test_alpha_zero_is_static_concat_not_pure_dynamic_fusion(self):
        static_config = configuration(False).clone()
        static_config.defrost()
        static_config.MODEL.MULTI_GRANULARITY_DYNAMIC_GATING = False
        static_config.freeze()
        static = build_model(static_config, 3)
        static.base = CountingBackbone()
        alpha_zero = model(True, alpha=0.0)
        copy_state(static, alpha_zero)
        static.eval(); alpha_zero.eval()
        values = torch.randn(2, 3, 8, 4)
        with torch.no_grad():
            self.assertTrue(torch.equal(static(values), alpha_zero(values)))

    def test_alpha_point_three_matches_formula_and_preserves_global_block(self):
        network = model(True, alpha=.3)
        with torch.no_grad():
            nn.init.normal_(network.multi_granularity_dynamic_gate.controller.weight, std=.01)
        network.eval(); network.neck_feat = "before"
        values = torch.randn(2, 3, 8, 4)
        with torch.no_grad():
            fmap = network.base(values)
            global_feat = network.gap(fmap).view(2, -1)
            local = network.multi_granularity_part_head(fmap)
            _logits, probabilities, weights = network.dynamic_gating_values(global_feat, local)
            expected = torch.cat((global_feat,) + tuple(
                value * (1.0 + .3 * weights[:, index:index + 1])
                for index, value in enumerate(local)
            ), dim=1)
            actual = network(values)
        self.assertEqual(tuple(actual.shape), (2, 2816))
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.equal(actual[:, :2048], global_feat))
        evidence = network._last_dynamic_gating
        self.assertTrue(torch.allclose(evidence["probabilities"].sum(1), torch.ones(2)))
        self.assertTrue(torch.allclose(evidence["weights"].sum(1), torch.full((2,), 3.0)))
        self.assertTrue(torch.allclose(evidence["residual_coefficients"], .3 * evidence["weights"]))
        self.assertTrue(torch.allclose(evidence["local_coefficients"], 1.0 + .3 * evidence["weights"]))
        self.assertEqual(network.base.forward_calls, 2)

    def test_zero_initialized_controller_means_local_coefficient_1p3(self):
        network = model(True, alpha=.3)
        network.eval(); network.neck_feat = "before"
        values = torch.randn(2, 3, 8, 4)
        with torch.no_grad():
            fmap = network.base(values)
            global_feat = network.gap(fmap).view(2, -1)
            local = network.multi_granularity_part_head(fmap)
            actual = network(values)
        expected = torch.cat((global_feat,) + tuple(1.3 * value for value in local), dim=1)
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-6, atol=1e-6))

    def test_gradients_are_connected_and_checkpoint_round_trip_is_strict(self):
        source = model(True, alpha=.3)
        with torch.no_grad():
            nn.init.normal_(source.multi_granularity_dynamic_gate.controller.weight, std=.01)
        source.train()
        scores, descriptor = source(torch.randn(4, 3, 8, 4))
        (scores.square().mean() + descriptor.square().mean()).backward()
        for parameter in (source.base.projection.weight,
                          next(source.multi_granularity_part_head.parameters()),
                          source.multi_granularity_dynamic_gate.controller.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        target = model(True, alpha=.3)
        target.load_state_dict(source.state_dict(), strict=True)
        self.assertTrue(torch.equal(
            source.multi_granularity_dynamic_gate.controller.weight,
            target.multi_granularity_dynamic_gate.controller.weight,
        ))

    def test_invalid_alpha_and_missing_gate_fail_closed(self):
        for alpha in (True, -0.1, float("nan"), float("inf")):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    model(True, alpha=alpha)
        invalid = configuration(True).clone()
        invalid.defrost(); invalid.MODEL.MULTI_GRANULARITY_DYNAMIC_GATING = False; invalid.freeze()
        with self.assertRaisesRegex(ValueError, "requires"):
            build_model(invalid, 3)


if __name__ == "__main__":
    unittest.main()
