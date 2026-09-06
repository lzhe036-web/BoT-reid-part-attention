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
from tools.verify_g2_e1_static_dynamic_protocol import ProtocolError, verify


REPO_ROOT = Path(__file__).resolve().parents[1]
G2_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_global_local_autodl.yml"
)
E1_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_e1_static_dynamic_alpha0p5_autodl.yml"
)


class CountingBackbone(nn.Module):
    def __init__(self):
        super(CountingBackbone, self).__init__()
        self.projection = nn.Conv2d(3, 2048, 1, bias=False)

    def forward(self, inputs):
        return F.adaptive_avg_pool2d(self.projection(inputs), (7, 3))


def configuration(e1=False, alpha=0.5, residual=True):
    result = cfg.clone()
    result.merge_from_file(str(E1_CONFIG if e1 else G2_CONFIG))
    result.defrost()
    result.MODEL.PRETRAIN_CHOICE = "none"
    result.MODEL.PRETRAIN_PATH = ""
    result.MODEL.IF_LABELSMOOTH = "off"
    if e1:
        result.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL = residual
        result.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA = alpha
    result.freeze()
    return result


def model(e1=False, alpha=0.5, residual=True):
    result = build_model(configuration(e1, alpha, residual), 3)
    result.base = CountingBackbone()
    return result


def copy_shared(source, target):
    target_state = target.state_dict()
    for key, value in source.state_dict().items():
        if key in target_state:
            target_state[key].copy_(value)


class StaticDynamicResidualTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_protocol_diff_is_only_residual_switch_alpha_and_output(self):
        record = verify(G2_CONFIG, E1_CONFIG)
        self.assertEqual(record["status"], "verified")
        self.assertEqual(record["allowed_differences"], [
            "MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA",
            "MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL", "OUTPUT_DIR",
        ])

    def test_protocol_rejects_tau_change(self):
        candidate = copy.deepcopy(yaml.safe_load(E1_CONFIG.read_text(encoding="utf-8")))
        candidate["MODEL"]["MULTI_GRANULARITY_GATING_TAU"] = 0.5
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.yml"
            path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "GATING_TAU"):
                verify(G2_CONFIG, path)

    def test_residual_switch_off_restores_original_g2_output(self):
        g2 = model(False)
        disabled = model(True, alpha=0.5, residual=False)
        copy_shared(g2, disabled)
        g2.eval(); disabled.eval()
        values = torch.randn(2, 3, 8, 4)
        with torch.no_grad():
            self.assertTrue(torch.equal(g2(values), disabled(values)))

    def test_alpha_zero_equals_ungated_static_concatenation(self):
        static = configuration(False).clone()
        static.defrost(); static.MODEL.MULTI_GRANULARITY_DYNAMIC_GATING = False; static.freeze()
        static_model = build_model(static, 3); static_model.base = CountingBackbone()
        alpha_zero = model(True, alpha=0.0, residual=True)
        copy_shared(static_model, alpha_zero)
        static_model.eval(); alpha_zero.eval()
        values = torch.randn(2, 3, 8, 4)
        with torch.no_grad():
            self.assertTrue(torch.equal(static_model(values), alpha_zero(values)))

    def test_alpha_half_matches_stated_formula_and_keeps_2816_dimensions(self):
        network = model(True, alpha=0.5, residual=True)
        with torch.no_grad():
            nn.init.normal_(network.multi_granularity_dynamic_gate.controller.weight, std=0.01)
        network.eval(); network.neck_feat = "before"
        values = torch.randn(2, 3, 8, 4)
        with torch.no_grad():
            fmap = network.base(values)
            global_feat = network.gap(fmap).view(2, -1)
            local = network.multi_granularity_part_head(fmap)
            _logits, probabilities, weights = network.dynamic_gating_values(global_feat, local)
            expected = torch.cat(
                (global_feat,) + tuple(
                    feature * (1.0 + 0.5 * weights[:, index:index + 1])
                    for index, feature in enumerate(local)
                ), dim=1,
            )
            actual = network(values)
        self.assertEqual(tuple(actual.shape), (2, 2816))
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-6, atol=1e-6))
        evidence = network._last_dynamic_gating
        self.assertTrue(torch.allclose(evidence["probabilities"].sum(1), torch.ones(2)))
        self.assertTrue(torch.allclose(evidence["weights"].sum(1), torch.full((2,), 3.0)))
        self.assertTrue(torch.allclose(evidence["residual_coefficients"], 0.5 * evidence["weights"]))
        self.assertTrue(torch.allclose(evidence["local_coefficients"], 1.0 + 0.5 * evidence["weights"]))

    def test_gradients_and_checkpoint_round_trip_are_finite(self):
        source = model(True, alpha=0.5, residual=True)
        with torch.no_grad():
            nn.init.normal_(source.multi_granularity_dynamic_gate.controller.weight, std=0.01)
        source.train()
        score, descriptor = source(torch.randn(4, 3, 8, 4))
        (score.square().mean() + descriptor.square().mean()).backward()
        for parameter in (source.base.projection.weight,
                          next(source.multi_granularity_part_head.parameters()),
                          source.multi_granularity_dynamic_gate.controller.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        target = model(True, alpha=0.5, residual=True)
        target.load_state_dict(source.state_dict(), strict=True)
        self.assertTrue(torch.equal(
            target.multi_granularity_dynamic_gate.controller.weight,
            source.multi_granularity_dynamic_gate.controller.weight,
        ))


if __name__ == "__main__":
    unittest.main()
