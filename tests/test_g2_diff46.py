import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
import yaml

from config import cfg
from modeling import build_model
from modeling.baseline import MultiGranularityDynamicGate
from tools import recover_g2_diff46_experiment as diff46_recovery
from tools import recover_g2_global_local_experiment as shared_recovery
from tools.analyze_g2_global_local_gating import _controller_blocks, _load_configuration
from tools.verify_g2_diff46_protocol import ProtocolError, verify


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_global_local_tau0p5_autodl.yml"
)
DIFF46_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_g2_diff46_autodl.yml"
)


class G2Diff46Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _gate(self):
        return MultiGranularityDynamicGate(
            2048, 3, temperature=0.5,
            gating_input="concat_global_local_diff46", local_feature_dim=256,
        )

    def test_protocol_has_only_input_mode_and_output_directory_differences(self):
        record = verify(BASE_CONFIG, DIFF46_CONFIG)
        self.assertEqual(record["status"], "verified")
        self.assertEqual(record["allowed_differences"], [
            "MODEL.MULTI_GRANULARITY_GATING_INPUT", "OUTPUT_DIR",
        ])

    def test_protocol_rejects_unrelated_change(self):
        candidate = yaml.safe_load(DIFF46_CONFIG.read_text(encoding="utf-8"))
        candidate = copy.deepcopy(candidate)
        candidate["SOLVER"]["MAX_EPOCHS"] = 121
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.yml"
            path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "SOLVER.MAX_EPOCHS"):
                verify(BASE_CONFIG, path)

    def test_gate_input_order_width_and_zero_initialization(self):
        gate = self._gate()
        g = torch.randn(3, 2048)
        z2, z4, z6 = (torch.randn(3, 256) for _ in range(3))
        actual = gate.controller_input(g, (z2, z4, z6))
        self.assertEqual(tuple(actual.shape), (3, 3072))
        self.assertTrue(torch.equal(actual[:, :2048], g))
        self.assertTrue(torch.equal(actual[:, 2048:2304], z2))
        self.assertTrue(torch.equal(actual[:, 2304:2560], z4))
        self.assertTrue(torch.equal(actual[:, 2560:2816], z6))
        self.assertTrue(torch.equal(actual[:, 2816:3072], torch.abs(z4 - z6)))
        self.assertEqual(gate.controller.in_features, 3072)
        self.assertEqual(tuple(gate.controller.weight.shape), (3, 3072))
        self.assertEqual(gate.controller.weight.count_nonzero().item(), 0)
        self.assertEqual(gate.controller.bias.count_nonzero().item(), 0)
        _logits, probabilities, weights = gate(g, (z2, z4, z6))
        self.assertTrue(torch.allclose(probabilities, torch.full_like(probabilities, 1.0 / 3.0)))
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(3)))
        self.assertTrue(torch.allclose(weights.sum(1), torch.full((3,), 3.0)))

    def test_delta46_is_graph_connected_and_zero_when_z4_equals_z6(self):
        gate = self._gate()
        g = torch.randn(2, 2048, requires_grad=True)
        z2 = torch.randn(2, 256, requires_grad=True)
        z4 = torch.randn(2, 256, requires_grad=True)
        z6 = torch.randn(2, 256, requires_grad=True)
        with torch.no_grad():
            gate.controller.weight.normal_(std=0.01)
        logits, probabilities, weights = gate(g, (z2, z4, z6))
        (logits.square().mean() + probabilities.square().mean() + weights.square().mean()).backward()
        self.assertIsNotNone(z4.grad); self.assertIsNotNone(z6.grad)
        self.assertTrue(torch.isfinite(z4.grad).all()); self.assertTrue(torch.isfinite(z6.grad).all())
        identical = torch.randn(2, 256, requires_grad=True)
        inputs = gate.controller_input(g.detach(), (z2.detach(), identical, identical))
        self.assertEqual(inputs[:, 2816:].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(inputs).all())

    def test_old_g2_mode_schema_and_diff46_checkpoint_are_incompatible(self):
        old = MultiGranularityDynamicGate(2048, 3, temperature=0.5,
                                          gating_input="concat_global_local", local_feature_dim=256)
        new = self._gate()
        self.assertEqual(tuple(old.controller.weight.shape), (3, 2816))
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            new.load_state_dict(old.state_dict(), strict=True)

    def test_wider_controller_does_not_perturb_same_seed_shared_initialization(self):
        g2a = cfg.clone(); g2a.merge_from_file(str(BASE_CONFIG)); g2a.defrost()
        g2a.MODEL.PRETRAIN_CHOICE = "none"; g2a.MODEL.PRETRAIN_PATH = ""; g2a.freeze()
        diff46 = cfg.clone(); diff46.merge_from_file(str(DIFF46_CONFIG)); diff46.defrost()
        diff46.MODEL.PRETRAIN_CHOICE = "none"; diff46.MODEL.PRETRAIN_PATH = ""; diff46.freeze()
        torch.manual_seed(42); baseline = build_model(g2a, 3)
        torch.manual_seed(42); candidate = build_model(diff46, 3)
        for key, value in baseline.state_dict().items():
            if not key.startswith("multi_granularity_dynamic_gate."):
                self.assertTrue(torch.equal(value, candidate.state_dict()[key]), key)

    def test_analyzer_records_five_actual_controller_blocks(self):
        configuration = cfg.clone(); configuration.merge_from_file(str(DIFF46_CONFIG)); configuration.freeze()
        self.assertEqual(_controller_blocks(configuration), (
            ("g", 2048), ("z2", 256), ("z4", 256), ("z6", 256),
            ("delta46_abs_z4_minus_z6", 256),
        ))
        self.assertEqual(str(_load_configuration(DIFF46_CONFIG, 0.5, "concat_global_local_diff46").MODEL.MULTI_GRANULARITY_GATING_INPUT), "concat_global_local_diff46")

    def test_recovery_wrapper_sets_only_diff46_profile_for_shared_recovery(self):
        names = ("EXPERIMENT_ID", "EXPECTED_BRANCH", "EXPECTED_PARENT_BRANCH", "EXPECTED_PARENT_COMMIT",
                 "EXPECTED_GATING_INPUT", "EXPECTED_GATING_INPUT_DESCRIPTION", "EXPECTED_CONTROLLER_INPUT_DIM",
                 "EXPECTED_DELTA46_DEFINITION", "EXPECTED_GATING_TAU", "RESULT_FILENAME", "METHOD_VARIANT",
                 "METHOD_LABEL", "BASELINE_LABEL", "REQUIRE_RESULT_GATING_TEMPERATURE", "REQUIRE_RESULT_INPUT_METADATA", "DEFAULT_CONFIG")
        original = {name: getattr(shared_recovery, name) for name in names}
        try:
            sentinel = object()
            with mock.patch.object(shared_recovery, "recover", return_value=sentinel) as recover:
                self.assertIs(diff46_recovery.recover("a", "b", "c", "d", "e"), sentinel)
            recover.assert_called_once_with("a", "b", "c", "d", "e")
            self.assertEqual(shared_recovery.EXPECTED_GATING_INPUT, "concat_global_local_diff46")
            self.assertEqual(shared_recovery.EXPECTED_CONTROLLER_INPUT_DIM, 3072)
            self.assertTrue(shared_recovery.REQUIRE_RESULT_INPUT_METADATA)
        finally:
            for name, value in original.items(): setattr(shared_recovery, name, value)


if __name__ == "__main__":
    unittest.main()
