import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from tools import recover_g2_global_local_experiment as shared_recovery
from tools import recover_g2_global_local_tau2_experiment as tau2_recovery
from tools.verify_g2_tau0p5_protocol import ProtocolError, verify


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_global_local_tau0p5_autodl.yml"
)
CANDIDATE_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_global_local_tau2_autodl.yml"
)


class G2Tau2ProfileTest(unittest.TestCase):
    def test_config_changes_only_temperature_and_output_dir(self):
        record = verify(BASE_CONFIG, CANDIDATE_CONFIG, 0.5, 2.0)
        self.assertEqual(record["status"], "verified")
        self.assertEqual(record["algorithm_variable"],
                         "MODEL.MULTI_GRANULARITY_GATING_TAU: 0.5 -> 2.0")
        self.assertEqual(record["allowed_differences"], [
            "MODEL.MULTI_GRANULARITY_GATING_TAU", "OUTPUT_DIR",
        ])

    def test_config_rejects_unrelated_change(self):
        candidate = copy.deepcopy(yaml.safe_load(CANDIDATE_CONFIG.read_text(encoding="utf-8")))
        candidate["MODEL"]["MULTI_GRANULARITY_GATING_INPUT"] = "concat_local"
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "altered.yml"
            altered.write_text(yaml.safe_dump(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "GATING_INPUT"):
                verify(BASE_CONFIG, altered, 0.5, 2.0)

    def test_recovery_wrapper_supplies_tau2_identity_and_parent(self):
        names = (
            "EXPERIMENT_ID", "EXPECTED_BRANCH", "EXPECTED_PARENT_BRANCH", "EXPECTED_PARENT_COMMIT",
            "EXPECTED_GATING_TAU", "RESULT_FILENAME", "METHOD_VARIANT", "METHOD_LABEL",
            "BASELINE_LABEL", "REQUIRE_RESULT_GATING_TEMPERATURE", "DEFAULT_CONFIG",
        )
        original = {name: getattr(shared_recovery, name) for name in names}
        try:
            sentinel = object()
            with mock.patch.object(shared_recovery, "recover", return_value=sentinel) as recover:
                self.assertIs(tau2_recovery.recover("a", "b", "c", "d", "e"), sentinel)
            recover.assert_called_once_with("a", "b", "c", "d", "e")
            self.assertEqual(shared_recovery.EXPERIMENT_ID, "C2-L03-MGDG-G2-GL-T2-S42")
            self.assertEqual(shared_recovery.EXPECTED_BRANCH, "codex/g2-global-local-gating-tau2")
            self.assertEqual(shared_recovery.EXPECTED_PARENT_COMMIT,
                             "d724a6536e4a819c5d2932412e90b7dea224041b")
            self.assertEqual(shared_recovery.EXPECTED_GATING_TAU, 2.0)
            self.assertTrue(shared_recovery.REQUIRE_RESULT_GATING_TEMPERATURE)
        finally:
            for name, value in original.items():
                setattr(shared_recovery, name, value)


if __name__ == "__main__":
    unittest.main()
