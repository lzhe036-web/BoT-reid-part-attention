import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from tools import recover_g2_global_local_experiment as shared_recovery
from tools import recover_g2_global_local_tau0p5_experiment as candidate_recovery
from tools.verify_g2_tau0p5_protocol import ProtocolError, verify


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_global_local_autodl.yml"
)
CANDIDATE_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_global_local_tau0p5_autodl.yml"
)


class G2Tau0p5ProfileTest(unittest.TestCase):
    def test_config_changes_only_temperature_and_output_dir(self):
        record = verify(BASE_CONFIG, CANDIDATE_CONFIG)
        self.assertEqual(record["status"], "verified")
        self.assertEqual(record["allowed_differences"], [
            "MODEL.MULTI_GRANULARITY_GATING_TAU", "OUTPUT_DIR",
        ])

    def test_config_rejects_any_unrelated_change(self):
        candidate = yaml.safe_load(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
        candidate = copy.deepcopy(candidate)
        candidate["SOLVER"]["MAX_EPOCHS"] = 121
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "altered.yml"
            altered.write_text(yaml.safe_dump(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "SOLVER.MAX_EPOCHS"):
                verify(BASE_CONFIG, altered)

    def test_wrapper_supplies_tau_candidate_identity_to_shared_recovery(self):
        names = (
            "EXPERIMENT_ID", "EXPECTED_BRANCH", "EXPECTED_PARENT_BRANCH", "EXPECTED_PARENT_COMMIT",
            "EXPECTED_GATING_TAU", "RESULT_FILENAME", "METHOD_VARIANT",
            "METHOD_LABEL", "BASELINE_LABEL",
            "REQUIRE_RESULT_GATING_TEMPERATURE", "DEFAULT_CONFIG",
        )
        original = {name: getattr(shared_recovery, name) for name in names}
        try:
            sentinel = object()
            with mock.patch.object(shared_recovery, "recover", return_value=sentinel) as recover:
                self.assertIs(candidate_recovery.recover("a", "b", "c", "d", "e"), sentinel)
            recover.assert_called_once_with("a", "b", "c", "d", "e")
            self.assertEqual(shared_recovery.EXPERIMENT_ID, candidate_recovery.EXPERIMENT_ID)
            self.assertEqual(shared_recovery.EXPECTED_BRANCH, candidate_recovery.EXPECTED_BRANCH)
            self.assertEqual(shared_recovery.EXPECTED_PARENT_BRANCH,
                             candidate_recovery.EXPECTED_PARENT_BRANCH)
            self.assertEqual(shared_recovery.EXPECTED_PARENT_COMMIT,
                             candidate_recovery.EXPECTED_PARENT_COMMIT)
            self.assertEqual(shared_recovery.EXPECTED_GATING_TAU, 0.5)
            self.assertTrue(shared_recovery.REQUIRE_RESULT_GATING_TEMPERATURE)
        finally:
            for name, value in original.items():
                setattr(shared_recovery, name, value)


if __name__ == "__main__":
    unittest.main()
