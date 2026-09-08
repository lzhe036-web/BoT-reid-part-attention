import unittest
from unittest import mock

from tools.analyze_g2_e_static_dynamic_alpha0p3_tau0p5 import _coefficient_rows, _statistics
from tools import recover_g2_e_static_dynamic_alpha0p3_tau0p5_experiment as e_recovery
from tools import recover_g2_global_local_experiment as shared_recovery
from tools.package_g2_e_static_dynamic_alpha0p3_tau0p5_result import package


class G2EEvidenceToolsTest(unittest.TestCase):
    def test_coefficients_keep_p_w_residual_and_c_separate(self):
        fields, rows = _coefficient_rows([{
            "stable_sample_key": "sample", "dataset_split": "query", "pid": "1", "camid": "2",
            "p2": ".2", "p4": ".3", "p6": ".5", "w2": ".6", "w4": ".9", "w6": "1.5",
            "dominant_k": "6",
        }], "a" * 64)
        self.assertIn("c6", fields)
        self.assertAlmostEqual(rows[0]["residual6"], .45)
        self.assertAlmostEqual(rows[0]["c6"], 1.45)
        self.assertAlmostEqual(rows[0]["w6"], 3.0 * rows[0]["p6"])

    def test_statistics_are_recomputed_from_per_sample_values(self):
        _fields, rows = _coefficient_rows([{
            "stable_sample_key": str(index), "dataset_split": "query", "pid": "1", "camid": "2",
            "p2": ".6" if index == 0 else ".2", "p4": ".3" if index == 0 else ".7",
            "p6": ".1", "w2": "1.8" if index == 0 else ".6",
            "w4": ".9" if index == 0 else "2.1", "w6": ".3", "dominant_k": "2" if index == 0 else "4",
        } for index in range(2)], "b" * 64)
        _fields, summary = _statistics(rows)
        p2 = next(row for row in summary if row["quantity"] == "probability" and row["scale"] == 2)
        self.assertAlmostEqual(p2["mean"], .4)
        self.assertAlmostEqual(p2["dominant_ratio"], .5)

    def test_recovery_profile_is_direct_g2a_and_alpha_point_three(self):
        names = ("EXPERIMENT_ID", "EXPECTED_BRANCH", "EXPECTED_PARENT_BRANCH",
                 "EXPECTED_PARENT_COMMIT", "RESULT_FILENAME", "FORMAL_RESULT_ARTIFACT_TYPE",
                 "METHOD_VARIANT", "METHOD_LABEL", "BASELINE_LABEL",
                 "EXPECTED_GATING_TAU", "REQUIRE_STATIC_DYNAMIC_RESIDUAL",
                 "EXPECTED_STATIC_DYNAMIC_ALPHA")
        original = {name: getattr(shared_recovery, name) for name in names}
        try:
            marker = object()
            with mock.patch.object(shared_recovery, "recover", return_value=marker) as recover:
                self.assertIs(e_recovery.recover("a", "b", "c", "d", "e"), marker)
            recover.assert_called_once_with("a", "b", "c", "d", "e")
            self.assertEqual(shared_recovery.EXPECTED_PARENT_COMMIT,
                             "d724a6536e4a819c5d2932412e90b7dea224041b")
            self.assertEqual(shared_recovery.EXPECTED_STATIC_DYNAMIC_ALPHA, .3)
            self.assertTrue(shared_recovery.REQUIRE_STATIC_DYNAMIC_RESIDUAL)
        finally:
            for name, value in original.items():
                setattr(shared_recovery, name, value)

    def test_package_fails_closed_when_result_is_missing(self):
        with self.assertRaises(FileNotFoundError):
            package("missing-output-dir")


if __name__ == "__main__":
    unittest.main()
