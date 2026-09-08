import unittest

from tools.analyze_g2_f_top2_tau0p5 import _combination, _combination_statistics
from tools.package_g2_f_top2_tau0p5_result import _validate_rows


def row():
    return {
        "stable_sample_key": "sample", "dataset_split": "query", "pid": "1", "camid": "0",
        "p_dense2": ".50", "p_dense4": ".30", "p_dense6": ".20",
        "p_top2_2": ".625", "p_top2_4": ".375", "p_top2_6": "0.0",
        "w2": "1.875", "w4": "1.125", "w6": "0.0", "mask2": "1", "mask4": "1", "mask6": "0",
        "top2_combination": "K2+K4", "disabled_scale": "K6", "dominant_k": "2",
        "selection_boundary_tie": "0", "checkpoint_sha256": "a" * 64,
    }


class G2FTop2EvidenceToolsTest(unittest.TestCase):
    def test_combination_is_mask_based_not_positive_weight_based(self):
        self.assertEqual(_combination([True, False, True]), ("K2+K6", "K4"))

    def test_top2_rows_require_actual_mask_formula(self):
        _validate_rows([row()], "a" * 64)
        invalid = row(); invalid["mask6"] = "1"
        with self.assertRaises(ValueError):
            _validate_rows([invalid], "a" * 64)

    def test_combination_ratios_sum_to_one(self):
        first, second = row(), row()
        second.update({"stable_sample_key": "other", "mask4": "0", "mask6": "1",
                       "top2_combination": "K2+K6", "disabled_scale": "K4"})
        stats = _combination_statistics([first, second])[1]
        self.assertEqual(sum(item["sample_count"] for item in stats), 2)
        self.assertAlmostEqual(sum(item["ratio"] for item in stats), 1.0)


if __name__ == "__main__":
    unittest.main()
