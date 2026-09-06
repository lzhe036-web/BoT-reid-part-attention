import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.analyze_g2_e1_static_dynamic_alpha import _coefficient_rows
from tools.compare_g1_g2_e1_static_dynamic import _read_result, _stats
from tools import recover_g2_e1_static_dynamic_experiment as e1_recovery
from tools import recover_g2_global_local_experiment as shared_recovery


class E1EvidenceToolsTest(unittest.TestCase):
    def test_e1_coefficient_rows_keep_p_w_residual_and_c_separate(self):
        fields, rows = _coefficient_rows([{
            "stable_sample_key": "sample", "dataset_split": "query", "pid": "1", "camid": "2",
            "p2": "0.2", "p4": "0.3", "p6": "0.5", "w2": "0.6", "w4": "0.9", "w6": "1.5",
            "dominant_k": "6",
        }], "a" * 64)
        self.assertIn("c6", fields); self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["residual6"], .75)
        self.assertAlmostEqual(rows[0]["c6"], 1.75)
        self.assertAlmostEqual(rows[0]["w6"], 3.0 * rows[0]["p6"])

    def test_statistics_are_sample_weighted_from_per_sample_probabilities(self):
        rows = [
            {"p2": .6, "p4": .3, "p6": .1, "dominant_k": 2},
            {"p2": .2, "p4": .7, "p6": .1, "dominant_k": 4},
        ]
        summary = _stats("E1", rows)
        self.assertEqual(summary["sample_count"], 2)
        self.assertAlmostEqual(summary["p2_mean"], .4)
        self.assertAlmostEqual(summary["dominant_k4_ratio"], .5)

    def test_metrics_reader_requires_one_formal_source_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps({
                "metrics": {"rank1_percent": 90.0, "rank5_percent": 95.0,
                            "rank10_percent": 97.0, "map_percent": 80.0},
                "selected_checkpoint": {"sha256": "b" * 64, "epoch": 120},
            }), encoding="utf-8")
            record = _read_result(path, "E1")
            self.assertEqual(record["selected_epoch"], 120)
            self.assertAlmostEqual(record["mAP"], 80.0)

    def test_recovery_profile_records_direct_original_g2_parent(self):
        names = (
            "EXPERIMENT_ID", "EXPECTED_BRANCH", "EXPECTED_PARENT_BRANCH", "EXPECTED_PARENT_COMMIT",
            "RESULT_FILENAME", "FORMAL_RESULT_ARTIFACT_TYPE", "METHOD_VARIANT", "METHOD_LABEL",
            "BASELINE_LABEL", "REQUIRE_STATIC_DYNAMIC_RESIDUAL", "EXPECTED_STATIC_DYNAMIC_ALPHA",
        )
        original = {name: getattr(shared_recovery, name) for name in names}
        try:
            marker = object()
            with mock.patch.object(shared_recovery, "recover", return_value=marker) as recover:
                self.assertIs(e1_recovery.recover("a", "b", "c", "d", "e"), marker)
            recover.assert_called_once_with("a", "b", "c", "d", "e")
            self.assertEqual(shared_recovery.EXPECTED_PARENT_COMMIT,
                             "5a9a2a33e19f3d9dfc3afaf356dc8b9ee3f0f737")
            self.assertEqual(shared_recovery.EXPECTED_STATIC_DYNAMIC_ALPHA, .5)
            self.assertTrue(shared_recovery.REQUIRE_STATIC_DYNAMIC_RESIDUAL)
        finally:
            for name, value in original.items():
                setattr(shared_recovery, name, value)


if __name__ == "__main__":
    unittest.main()
