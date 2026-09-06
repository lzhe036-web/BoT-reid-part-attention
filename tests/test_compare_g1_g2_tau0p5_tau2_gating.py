import json
import tempfile
import unittest
from pathlib import Path

from tools.compare_g1_g2_tau0p5_tau2_gating import (
    ComparisonError, _four_metrics, _require_tau,
)


class FourWayGatingComparisonTest(unittest.TestCase):
    def _manifest(self, path, tau, rank1=90.0, m_ap=80.0, include_tau=True):
        record = {
            "dataset": "market1501", "seed": 42, "run_id": "run-{}".format(tau),
            "selected_checkpoint": {"sha256": "a" * 64},
            "metrics": {"rank1_percent": rank1, "rank5_percent": 95.0,
                        "rank10_percent": 97.0, "map_percent": m_ap,
                        "selected_epoch": 120},
        }
        if include_tau:
            record["gating_temperature"] = tau
        Path(path).write_text(json.dumps(record), encoding="utf-8")

    def test_metrics_use_tau0p5_as_the_direct_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {name: root / (name + ".json") for name in ("g1", "g2", "tau0p5", "tau2")}
            self._manifest(paths["g1"], 1.0, 89.0, 79.0)
            self._manifest(paths["g2"], 1.0, 90.0, 80.0)
            self._manifest(paths["tau0p5"], 0.5, 91.0, 81.0)
            self._manifest(paths["tau2"], 2.0, 92.5, 82.0)
            fields, rows = _four_metrics(paths)
            self.assertIn("delta_Rank-1_vs_tau0p5", fields)
            self.assertNotIn("delta_Rank-1_vs_G2", fields)
            self.assertEqual(rows[0]["delta_Rank-1_vs_tau0p5"], "not_applicable")
            self.assertAlmostEqual(rows[-1]["delta_Rank-1_vs_tau0p5"], 1.5)
            self.assertAlmostEqual(rows[-1]["delta_mAP_vs_tau0p5"], 1.0)

    def test_legacy_g1_may_lack_temperature_but_tau2_cannot(self):
        with tempfile.TemporaryDirectory() as temporary:
            g1 = Path(temporary) / "g1.json"; tau2 = Path(temporary) / "tau2.json"
            self._manifest(g1, 1.0, include_tau=False)
            self._manifest(tau2, 2.0, include_tau=False)
            self.assertEqual(_require_tau(g1, "G1", 1.0, require_temperature=False)["seed"], 42)
            with self.assertRaisesRegex(ComparisonError, "gating_temperature"):
                _require_tau(tau2, "tau2", 2.0)


if __name__ == "__main__":
    unittest.main()
