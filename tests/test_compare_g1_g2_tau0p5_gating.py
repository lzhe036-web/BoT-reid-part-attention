import csv
import tempfile
import unittest
from pathlib import Path

from tools.compare_g1_g2_tau0p5_gating import (
    ComparisonError,
    FIXED_FIELDS,
    _read_fixed_gates,
    _statistics,
)


class G2Tau0p5ComparisonTest(unittest.TestCase):
    def _row(self, key, p2, p4, p6):
        return {
            "stable_sample_key": key, "split": "query", "relative_path": "market1501/query/x.jpg",
            "pid": 1, "camid": 0, "selection_hash": "a", "image_sha256": "b",
            "p2": p2, "p4": p4, "p6": p6, "w2": 3 * p2, "w4": 3 * p4,
            "w6": 3 * p6, "dominant_k": 2 if p2 >= p4 and p2 >= p6 else (4 if p4 >= p6 else 6),
            "checkpoint_sha256": "c" * 64,
        }

    def _write(self, path, rows):
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIXED_FIELDS, delimiter="\t")
            writer.writeheader(); writer.writerows(rows)

    def test_fixed_gate_reader_and_statistics_use_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gates.tsv"
            rows = [self._row("q1", 0.5, 0.4, 0.1), self._row("q2", 0.1, 0.6, 0.3)]
            self._write(path, rows)
            mapping = _read_fixed_gates(path, "test")
            stats = _statistics("test", list(mapping.values()))
            self.assertEqual(stats["sample_count"], 2)
            self.assertAlmostEqual(stats["p4_mean"], 0.5)
            self.assertAlmostEqual(stats["dominant_k2_ratio"], 0.5)
            self.assertAlmostEqual(stats["dominant_k4_ratio"], 0.5)

    def test_reader_rejects_non_normalized_probability(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gates.tsv"
            row = self._row("q1", 0.5, 0.4, 0.1); row["p6"] = 0.2
            self._write(path, [row])
            with self.assertRaisesRegex(ComparisonError, "scaled-softmax"):
                _read_fixed_gates(path, "test")


if __name__ == "__main__":
    unittest.main()
