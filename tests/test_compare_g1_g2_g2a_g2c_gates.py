import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.compare_g1_g2_g2a_g2c_gates import compare


class FourVersionGateComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = {}
        fields = ["stable_sample_key", "dataset_split", "pid", "camid", "p2", "p4", "p6", "w2", "w4", "w6", "entropy", "dominant_k", "checkpoint_sha256"]
        for index, label in enumerate(("G1", "G2", "G2-A", "G2-C")):
            path = self.root / "{}.tsv".format(label)
            rows = [{
                "stable_sample_key": "query|market1501/query/0001_c1s1_000001_00.jpg|1|0",
                "dataset_split": "query", "pid": 1, "camid": 0,
                "p2": 0.2 + 0.01 * index, "p4": 0.5, "p6": 0.3 - 0.01 * index,
                "w2": 0.6 + 0.03 * index, "w4": 1.5, "w6": 0.9 - 0.03 * index,
                "entropy": 1.0, "dominant_k": 4, "checkpoint_sha256": label.lower() * 16,
            }]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            self.paths[label] = path

    def tearDown(self):
        self.temp.cleanup()

    def test_compares_only_exactly_identical_sample_set(self):
        out = compare(self.paths, {}, self.root, self.root / "out")
        self.assertTrue((out / "gate_statistics.csv").is_file())
        self.assertTrue((out / "figures" / "gate_probability_distributions.pdf").is_file())
        self.assertTrue((out / "figures" / "k4_dominant_examples.png").is_file())
        manifest = json.loads((out / "comparison_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sample_set_status"], "identical_stable_sample_key_order_verified")

    def test_rejects_even_one_different_key(self):
        text = self.paths["G2-A"].read_text(encoding="utf-8")
        self.paths["G2-A"].write_text(text.replace("0001_c1s1", "0002_c1s1"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly the same frozen sample list"):
            compare(self.paths, {}, self.root, self.root / "out")


if __name__ == "__main__":
    unittest.main()
