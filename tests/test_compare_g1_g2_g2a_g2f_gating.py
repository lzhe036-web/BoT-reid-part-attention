import csv
import tempfile
import unittest
from pathlib import Path

from tools.compare_g1_g2_g2a_g2f_gating import compare


def write_samples(path, keys):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stable_sample_key", "w2", "w4", "w6"],
                                delimiter="\t")
        writer.writeheader()
        for index, key in enumerate(keys):
            writer.writerow({"stable_sample_key": key, "w2": 1.5 + index * .1,
                             "w4": 1.0, "w6": .5 - index * .1})


class CrossVersionGatingComparisonTest(unittest.TestCase):
    def test_requires_identical_stable_key_order_and_emits_png_pdf_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for label in ("g1", "g2", "g2a", "g2f"):
                path = root / (label + ".tsv")
                write_samples(path, ["a", "b"])
                paths[label] = str(path)
            output = compare({"G1": paths["g1"], "G2": paths["g2"],
                              "G2-A-tau0p5": paths["g2a"], "G2-F-Top2-tau0p5": paths["g2f"]},
                             root / "out")
            self.assertTrue((output / "cross_version_plot_data.csv").is_file())
            self.assertTrue((output / "g1_g2_g2a_g2f_actual_weight_distribution.png").is_file())
            self.assertTrue((output / "g1_g2_g2a_g2f_actual_weight_distribution.pdf").is_file())

    def test_rejects_mismatched_order_without_intersection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for label in ("g1", "g2", "g2a", "g2f"):
                path = root / (label + ".tsv")
                write_samples(path, ["a", "b"] if label != "g2f" else ["b", "a"])
                paths[label] = str(path)
            with self.assertRaisesRegex(ValueError, "refusing a silent intersection"):
                compare({"G1": paths["g1"], "G2": paths["g2"],
                         "G2-A-tau0p5": paths["g2a"], "G2-F-Top2-tau0p5": paths["g2f"]},
                        root / "out")


if __name__ == "__main__":
    unittest.main()
