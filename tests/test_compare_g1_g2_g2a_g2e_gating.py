import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.compare_g1_g2_g2a_g2e_gating import compare


def _rows(alpha=None):
    output = []
    for key, p2, p4, p6 in (("a" * 64, .6, .3, .1), ("b" * 64, .2, .7, .1)):
        row = {"stable_sample_key": key, "p2": p2, "p4": p4, "p6": p6,
               "w2": 3 * p2, "w4": 3 * p4, "w6": 3 * p6,
               "dominant_k": 2 if p2 > p4 else 4, "checkpoint_sha256": "c" * 64}
        if alpha is not None:
            row.update({"alpha": alpha, "c2": 1 + alpha * row["w2"],
                        "c4": 1 + alpha * row["w4"], "c6": 1 + alpha * row["w6"]})
        output.append(row)
    return output


class FourVersionComparisonTest(unittest.TestCase):
    def _write(self, path, rows):
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(rows)

    def _result(self, path):
        Path(path).write_text(json.dumps({"metrics": {"rank1_percent": 90., "rank5_percent": 95., "rank10_percent": 97., "map_percent": 80.}, "selected_checkpoint": {"epoch": 120, "sha256": "c" * 64}}), encoding="utf-8")

    def test_refuses_silent_sample_intersection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); samples = []; results = []
            for index in range(4):
                sample, result = root / ("s{}.tsv".format(index)), root / ("r{}.json".format(index))
                rows = _rows(.3 if index == 3 else None)
                if index == 1: rows.reverse()
                self._write(sample, rows); self._result(result); samples.append(sample); results.append(result)
            with self.assertRaisesRegex(ValueError, "keys/order differ"):
                compare(samples, results, root / "out")

    def test_writes_four_version_outputs_only_for_identical_ordered_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); samples = []; results = []
            for index in range(4):
                sample, result = root / ("s{}.tsv".format(index)), root / ("r{}.json".format(index))
                self._write(sample, _rows(.3 if index == 3 else None))
                self._result(result); samples.append(sample); results.append(result)
            output = compare(samples, results, root / "out")
            self.assertTrue((output / "retrieval_metrics.csv").is_file())
            self.assertTrue((output / "per_sample_gating.tsv").is_file())
            self.assertTrue((output / "figures" / "four_version_weight_distributions.png").is_file())


if __name__ == "__main__":
    unittest.main()
