import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from tools.compare_g1_g2_g2a_diff46_gating import ComparisonError, run
from tools.compare_g1_g2_tau0p5_gating import FIXED_FIELDS


class FourVersionComparisonTest(unittest.TestCase):
    def _candidates(self, root):
        path = root / "candidates.tsv"
        rows = [
            {"stable_sample_key": "query|market1501/query/a.jpg|1|0", "split": "query", "relative_path": "market1501/query/a.jpg", "pid": "1", "camid": "0", "selection_hash": "a", "selection_rank": "1", "image_sha256": "a" * 64},
            {"stable_sample_key": "gallery|market1501/bounding_box_test/b.jpg|2|1", "split": "gallery", "relative_path": "market1501/bounding_box_test/b.jpg", "pid": "2", "camid": "1", "selection_hash": "b", "selection_rank": "2", "image_sha256": "b" * 64},
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t"); writer.writeheader(); writer.writerows(rows)
        return path, rows

    def _gates(self, root, name, candidates, omit_last=False):
        path = root / name
        rows = []
        for row in candidates[:1 if omit_last else len(candidates)]:
            rows.append({
                "stable_sample_key": row["stable_sample_key"], "split": row["split"], "relative_path": row["relative_path"], "pid": row["pid"], "camid": row["camid"],
                "selection_hash": row["selection_hash"], "image_sha256": row["image_sha256"],
                "p2": "0.2", "p4": "0.3", "p6": "0.5", "w2": "0.6", "w4": "0.9", "w6": "1.5", "dominant_k": "6", "checkpoint_sha256": "c" * 64,
            })
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIXED_FIELDS, delimiter="\t"); writer.writeheader(); writer.writerows(rows)
        return path, rows

    def _manifest(self, root, name):
        path = root / name
        path.write_text(json.dumps({"run_id": name, "selected_checkpoint": {"sha256": "d" * 64}, "metrics": {"rank1_percent": 90.0, "rank5_percent": 95.0, "rank10_percent": 97.0, "map_percent": 80.0, "selected_epoch": 120}}), encoding="utf-8")
        return path

    def _args(self, root, candidate, gates, manifests, output):
        resolved = root / "resolved.yml"; resolved.write_text("x: y\n", encoding="utf-8")
        checkpoint = root / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint")
        return Namespace(candidate_manifest=str(candidate), g1_fixed_gates=str(gates[0]), g2_fixed_gates=str(gates[1]), g2a_fixed_gates=str(gates[2]),
                         g1_run_manifest=str(manifests[0]), g2_run_manifest=str(manifests[1]), g2a_run_manifest=str(manifests[2]), diff46_run_manifest=str(manifests[3]),
                         config_file=str(root / "candidate.yml"), resolved_config=str(resolved), checkpoint=str(checkpoint), dataset_root=str(root), output_dir=str(output), device="cpu")

    def test_mismatched_g2a_keys_fail_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); candidate, rows = self._candidates(root)
            gates = [self._gates(root, "g{}.tsv".format(index), rows, omit_last=(index == 2))[0] for index in range(3)]
            manifests = [self._manifest(root, "m{}.json".format(index)) for index in range(4)]
            with self.assertRaisesRegex(ComparisonError, "exactly match"):
                run(self._args(root, candidate, gates, manifests, root / "out"))

    def test_exact_four_way_pairing_writes_four_version_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); candidate, rows = self._candidates(root)
            gates_and_rows = [self._gates(root, "g{}.tsv".format(index), rows) for index in range(3)]
            gates = [item[0] for item in gates_and_rows]; diff_rows = gates_and_rows[0][1]
            manifests = [self._manifest(root, "m{}.json".format(index)) for index in range(4)]
            args = self._args(root, candidate, gates, manifests, root / "out")
            with mock.patch("tools.compare_g1_g2_g2a_diff46_gating._load_candidate_config", return_value=object()), \
                 mock.patch("tools.compare_g1_g2_g2a_diff46_gating._market_root", return_value=root), \
                 mock.patch("tools.compare_g1_g2_g2a_diff46_gating._extract_tau_gates", return_value=(diff_rows, "c" * 64)), \
                 mock.patch("tools.compare_g1_g2_g2a_diff46_gating._plot"), \
                 mock.patch("tools.compare_g1_g2_g2a_diff46_gating._examples", return_value=[]):
                output = run(args)
            with (output / "retrieval_metrics.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)


if __name__ == "__main__":
    unittest.main()
