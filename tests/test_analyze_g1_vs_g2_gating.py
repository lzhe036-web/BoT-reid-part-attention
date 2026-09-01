import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools import analyze_g1_vs_g2_gating as analysis


class G1VsG2GatingAnalysisTest(unittest.TestCase):
    def _candidate(self, key="query|market1501/query/0001_c1s1_000001_00.jpg|1|0"):
        return {
            "stable_sample_key": key, "split": "query", "relative_path": "market1501/query/0001_c1s1_000001_00.jpg",
            "pid": 1, "camid": 0, "selection_hash": "a" * 64, "selection_rank": 1, "image_sha256": "b" * 64,
        }

    def _gate(self, probabilities, checkpoint="a" * 64):
        weights = [3.0 * value for value in probabilities]
        return {"p": probabilities, "w": weights, "dominant": analysis.SCALES[max(range(3), key=lambda index: weights[index])], "checkpoint_sha256": checkpoint,
                "split": "query", "relative_path": "market1501/query/0001_c1s1_000001_00.jpg", "pid": "1", "camid": "0"}

    def test_fixed_candidate_selection_is_splitwise_and_order_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets" / "market1501"
            (root / "query").mkdir(parents=True); (root / "bounding_box_test").mkdir()
            for name in ("0002_c2s1_000001_00.jpg", "0001_c1s1_000002_00.jpg", "-1_c1s1_000003_00.jpg"):
                (root / "query" / name).write_bytes(name.encode("utf-8"))
            for name in ("0004_c3s1_000004_00.jpg", "0003_c2s1_000005_00.jpg"):
                (root / "bounding_box_test" / name).write_bytes(name.encode("utf-8"))
            duke = root.parent / "DukeMTMC-reID" / "query"
            duke.mkdir(parents=True)
            (duke / "9999_c1_f000001.jpg").write_bytes(b"must-not-be-scanned")
            first, second = Path(directory) / "one.tsv", Path(directory) / "two.tsv"
            rows_one, _ = analysis.build_fixed_candidates(root.parent, first, query_limit=2, gallery_limit=2)
            rows_two, _ = analysis.build_fixed_candidates(root.parent, second, query_limit=2, gallery_limit=2)
            self.assertEqual(rows_one, rows_two)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual([row["split"] for row in rows_one].count("query"), 2)
            self.assertEqual([row["split"] for row in rows_one].count("gallery"), 2)
            self.assertTrue(all("__MACOSX" not in row["relative_path"] for row in rows_one))
            self.assertTrue(all(row["pid"] != -1 for row in rows_one))

    def test_legacy_null_field_does_not_mask_metrics_or_selected_checkpoint_evidence(self):
        manifest = {
            "dataset": None,
            "selected_epoch": None,
            "metrics": {"dataset": "market1501", "selected_epoch": 120},
            "selected_checkpoint": {"epoch": 80},
        }
        self.assertEqual(
            analysis._manifest_field_with_source(manifest, "dataset"),
            ("market1501", "run_manifest.metrics.dataset"),
        )
        self.assertEqual(
            analysis._manifest_field_with_source(manifest, "selected_epoch"),
            (120, "run_manifest.metrics.selected_epoch"),
        )
        del manifest["metrics"]["selected_epoch"]
        self.assertEqual(
            analysis._manifest_field_with_source(manifest, "selected_epoch"),
            (80, "run_manifest.selected_checkpoint.epoch"),
        )
        del manifest["selected_checkpoint"]["epoch"]
        self.assertEqual(
            analysis._manifest_field_with_source(manifest, "selected_epoch"),
            (analysis.NOT_RECORDED, analysis.NOT_RECORDED),
        )

    def test_historical_tsv_requires_probability_weight_order_and_sums(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.tsv"
            fields = ("stable_sample_key", "dataset_split", "pid", "camid", "p2", "p4", "p6", "w2", "w4", "w6", "entropy", "dominant_k", "checkpoint_sha256")
            row = {"stable_sample_key": "key", "dataset_split": "query", "pid": "1", "camid": "0", "p2": "0.2", "p4": "0.3", "p6": "0.5", "w2": "0.6", "w4": "0.9", "w6": "1.5", "entropy": "1.0", "dominant_k": "6", "checkpoint_sha256": "a" * 64}
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t"); writer.writeheader(); writer.writerow(row)
            self.assertEqual(analysis._validate_historical_gating_samples(path, "a" * 64, "test"), 1)
            row["w6"] = "1.0"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t"); writer.writeheader(); writer.writerow(row)
            with self.assertRaises(analysis.EvidenceError):
                analysis._validate_historical_gating_samples(path, "a" * 64, "test")

    def test_fixed_gate_reader_rejects_checkpoint_or_dominant_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.tsv"
            fields = ("stable_sample_key", "split", "relative_path", "pid", "camid", "selection_hash", "image_sha256", "p2", "p4", "p6", "w2", "w4", "w6", "dominant_k", "checkpoint_sha256")
            row = {"stable_sample_key": "key", "split": "query", "relative_path": "market1501/query/image.jpg", "pid": 1, "camid": 0, "selection_hash": "x", "image_sha256": "y", "p2": .2, "p4": .3, "p6": .5, "w2": .6, "w4": .9, "w6": 1.5, "dominant_k": 6, "checkpoint_sha256": "a" * 64}
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t"); writer.writeheader(); writer.writerow(row)
            self.assertIn("key", analysis.read_fixed_gates(path, "a" * 64))
            with self.assertRaises(analysis.EvidenceError):
                analysis.read_fixed_gates(path, "b" * 64)

    def test_describe_and_dominant_statistics_are_computed_from_values(self):
        summary = analysis._describe([1.0, 2.0, 3.0], seed=7, replicates=100)
        self.assertEqual(summary["count"], 3); self.assertAlmostEqual(summary["mean"], 2.0); self.assertAlmostEqual(summary["std"], (2.0 / 3.0) ** .5)
        g1 = {"a": self._gate([.2, .7, .1]), "b": self._gate([.2, .6, .2])}
        g2 = {"a": self._gate([.2, .2, .6]), "b": self._gate([.3, .2, .5])}
        rows, _one, _two, intersection, pairing = analysis._collapse_rows(g1, g2, seed=7, replicates=100)
        self.assertEqual(intersection, ["a", "b"]); self.assertTrue(pairing["complete_pairing"])
        g1_k4 = next(row for row in rows if row["model"] == analysis.G1_LABEL and row["metric"] == "dominant_k4_ratio")
        g2_k6 = next(row for row in rows if row["model"] == analysis.G2_LABEL and row["metric"] == "dominant_k6_ratio")
        self.assertEqual(g1_k4["mean"], 1.0); self.assertEqual(g2_k6["mean"], 1.0)

    def test_annotation_validation_and_empty_type_outputs_do_not_invent_labels(self):
        candidate = self._candidate()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifests, tables, samples = root / "manifests", root / "tables", root / "samples"
            manifests.mkdir(); tables.mkdir(); samples.mkdir()
            output = analysis._type_outputs([], [candidate], {candidate["stable_sample_key"]: self._gate([.2, .3, .5])}, {candidate["stable_sample_key"]: self._gate([.2, .3, .5])}, root, samples, tables, 42, 100)
            self.assertEqual(output[2], "not_recorded: blind annotations are empty")
            self.assertTrue((manifests / "sample_gating_weights.tsv").is_file())
            annotation = manifests / "annotations.tsv"
            annotation.write_text("stable_sample_key\timage_type\tannotation_method\tannotation_version\tshort_reason\nunknown\tclear\tmanual\tv1\treason\n", encoding="utf-8")
            with self.assertRaises(analysis.EvidenceError):
                analysis._read_annotations(annotation, [candidate])

    def test_k6_type_comparison_marks_insufficient_samples(self):
        candidate = self._candidate(); candidates = [candidate]
        annotations = [{"stable_sample_key": candidate["stable_sample_key"], "image_type": "clear", "annotation_method": "manual", "annotation_version": "v1", "short_reason": "reason"}]
        with tempfile.TemporaryDirectory() as directory:
            rows = analysis._k6_type_comparisons(annotations, candidates, {candidate["stable_sample_key"]: self._gate([.1, .2, .7])}, directory, 42, 100)
            self.assertTrue(rows and all(row["status"] == "insufficient_blind_annotated_fixed_samples" for row in rows))


if __name__ == "__main__":
    unittest.main()
