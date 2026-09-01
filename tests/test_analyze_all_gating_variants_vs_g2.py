import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools import analyze_all_gating_variants_vs_g2 as analysis


class AllGatingVariantsAnalysisTest(unittest.TestCase):
    def _sample_row(self, spec, key="sample-key", checkpoint="a" * 64):
        probabilities = {scale: 1.0 / len(spec.active_scales) for scale in spec.active_scales}
        row = {
            "stable_sample_key": key, "dataset_split": "query", "pid": "1", "camid": "0",
            "entropy": "0.0", "dominant_k": str(spec.active_scales[0]),
            "checkpoint_sha256": checkpoint,
        }
        for scale in spec.active_scales:
            row["p{}".format(scale)] = str(probabilities[scale])
            row["w{}".format(scale)] = str(spec.expected_native_weight_sum * probabilities[scale])
        return row

    def _write_tsv(self, root, spec, rows):
        path = Path(root) / (spec.key + ".tsv")
        fields = ["stable_sample_key", "dataset_split", "pid", "camid"]
        fields += ["p{}".format(scale) for scale in spec.active_scales]
        fields += ["w{}".format(scale) for scale in spec.active_scales]
        fields += ["entropy", "dominant_k", "checkpoint_sha256"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_every_variant_uses_the_fixed_g2_baseline(self):
        inventory = {
            spec.key: {
                "formal_status": analysis.MISSING_FORMAL,
                "rank1": analysis.NOT_RECORDED,
                "map": analysis.NOT_RECORDED,
            }
            for spec in analysis.VARIANTS
        }
        inventory["g2"].update({"formal_status": "success", "rank1": "94.0", "map": "87.0"})
        _fields, rows = analysis.performance_rows(inventory)
        self.assertEqual({row["baseline"] for row in rows}, {analysis.BASELINE_LABEL})
        self.assertEqual(rows[0]["rank1_delta_vs_g2"], 0.0)
        self.assertEqual(rows[0]["map_delta_vs_g2"], 0.0)

    def test_inherited_or_smoke_rows_cannot_become_a_comparator_formal_result(self):
        g1 = analysis.SPEC_BY_KEY["g1"]
        inherited_g2 = {
            "experiment_id": analysis.SPEC_BY_KEY["g2"].experiment_id,
            "run_kind": "formal", "status": "success",
        }
        g1_smoke = {"experiment_id": g1.experiment_id, "run_kind": "smoke", "status": "success"}
        g1_incomplete = {"experiment_id": g1.experiment_id, "run_kind": "formal", "status": "incomplete"}
        self.assertIsNone(analysis.find_formal_row([inherited_g2, g1_smoke, g1_incomplete], g1))

    def test_three_and_two_way_weight_contracts_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            g2 = analysis.SPEC_BY_KEY["g2"]
            without_z6 = analysis.SPEC_BY_KEY["g2_without_z6"]
            three_path = self._write_tsv(directory, g2, [self._sample_row(g2)])
            two_path = self._write_tsv(directory, without_z6, [self._sample_row(without_z6)])
            self.assertEqual(
                set(analysis.read_gating_samples(three_path, g2)["sample-key"]["weights"]),
                {2, 4, 6},
            )
            self.assertAlmostEqual(
                sum(analysis.read_gating_samples(three_path, g2)["sample-key"]["weights"].values()), 3.0
            )
            self.assertAlmostEqual(
                sum(analysis.read_gating_samples(two_path, without_z6)["sample-key"]["weights"].values()), 1.0
            )
            bad = self._sample_row(without_z6)
            bad["w2"] = "1.5"
            bad["w4"] = "0.5"
            bad_path = self._write_tsv(directory, without_z6, [bad])
            with self.assertRaises(analysis.AnalysisError):
                analysis.read_gating_samples(bad_path, without_z6)

    def test_excluded_scale_is_na_not_a_zero_measurement(self):
        rows = analysis.unavailable_weight_rows(
            analysis.SPEC_BY_KEY["g2_without_z6"], "missing_formal_evidence"
        )
        excluded = [row for row in rows if row["weight"] == "w6"]
        self.assertEqual(len(excluded), 2)
        for row in excluded:
            self.assertEqual(row["status"], "excluded")
            self.assertEqual(row["mean"], analysis.NOT_APPLICABLE)

    def test_normalized_entropy_uses_the_active_scale_count(self):
        sample_map = {
            "x": {
                "probabilities": {2: 0.5, 4: 0.5},
                "weights": {2: 0.5, 4: 0.5}, "dominant": 2,
            }
        }
        metrics = analysis.collapse_metrics(sample_map, (2, 4))[0]
        self.assertAlmostEqual(metrics["normalized_entropy"], 1.0)
        self.assertAlmostEqual(metrics["normalized_effective_active_scales"], 1.0)

    def test_fixed_manifest_is_order_independent_and_does_not_read_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets"
            market = root / "market1501"
            for split in ("query", "bounding_box_test"):
                (market / split).mkdir(parents=True)
            for name in ("0002_c2s1_000001_00.jpg", "0001_c1s1_000002_00.jpg"):
                (market / "query" / name).write_bytes(b"not-an-image")
            (market / "bounding_box_test" / "0003_c3s1_000003_00.jpg").write_bytes(b"not-an-image")
            first = Path(directory) / "first.tsv"
            second = Path(directory) / "second.tsv"
            rows_one = analysis.build_fixed_manifest(root, first, limit=3)
            rows_two = analysis.build_fixed_manifest(root, second, limit=3)
            self.assertEqual(rows_one, rows_two)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            for row in rows_one:
                self.assertEqual(
                    row["legacy_sample_key"],
                    hashlib.sha256(row["stable_sample_key"].encode("utf-8")).hexdigest(),
                )
                self.assertTrue(row["stable_sample_key"].startswith(("query|", "gallery|")))

    def test_sha_mismatch_does_not_expose_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "gating_samples.tsv"
            artifact.write_text("evidence", encoding="utf-8")
            row = {"gating_samples_path": str(artifact), "gating_samples_sha256": "0" * 64}
            found, status = analysis._find_artifact(row, "gating_samples")
            self.assertIsNone(found)
            self.assertEqual(status, "not_archived")

    def test_fixed_pairing_refuses_a_missing_candidate(self):
        spec = analysis.SPEC_BY_KEY["g2"]
        candidates = [{"stable_sample_key": "s", "legacy_sample_key": "legacy"}]
        with self.assertRaises(analysis.AnalysisError):
            analysis.pair_fixed_samples(candidates, {}, spec)

    def test_fixed_pairing_uses_the_same_legacy_key_for_both_variants(self):
        spec = analysis.SPEC_BY_KEY["g2_without_z4"]
        candidates = [{"stable_sample_key": "readable", "legacy_sample_key": "legacy"}]
        paired = analysis.pair_fixed_samples(
            candidates,
            {"legacy": {"probabilities": {2: 0.5, 6: 0.5}, "weights": {2: 0.5, 6: 0.5}, "dominant": 2}},
            spec,
        )
        self.assertIn("readable", paired)

    def test_markdown_is_written_from_the_same_row_values_as_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = ("baseline", "variant", "status")
            rows = [{"baseline": analysis.BASELINE_LABEL, "variant": "G1", "status": analysis.MISSING_FORMAL}]
            analysis._write_csv(root / "table.csv", fields, rows)
            analysis._write_markdown_from_rows(root / "table.md", "title", fields, rows)
            with (root / "table.csv").open(encoding="utf-8", newline="") as handle:
                csv_values = next(csv.DictReader(handle))
            markdown = (root / "table.md").read_text(encoding="utf-8")
            for value in csv_values.values():
                self.assertIn(value, markdown)

    def test_paired_figure_data_and_figures_use_only_paired_fixed_samples(self):
        spec = analysis.SPEC_BY_KEY["g2_without_z6"]
        baseline = {
            "a": {"probabilities": {2: 0.2, 4: 0.3, 6: 0.5}, "weights": {2: 0.6, 4: 0.9, 6: 1.5}, "dominant": 6},
            "b": {"probabilities": {2: 0.5, 4: 0.4, 6: 0.1}, "weights": {2: 1.5, 4: 1.2, 6: 0.3}, "dominant": 2},
        }
        variant = {
            "a": {"probabilities": {2: 0.4, 4: 0.6}, "weights": {2: 0.4, 4: 0.6}, "dominant": 4},
            "b": {"probabilities": {2: 0.7, 4: 0.3}, "weights": {2: 0.7, 4: 0.3}, "dominant": 2},
        }
        paired_rows = analysis.paired_sample_rows(spec, variant, baseline)
        self.assertEqual([row["stable_sample_key"] for row in paired_rows], ["a", "b"])
        self.assertEqual(paired_rows[0]["variant_p6"], "excluded")
        self.assertAlmostEqual(paired_rows[0]["delta_p2_vs_g2"], 0.2)
        with tempfile.TemporaryDirectory() as directory:
            paths = analysis.generate_pair_figures(spec, variant, baseline, directory)
            self.assertEqual(len(paths), 12)
            self.assertTrue(all(Path(path).is_file() for path in paths))
if __name__ == "__main__":
    unittest.main()
