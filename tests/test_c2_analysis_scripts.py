import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.analyze_cross_camera_batch_coverage import (
    _prepare_new_output_dir as prepare_coverage_output_dir,
    analyze_batch,
    summarize_rows,
)
from tools.analyze_distance_distributions import (
    PAIR_TYPE_TO_CODE,
    _assert_manifest_pid_policy,
    _audit_raw_image_sources,
    _build_sample_manifest,
    _build_separation_summary,
    _filter_eval_samples,
    _load_cfg,
    _prepare_new_output_dir as prepare_distance_output_dir,
    _resolve_pid_filter,
    _validate_protocols,
    generate_pair_indices,
    l2_normalize,
    pairwise_squared_euclidean,
    summarize_distances,
)


class DistanceAnalysisTest(unittest.TestCase):
    def test_market_raw_audit_separates_junk_background_and_people(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_dir = root / "query"
            gallery_dir = root / "gallery"
            query_dir.mkdir()
            gallery_dir.mkdir()
            for filename in ("0001_c1_a.jpg", "0002_c2_b.jpg"):
                (query_dir / filename).write_bytes(b"")
            for filename in (
                "-1_c1_junk.jpg",
                "0000_c2_background.jpg",
                "0003_c3_person.jpg",
                "unparsed.jpg",
            ):
                (gallery_dir / filename).write_bytes(b"")

            audit = _audit_raw_image_sources(
                SimpleNamespace(
                    query_dir=str(query_dir), gallery_dir=str(gallery_dir)
                )
            )

        self.assertEqual(audit["query"]["positive_pid_count"], 2)
        self.assertEqual(audit["gallery"]["pid_minus_one_count"], 1)
        self.assertEqual(audit["gallery"]["pid_zero_count"], 1)
        self.assertEqual(audit["gallery"]["positive_pid_count"], 1)
        self.assertEqual(audit["gallery"]["unparsed_count"], 1)

    def test_market_positive_only_filter_precedes_manifest(self):
        query = [
            ("q_junk.jpg", -1, 0),
            ("q_background.jpg", 0, 1),
            ("q_person.jpg", 2, 2),
        ]
        gallery = [
            ("g_person.jpg", 1, 0),
            ("g_background.jpg", 0, 3),
        ]
        samples, splits, report = _filter_eval_samples(
            query, gallery, "market1501", "positive-only"
        )
        manifest = _build_sample_manifest(samples, splits)

        self.assertEqual([row["pid"] for row in manifest], [2, 1])
        self.assertEqual([row["split"] for row in manifest], ["query", "gallery"])
        self.assertEqual([row["sample_index"] for row in manifest], [0, 1])
        self.assertEqual(report["excluded_by_pid"], {"-1": 1, "0": 2})
        _assert_manifest_pid_policy(manifest, "positive-only")

    def test_auto_pid_filter_is_dataset_specific(self):
        self.assertEqual(
            _resolve_pid_filter("market1501", "auto"), "positive-only"
        )
        self.assertEqual(_resolve_pid_filter("dukemtmc", "auto"), "none")

    def test_positive_only_policy_rejects_nonpositive_manifest(self):
        manifest = [
            {
                "sample_index": 0,
                "image_path": "background.jpg",
                "pid": 0,
                "camid": 0,
                "split": "gallery",
            }
        ]
        with self.assertRaises(RuntimeError):
            _assert_manifest_pid_policy(manifest, "positive-only")

    def test_pair_types_are_mutually_exclusive_and_unordered(self):
        pids = np.asarray([1, 1, 1, 2])
        camids = np.asarray([0, 0, 1, 0])
        pair_i, pair_j, codes, sampling = generate_pair_indices(
            pids, camids, max_different_id_pairs=0, seed=7
        )
        self.assertTrue(np.all(pair_i < pair_j))
        self.assertEqual(len(set(zip(pair_i.tolist(), pair_j.tolist()))), 6)
        counts = {
            name: int((codes == code).sum())
            for name, code in PAIR_TYPE_TO_CODE.items()
        }
        self.assertEqual(counts["same-id same-camera"], 1)
        self.assertEqual(counts["same-id different-camera"], 2)
        self.assertEqual(counts["different-id"], 3)
        self.assertFalse(sampling["different_id_sampled"])

    def test_pair_sampling_and_distance_are_reproducible(self):
        pids = np.asarray([1, 1, 2, 2, 3, 3])
        camids = np.asarray([0, 1, 0, 1, 0, 1])
        first = generate_pair_indices(pids, camids, 4, 123)
        second = generate_pair_indices(pids, camids, 4, 123)
        for first_array, second_array in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(first_array, second_array)

        features = l2_normalize(
            np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        distances = pairwise_squared_euclidean(
            features,
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([1, 2], dtype=np.int64),
            chunk_size=1,
        )
        self.assertAlmostEqual(float(distances[0]), 2.0, places=6)
        self.assertEqual(summarize_distances(distances)["count"], 2)

    def test_cross_camera_separation_gap(self):
        pair_type_codes = np.asarray([1, 1, 2, 2], dtype=np.int8)
        baseline = np.asarray([0.5, 0.7, 1.2, 1.4])
        c2 = np.asarray([0.4, 0.5, 1.3, 1.5])
        rows = _build_separation_summary(pair_type_codes, baseline, c2)
        self.assertAlmostEqual(rows[0]["mean_gap"], 0.7)
        self.assertAlmostEqual(rows[1]["mean_gap"], 0.95)
        self.assertAlmostEqual(rows[2]["mean_gap"], 0.25)

    def test_c2_l03_protocol_identity_is_enforced(self):
        baseline = _load_cfg(
            "configs/softmax_triplet_c2_baseline_control_autodl.yml"
        )
        c2_l03 = _load_cfg(
            "configs/softmax_triplet_cross_camera_positive_lambda03_autodl.yml"
        )
        _validate_protocols(baseline, c2_l03)

        wrong_lambda = c2_l03.clone()
        wrong_lambda.defrost()
        wrong_lambda.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA = 0.5
        wrong_lambda.freeze()
        with self.assertRaises(ValueError):
            _validate_protocols(baseline, wrong_lambda)

    def test_nonempty_output_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "existing_result.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaises(FileExistsError):
                prepare_distance_output_dir(directory)
            with self.assertRaises(FileExistsError):
                prepare_coverage_output_dir(directory)


class CoverageAnalysisTest(unittest.TestCase):
    def test_valid_cross_camera_anchor_definition(self):
        samples = [
            {"pid": 1, "camid": 0},
            {"pid": 1, "camid": 1},
            {"pid": 2, "camid": 0},
            {"pid": 2, "camid": 0},
        ]
        row = analyze_batch(samples, [0, 1, 2, 3], epoch=1, batch_index=0)
        self.assertEqual(row["valid_cross_camera_anchor_count"], 2)
        self.assertEqual(row["valid_cross_camera_anchor_ratio"], 0.5)
        self.assertEqual(row["cross_camera_positive_ordered_pair_count"], 2)
        self.assertEqual(row["all_same_id_positive_ordered_pair_count"], 4)
        summary = summarize_rows([row])
        self.assertTrue(0.0 <= summary["weighted_valid_anchor_ratio"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
