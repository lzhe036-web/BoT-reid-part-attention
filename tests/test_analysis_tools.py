import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.append_experiment_result import (
    CROSS_CAMERA_POSITIVE_SECTION_TITLE,
    HEADER,
    PENDING,
    ensure_section,
    parse_metrics,
)
from tools.analyze_cross_camera_batch_coverage import (
    _prepare_new_output_dir as prepare_coverage_output_dir,
    analyze_batch,
    summarize_rows,
)
from tools.analyze_distance_distributions import (
    PAIR_TYPE_TO_CODE,
    _audit_raw_image_sources,
    _assert_manifest_pid_policy,
    _build_sample_manifest,
    _build_separation_summary,
    _build_summary,
    _filter_eval_samples,
    _load_cfg,
    _plot_distributions,
    _prepare_new_output_dir as prepare_distance_output_dir,
    _resolve_pid_filter,
    _validate_protocols,
    _write_summary,
    generate_pair_indices,
    l2_normalize,
    pairwise_squared_euclidean,
    summarize_distances,
)


class DistanceAnalysisTest(unittest.TestCase):
    def test_raw_source_audit_distinguishes_junk_background_and_people(self):
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
        self.assertEqual(audit["gallery"]["jpg_count"], 4)
        self.assertEqual(audit["gallery"]["pid_minus_one_count"], 1)
        self.assertEqual(audit["gallery"]["pid_zero_count"], 1)
        self.assertEqual(audit["gallery"]["positive_pid_count"], 1)
        self.assertEqual(audit["gallery"]["unparsed_count"], 1)

    def test_positive_only_filter_runs_before_manifest_and_preserves_order(self):
        query = [
            ("q_junk.jpg", -1, 0),
            ("q_background.jpg", 0, 1),
            ("q_person.jpg", 2, 2),
        ]
        gallery = [
            ("g_person.jpg", 1, 0),
            ("g_background.jpg", 0, 3),
        ]
        samples, split_names, report = _filter_eval_samples(
            query, gallery, "market1501", "positive-only"
        )
        manifest = _build_sample_manifest(samples, split_names)

        self.assertEqual([row["pid"] for row in manifest], [2, 1])
        self.assertEqual([row["split"] for row in manifest], ["query", "gallery"])
        self.assertEqual([row["sample_index"] for row in manifest], [0, 1])
        self.assertEqual(report["excluded_nonpositive_count"], 3)
        self.assertEqual(report["excluded_by_pid"], {"-1": 1, "0": 2})
        _assert_manifest_pid_policy(manifest, "positive-only")

    def test_market_auto_filter_resolves_to_positive_only(self):
        self.assertEqual(
            _resolve_pid_filter("market1501", "auto"), "positive-only"
        )

    def test_duke_auto_filter_is_none_and_samples_are_unchanged(self):
        query = [("q1.jpg", 1, 0)]
        gallery = [("g1.jpg", 2, 1)]
        samples, split_names, report = _filter_eval_samples(
            query, gallery, "dukemtmc", "auto"
        )
        self.assertEqual(samples, query + gallery)
        self.assertEqual(split_names, ["query", "gallery"])
        self.assertEqual(report["effective_policy"], "none")
        self.assertEqual(report["excluded_nonpositive_count"], 0)

    def test_positive_only_manifest_rejects_nonpositive_pid(self):
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

    def test_pair_categories_are_mutually_exclusive_and_unordered(self):
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

    def test_different_id_sampling_is_reproducible(self):
        pids = np.asarray([1, 1, 2, 2, 3, 3])
        camids = np.asarray([0, 1, 0, 1, 0, 1])
        first = generate_pair_indices(pids, camids, 4, 123)
        second = generate_pair_indices(pids, camids, 4, 123)
        for first_array, second_array in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(first_array, second_array)
        self.assertTrue(first[3]["different_id_sampled"])

    def test_distance_and_summary(self):
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
        summary = summarize_distances(distances)
        self.assertEqual(summary["count"], 2)
        self.assertGreaterEqual(summary["q95"], summary["median"])

    def test_synthetic_plots_are_written(self):
        pair_type_codes = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8)
        baseline = np.asarray([0.2, 0.3, 0.5, 0.6, 1.2, 1.4])
        c2 = np.asarray([0.2, 0.25, 0.4, 0.5, 1.25, 1.45])
        with tempfile.TemporaryDirectory() as directory:
            _plot_distributions(
                directory, pair_type_codes, baseline, c2
            )
            histogram = Path(directory) / "distance_histogram.png"
            boxplot = Path(directory) / "distance_boxplot.png"
            self.assertGreater(histogram.stat().st_size, 0)
            self.assertGreater(boxplot.stat().st_size, 0)

    def test_cross_camera_separation_gap(self):
        pair_type_codes = np.asarray([1, 1, 2, 2], dtype=np.int8)
        baseline = np.asarray([0.5, 0.7, 1.2, 1.4])
        c2 = np.asarray([0.4, 0.5, 1.3, 1.5])
        rows = _build_separation_summary(
            pair_type_codes, baseline, c2
        )
        self.assertAlmostEqual(rows[0]["mean_gap"], 0.7)
        self.assertAlmostEqual(rows[1]["mean_gap"], 0.95)
        self.assertAlmostEqual(rows[2]["mean_gap"], 0.25)

    def test_summary_outputs_include_separation_gap(self):
        pair_type_codes = np.asarray([0, 1, 2], dtype=np.int8)
        baseline = np.asarray([0.2, 0.5, 1.2])
        c2 = np.asarray([0.2, 0.4, 1.3])
        summary_rows = _build_summary(
            pair_type_codes, baseline, c2
        )
        separation_rows = _build_separation_summary(
            pair_type_codes, baseline, c2
        )
        with tempfile.TemporaryDirectory() as directory:
            _write_summary(directory, summary_rows, separation_rows)
            self.assertTrue(
                (Path(directory) / "distance_summary.json").is_file()
            )
            self.assertTrue(
                (Path(directory) / "separation_gap_summary.csv").is_file()
            )
            self.assertTrue(
                (Path(directory) / "separation_gap_summary.json").is_file()
            )

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

        wrong_normalization = c2_l03.clone()
        wrong_normalization.defrost()
        wrong_normalization.INPUT.PIXEL_MEAN = [0.0, 0.0, 0.0]
        wrong_normalization.freeze()
        with self.assertRaises(ValueError):
            _validate_protocols(baseline, wrong_normalization)

        caat_baseline = baseline.clone()
        caat_baseline.defrost()
        caat_baseline.MODEL.CAMERA_AWARE_TRIPLET = True
        caat_baseline.freeze()
        with self.assertRaises(ValueError):
            _validate_protocols(caat_baseline, c2_l03)

    def test_nonempty_output_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "existing_result.json"
            marker.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_distance_output_dir(directory)
            with self.assertRaises(FileExistsError):
                prepare_coverage_output_dir(directory)


class CoverageAnalysisTest(unittest.TestCase):
    def test_valid_anchor_definition_and_range(self):
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
        self.assertEqual(summary["zero_valid_batch_count"], 0)


class ExperimentRecorderTest(unittest.TestCase):
    def test_parse_metrics_keeps_rank5_and_rank10_from_best_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "log.txt"
            log_path.write_text(
                "\n".join(
                    [
                        "2026-07-18 10:00:00 Validation Results - Epoch: 40",
                        "2026-07-18 10:00:01 mAP: 70.0%",
                        "2026-07-18 10:00:02 CMC curve, Rank-1  :80.0%",
                        "2026-07-18 10:00:03 CMC curve, Rank-5  :90.0%",
                        "2026-07-18 10:00:04 CMC curve, Rank-10 :92.0%",
                        "2026-07-18 11:00:00 Validation Results - Epoch: 80",
                        "2026-07-18 11:00:01 mAP: 75.0%",
                        "2026-07-18 11:00:02 CMC curve, Rank-1  :85.0%",
                        "2026-07-18 11:00:03 CMC curve, Rank-5  :94.0%",
                        "2026-07-18 11:00:04 CMC curve, Rank-10 :96.0%",
                    ]
                ),
                encoding="utf-8",
            )
            metrics = parse_metrics(directory)
        self.assertEqual(metrics["best_epoch"], "80")
        self.assertEqual(metrics["rank1"], "85.0%")
        self.assertEqual(metrics["rank5"], "94.0%")
        self.assertEqual(metrics["rank10"], "96.0%")
        self.assertEqual(metrics["map"], "75.0%")

    def test_parse_metrics_points_to_the_log_that_supplied_best_result(self):
        with tempfile.TemporaryDirectory() as directory:
            best_log = Path(directory) / "older_best.log"
            latest_log = Path(directory) / "latest_lower.log"
            best_log.write_text(
                "\n".join(
                    [
                        "2026-07-18 10:00:00 Validation Results - Epoch: 80",
                        "2026-07-18 10:00:01 mAP: 75.0%",
                        "2026-07-18 10:00:02 CMC curve, Rank-1  :85.0%",
                    ]
                ),
                encoding="utf-8",
            )
            latest_log.write_text(
                "\n".join(
                    [
                        "2026-07-18 11:00:00 Validation Results - Epoch: 40",
                        "2026-07-18 11:00:01 mAP: 70.0%",
                        "2026-07-18 11:00:02 CMC curve, Rank-1  :80.0%",
                    ]
                ),
                encoding="utf-8",
            )
            os.utime(best_log, (1, 1))
            os.utime(latest_log, (2, 2))
            metrics = parse_metrics(directory)
        self.assertEqual(Path(metrics["log_path"]).name, best_log.name)
        self.assertEqual(metrics["rank1"], "85.0%")
        self.assertEqual(metrics["runtime"], "0:00:02")

    def test_legacy_experiment_rows_are_migrated_to_current_columns(self):
        legacy_header = (
            "| 实验编号 | 日期 | commit id | 分支 | 实验类型 | 数据集 | "
            "config 文件 | OUTPUT_DIR | 日志路径 | GPU | seed | lambda | "
            "运行时间 | best epoch | Rank-1 | mAP | 备注 |"
        )
        legacy_separator = "|" + "|".join(["---"] * 17) + "|"
        legacy_values = ["value{}".format(index) for index in range(17)]
        legacy_row = "| " + " | ".join(legacy_values) + " |"
        content = "\n".join(
            (
                "# Experiments",
                "",
                CROSS_CAMERA_POSITIVE_SECTION_TITLE,
                "",
                legacy_header,
                legacy_separator,
                legacy_row,
            )
        )
        migrated = ensure_section(
            content, CROSS_CAMERA_POSITIVE_SECTION_TITLE
        )
        lines = migrated.splitlines()
        self.assertIn(HEADER, lines)
        migrated_row = next(line for line in lines if line.startswith("| value0 |"))
        cells = [
            cell.strip() for cell in migrated_row.strip().strip("|").split("|")
        ]
        self.assertEqual(len(cells), 20)
        self.assertEqual(cells[15:17], [PENDING, PENDING])
        self.assertEqual(cells[18], PENDING)

    def test_pre_reranking_rows_gain_a_reranking_column(self):
        previous_values = ["value{}".format(index) for index in range(19)]
        previous_row = "| " + " | ".join(previous_values) + " |"
        content = "\n".join(
            (
                "# Experiments",
                "",
                CROSS_CAMERA_POSITIVE_SECTION_TITLE,
                "",
                "| previous 19-column header |",
                "|" + "|".join(["---"] * 19) + "|",
                previous_row,
            )
        )
        migrated = ensure_section(
            content, CROSS_CAMERA_POSITIVE_SECTION_TITLE
        )
        migrated_row = next(
            line for line in migrated.splitlines() if line.startswith("| value0 |")
        )
        cells = [
            cell.strip() for cell in migrated_row.strip().strip("|").split("|")
        ]
        self.assertEqual(len(cells), 20)
        self.assertEqual(cells[18], PENDING)
        self.assertEqual(cells[19], "value18")


if __name__ == "__main__":
    unittest.main()
