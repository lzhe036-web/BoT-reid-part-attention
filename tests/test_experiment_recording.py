# encoding: utf-8
"""Synthetic, filesystem-isolated tests for the bypass recording system."""

from __future__ import absolute_import

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from tools.analyze_anchor_coverage import summarize_batches
from tools.analyze_distance_distributions import compute_distance_statistics
from utils.experiment_recording import (
    AUTO_RESULTS_START,
    EvidenceError,
    MAIN_FIELDS,
    NOT_RECORDED,
    atomic_write_json,
    atomic_write_text,
    config_modules,
    csv_to_markdown,
    experiment_identity,
    finalize_run,
    git_metadata,
    normalized_path,
    sha256_file,
    validate_git_preflight,
)


FULL_COMMIT = "a" * 40


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_tsv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _base_config(output_dir, variant="cross", seed=42):
    model = {
        "NAME": "resnet50",
        "NECK": "bnneck",
        "PART_ATTENTION": True,
        "PART_ATTENTION_PARTS": 6,
        "CAMERA_AWARE_TRIPLET": False,
        "CROSS_CAMERA_POSITIVE_ONLY": False,
        "CROSS_CAMERA_POSITIVE_LAMBDA": 0.3,
        "CROSS_CAMERA_POSITIVE_MODE": "mean",
        "PART_CORRESPONDENCE_CONSISTENCY": False,
        "PCC_PARTS": 6,
        "PCC_LAMBDA": 0.1,
        "PCC_MODE": "fixed_index",
    }
    if variant == "cross":
        model["CROSS_CAMERA_POSITIVE_ONLY"] = True
    elif variant == "same":
        model["SAME_CAMERA_POSITIVE_ONLY"] = True
        model["SAME_CAMERA_POSITIVE_LAMBDA"] = 0.3
        model["SAME_CAMERA_POSITIVE_MODE"] = "mean"
    elif variant == "caat":
        model["CAMERA_AWARE_TRIPLET"] = True
        model["CAMERA_AWARE_TRIPLET_LAMBDA"] = 0.5
        model["CAMERA_AWARE_TRIPLET_MODE"] = "hard"
        model["HIERARCHICAL_CAMERA_AWARE_LOSS"] = True
        model["NORMALIZED_WEIGHTED_LOSS"] = True
        model["MULTI_GRANULARITY_PART"] = True
    elif variant == "pcc":
        model["CROSS_CAMERA_POSITIVE_ONLY"] = True
        model["PART_CORRESPONDENCE_CONSISTENCY"] = True
    config = {
        "SEED": seed,
        "MODEL": model,
        "DATASETS": {"NAMES": "market1501", "ROOT_DIR": "/synthetic/data"},
        "DATALOADER": {
            "SAMPLER": "softmax_triplet", "NUM_INSTANCE": 4, "NUM_WORKERS": 0,
        },
        "SOLVER": {"MARGIN": 0.3, "IMS_PER_BATCH": 64},
        "OUTPUT_DIR": str(output_dir),
    }
    return config


def _training_log(config_text):
    pcc_enabled = "PART_CORRESPONDENCE_CONSISTENCY: true" in config_text
    first_pcc = ""
    second_pcc = ""
    first_summary = []
    second_summary = []
    if pcc_enabled:
        first_pcc = ", loss_pcc: 0.2, valid_pcc_pair_count: 12.0, mean_fixed_index_part_distance: 0.2"
        second_pcc = ", loss_pcc: 0.1, valid_pcc_pair_count: 14.0, mean_fixed_index_part_distance: 0.1"
        first_summary = [
            "2026-08-07 10:10:00 reid_baseline.train INFO: PCC Epoch Summary - Epoch: 1 valid_pcc_pair_count: 1200 mean_fixed_index_part_distance: 0.200000"
        ]
        second_summary = [
            "2026-08-07 11:00:00 reid_baseline.train INFO: PCC Epoch Summary - Epoch: 2 valid_pcc_pair_count: 1400 mean_fixed_index_part_distance: 0.100000"
        ]
    return "\n".join([
        "2026-08-07 10:00:00 reid_baseline.train INFO: Start training",
        "2026-08-07 10:00:00 reid_baseline.train INFO: Loaded configuration file configs/synthetic.yml",
        config_text.rstrip("\n"),
        "2026-08-07 10:10:00 reid_baseline.train INFO: Epoch[1] Iteration[100/100] loss_total: 1.0, loss_id: 0.4, loss_triplet: 0.5, loss_camera_triplet: 0.0, loss_cross_camera_positive: 0.1{}, cross_camera_positive_count: 60.0, Acc: 0.8, Base Lr: 3.50e-04".format(first_pcc),
    ] + first_summary + [
        "2026-08-07 10:10:01 reid_baseline.train INFO: Validation Results - Epoch: 1",
        "2026-08-07 10:10:02 reid_baseline.train INFO: mAP: 80.0%",
        "2026-08-07 10:10:03 reid_baseline.train INFO: CMC curve, Rank-1  :90.0%",
        "2026-08-07 10:10:04 reid_baseline.train INFO: CMC curve, Rank-5  :96.0%",
        "2026-08-07 10:10:05 reid_baseline.train INFO: CMC curve, Rank-10 :98.0%",
        "2026-08-07 11:00:00 reid_baseline.train INFO: Epoch[2] Iteration[100/100] loss_total: 0.8, loss_id: 0.3, loss_triplet: 0.4, loss_camera_triplet: 0.0, loss_cross_camera_positive: 0.1{}, cross_camera_positive_count: 61.0, Acc: 0.9, Base Lr: 3.50e-04".format(second_pcc),
    ] + second_summary + [
        "2026-08-07 11:00:01 reid_baseline.train INFO: Validation Results - Epoch: 2",
        "2026-08-07 11:00:02 reid_baseline.train INFO: mAP: 87.8%",
        "2026-08-07 11:00:03 reid_baseline.train INFO: CMC curve, Rank-1  :95.0%",
        "2026-08-07 11:00:04 reid_baseline.train INFO: CMC curve, Rank-5  :98.5%",
        "2026-08-07 11:00:05 reid_baseline.train INFO: CMC curve, Rank-10 :99.0%",
        "",
    ])


def make_fixture(root, run_id="run-001", variant="cross", family="c2_lambda",
                 experiment_id="C2-L03", seed=42, include_log=True,
                 include_checkpoint=True, applied_seed=None,
                 training_exit_code=0, include_efficiency=True):
    root = Path(root)
    records = root / "experiment_records"
    run_dir = records / "runs" / run_id
    output = root / ("output-" + run_id)
    run_dir.mkdir(parents=True)
    output.mkdir(parents=True)
    config = _base_config(output, variant=variant, seed=seed)
    config_text = yaml.safe_dump(config, sort_keys=True)
    source = run_dir / "config_source.yml"
    resolved = run_dir / "config_resolved.yml"
    atomic_write_text(source, config_text)
    atomic_write_text(resolved, config_text)
    identity = experiment_identity(config)
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "experiment_family": family,
        "method": identity["method"],
        "dataset": identity["dataset"],
        "branch": "C2L03",
        "commit_id": FULL_COMMIT,
        "expected_branch": "C2L03",
        "config_file": "configs/synthetic.yml",
        "config_source_sha256": sha256_file(source),
        "config_resolved_sha256": sha256_file(resolved),
        "seed": seed,
        "lambda": identity["lambda"],
        "cross_camera_positive_lambda": identity["cross_camera_positive_lambda"],
        "pcc_enabled": identity["pcc_enabled"],
        "pcc_parts": identity["pcc_parts"],
        "pcc_lambda": identity["pcc_lambda"],
        "pcc_mode": identity["pcc_mode"],
        "alignment_strategy": identity["alignment_strategy"],
        "baseline": identity["baseline"],
        "margin": identity["margin"],
        "mode": identity["mode"],
        "modules": identity["modules"],
        "output_dir": normalized_path(output.resolve()),
        "start_time": "2026-08-07T10:00:00Z",
        "notes": "合成 fixture",
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "run_status.json", {
        "schema_version": 1,
        "run_id": run_id,
        "status": "training_complete" if training_exit_code == 0 else "failed",
        "training_exit_code": training_exit_code,
        "training_runtime_seconds": 3600.0,
        "training_end_time": "2026-08-07T11:00:05Z",
    })
    atomic_write_json(run_dir / "environment.json", {
        "gpus": [{"name": "Synthetic GPU", "total_memory_bytes": 1234}],
        "gpu_count": 1,
        "git_branch": "C2L03",
        "git_commit": FULL_COMMIT,
    })
    atomic_write_text(run_dir / "environment_packages.txt", "torch==synthetic\n")
    atomic_write_json(run_dir / "dataset_manifest.json", {
        "dataset": "market1501", "dataset_manifest_sha256": "b" * 64,
    })
    atomic_write_json(run_dir / "model_manifest.json", {
        "backbone": "resnet50", "modules": config_modules(config),
    })
    evidence_seed = seed if applied_seed is None else applied_seed
    atomic_write_json(output / "reproducibility.json", {
        "source_seed": seed,
        "resolved_seed": seed,
        "applied_seed": evidence_seed,
        "seed": evidence_seed,
        "PYTHONHASHSEED": str(evidence_seed),
        "python_random_seed": evidence_seed,
        "numpy_seed": evidence_seed,
        "torch_cpu_seed": evidence_seed,
        "torch_cuda_seed": evidence_seed,
        "dataloader_worker_seed_base": evidence_seed,
        "dataloader_worker_seed_strategy": "synthetic worker fixture",
        "sampler_seed": evidence_seed,
        "sampler_seed_strategy": "synthetic sampler fixture",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    })
    if include_log:
        atomic_write_text(output / "log.txt", _training_log(config_text))
    if include_checkpoint:
        (output / "resnet50_checkpoint_200.pt").write_bytes(b"synthetic checkpoint")
    atomic_write_json(run_dir / "distance_distribution.json", {
        "same_id_same_camera_mean": 0.4,
        "same_id_same_camera_std": 0.1,
        "same_id_cross_camera_mean": 0.6,
        "same_id_cross_camera_std": 0.2,
        "different_id_mean": 1.2,
        "different_id_std": 0.3,
        "cross_camera_gap": 0.2,
    })
    atomic_write_json(run_dir / "anchor_coverage.json", {
        "total_anchor_count": 1000,
        "valid_cross_camera_anchor_count": 900,
        "invalid_cross_camera_anchor_count": 100,
        "coverage_percent": 90.0,
        "cross_camera_positive_count": 2400,
        "same_camera_positive_count": 600,
    })
    if include_efficiency:
        atomic_write_json(run_dir / "efficiency_profile.json", {
            "status": "complete", "total_params": 25,
            "trainable_params": 24, "FLOPs": 100, "MACs": 50,
        })
    experiments = root / "EXPERIMENTS.md"
    if not experiments.exists():
        atomic_write_text(experiments, "# Experiments\n\n| historical | 95.0% | 87.8% |\n")
    return records, run_dir, output, experiments


def finalize_fixture(records, run_dir, experiments):
    return finalize_run(
        run_dir=run_dir,
        records_root=records,
        repo_root=run_dir,
        experiments_path=experiments,
        run_analyses=False,
        verify_git=False,
    )


class ExperimentRecordingTest(unittest.TestCase):
    def test_success_experiment_writes_main_results(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, output, experiments = make_fixture(directory)
            result = finalize_fixture(records, run_dir, experiments)
            rows = _read_csv(records / "tables" / "main_results.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["rank1"], "95")
            self.assertEqual(rows[0]["log_sha256"], sha256_file(output / "log.txt"))
            self.assertEqual(result["status"]["status"], "success")

    def test_lambda_experiment_writes_lambda_table(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(directory)
            finalize_fixture(records, run_dir, experiments)
            rows = _read_csv(records / "tables" / "lambda_sensitivity.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["lambda"], "0.3")

    def test_pcc_experiment_writes_explicit_lambda_fields_and_pcc_table(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(
                directory, variant="pcc",
                family="c2l03_fixed_index_part_correspondence_consistency",
                experiment_id="C2-PCC-Fixed",
            )
            records, run_dir, _output, experiments = fixture
            finalize_fixture(records, run_dir, experiments)
            main = _read_csv(records / "tables" / "main_results.csv")[0]
            pcc = _read_csv(records / "tables" / "pcc_ablation.csv")[0]
            self.assertEqual(main["cross_camera_positive_lambda"], "0.3")
            self.assertEqual(main["pcc_lambda"], "0.1")
            self.assertEqual(main["alignment_strategy"], "fixed_index")
            self.assertEqual(pcc["valid_pcc_pair_count"], "2600")
            self.assertAlmostEqual(
                float(pcc["mean_fixed_index_part_distance"]),
                (1200 * 0.2 + 1400 * 0.1) / 2600.0,
            )

    def test_same_camera_experiment_writes_same_camera_table(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(
                directory, variant="same", family="same_camera_ablation",
                experiment_id="S2-TEST",
            )
            records, run_dir, _output, experiments = fixture
            finalize_fixture(records, run_dir, experiments)
            rows = _read_csv(
                records / "tables" / "same_camera_positive_ablation.csv"
            )
            self.assertEqual(rows[0]["variant"], "same_camera_positive")
            self.assertEqual(rows[0]["positive_relation"], "same_camera")

    def test_caat_flags_are_derived_from_config(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(
                directory, variant="caat", family="caat_ablation",
                experiment_id="CAAT-TEST",
            )
            records, run_dir, _output, experiments = fixture
            finalize_fixture(records, run_dir, experiments)
            rows = _read_csv(records / "tables" / "caat_ablation.csv")
            self.assertEqual(rows[0]["camera_aware_triplet"], "True")
            self.assertEqual(rows[0]["hierarchical"], "True")
            self.assertEqual(rows[0]["weighted"], "True")
            self.assertEqual(rows[0]["multi_granularity"], "True")

    def test_distance_analysis_writes_distance_table(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(directory)
            finalize_fixture(records, run_dir, experiments)
            rows = _read_csv(records / "tables" / "distance_distribution.csv")
            self.assertEqual(rows[0]["same_id_cross_camera_mean"], "0.6")

    def test_anchor_analysis_writes_coverage_table(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(directory)
            finalize_fixture(records, run_dir, experiments)
            rows = _read_csv(records / "tables" / "anchor_coverage.csv")
            self.assertEqual(rows[0]["total_anchors"], "1000")
            self.assertEqual(rows[0]["coverage_percent"], "90.0")

    def test_same_run_id_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(directory)
            finalize_fixture(records, run_dir, experiments)
            finalize_fixture(records, run_dir, experiments)
            rows = _read_csv(records / "tables" / "main_results.csv")
            self.assertEqual(len(rows), 1)

    def test_different_run_ids_append(self):
        with tempfile.TemporaryDirectory() as directory:
            first = make_fixture(directory, run_id="run-001")
            finalize_fixture(first[0], first[1], first[3])
            second = make_fixture(directory, run_id="run-002")
            finalize_fixture(second[0], second[1], second[3])
            rows = _read_csv(first[0] / "tables" / "main_results.csv")
            self.assertEqual({row["run_id"] for row in rows}, {"run-001", "run-002"})

    def test_missing_checkpoint_does_not_register_success(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(
                directory, include_checkpoint=False
            )
            with self.assertRaises(EvidenceError):
                finalize_fixture(records, run_dir, experiments)
            self.assertEqual(_read_csv(records / "tables" / "main_results.csv"), [])

    def test_missing_log_does_not_register_success(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(
                directory, include_log=False
            )
            with self.assertRaises(EvidenceError):
                finalize_fixture(records, run_dir, experiments)
            self.assertEqual(_read_csv(records / "tables" / "main_results.csv"), [])

    def test_config_log_mismatch_does_not_register_success(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, output, experiments = make_fixture(directory)
            log_path = output / "log.txt"
            log_text = log_path.read_text(encoding="utf-8")
            atomic_write_text(
                log_path,
                log_text.replace("CROSS_CAMERA_POSITIVE_LAMBDA: 0.3", ""),
            )
            with self.assertRaisesRegex(EvidenceError, "exact source config"):
                finalize_fixture(records, run_dir, experiments)
            self.assertEqual(_read_csv(records / "tables" / "main_results.csv"), [])

    def test_seed_conflict_does_not_register_success(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(
                directory, applied_seed=7
            )
            with self.assertRaises(ValueError):
                finalize_fixture(records, run_dir, experiments)
            self.assertEqual(_read_csv(records / "tables" / "main_results.csv"), [])

    def test_missing_applied_seed_remains_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(
                directory, applied_seed=NOT_RECORDED
            )
            with self.assertRaisesRegex(EvidenceError, "Applied seed evidence"):
                finalize_fixture(records, run_dir, experiments)
            status = json.loads(
                (run_dir / "run_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "incomplete")
            self.assertEqual(_read_csv(records / "tables" / "main_results.csv"), [])

    def test_dirty_git_preflight_rejects_formal_training(self):
        with mock.patch(
            "utils.experiment_recording.git_metadata",
            return_value={"commit": FULL_COMMIT, "branch": "C2L03", "dirty": True},
        ):
            with self.assertRaisesRegex(EvidenceError, "clean Git"):
                validate_git_preflight("unused", "C2L03")

    def test_post_init_git_check_allows_only_current_run_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            run_dir = repo / "experiment_records" / "runs" / "run-001"
            run_dir.mkdir(parents=True)

            def controlled_output(_repo, args):
                if args[0] == "rev-parse":
                    return FULL_COMMIT
                if args[0] == "branch":
                    return "exp/c2l03-fixed-index-pcc"
                return "?? experiment_records/runs/run-001/run_manifest.json"

            with mock.patch(
                "utils.experiment_recording._git_output",
                side_effect=controlled_output,
            ):
                metadata = git_metadata(
                    repo, allowed_dirty_paths=(run_dir,)
                )
            self.assertFalse(metadata["dirty"])
            self.assertEqual(
                metadata["controlled_dirty_paths"],
                ["experiment_records/runs/run-001/run_manifest.json"],
            )

            def polluted_output(_repo, args):
                if args[0] == "rev-parse":
                    return FULL_COMMIT
                if args[0] == "branch":
                    return "exp/c2l03-fixed-index-pcc"
                return "\n".join([
                    "?? experiment_records/runs/run-001/run_manifest.json",
                    " M modeling/baseline.py",
                ])

            with mock.patch(
                "utils.experiment_recording._git_output",
                side_effect=polluted_output,
            ):
                metadata = git_metadata(
                    repo, allowed_dirty_paths=(run_dir,)
                )
            self.assertTrue(metadata["dirty"])
            self.assertEqual(metadata["dirty_paths"], ["modeling/baseline.py"])

    def test_checkpoint_sha256_is_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, output, experiments = make_fixture(directory)
            finalize_fixture(records, run_dir, experiments)
            rows = _read_csv(records / "tables" / "main_results.csv")
            expected = hashlib.sha256(b"synthetic checkpoint").hexdigest()
            self.assertEqual(rows[0]["checkpoint_sha256"], expected)
            self.assertEqual(
                sha256_file(output / "resnet50_checkpoint_200.pt"), expected
            )

    def test_checkpoint_global_iteration_is_mapped_to_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(directory)
            finalize_fixture(records, run_dir, experiments)
            rows = _read_tsv(run_dir / "checkpoint_manifest.tsv")
            self.assertEqual(rows[0]["global_iteration"], "200")
            self.assertEqual(rows[0]["epoch"], "2")
            self.assertEqual(rows[0]["selected"], "True")
            validation = [
                json.loads(line)
                for line in (run_dir / "validation_history.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(validation[-1]["global_iteration"], 200)

    def test_csv_to_markdown_is_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(directory)
            finalize_fixture(records, run_dir, experiments)
            csv_rows = _read_csv(records / "tables" / "main_results.csv")
            markdown = (records / "tables" / "main_results.md").read_text(
                encoding="utf-8"
            )
            self.assertIn(csv_rows[0]["run_id"], markdown)
            self.assertIn(csv_rows[0]["checkpoint_sha256"], markdown)

    def test_failed_training_does_not_pollute_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(
                directory, training_exit_code=3, include_log=False,
                include_checkpoint=False,
            )
            with self.assertRaises(EvidenceError):
                finalize_fixture(records, run_dir, experiments)
            status = json.loads(
                (run_dir / "run_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "failed")
            self.assertEqual(_read_csv(records / "tables" / "main_results.csv"), [])

    def test_historical_experiments_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(directory)
            historical = experiments.read_text(encoding="utf-8")
            finalize_fixture(records, run_dir, experiments)
            updated = experiments.read_text(encoding="utf-8")
            self.assertIn(historical.strip(), updated)
            self.assertIn(AUTO_RESULTS_START, updated)

    def test_missing_efficiency_is_not_fabricated_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            records, run_dir, _output, experiments = make_fixture(
                directory, include_efficiency=False
            )
            finalize_fixture(records, run_dir, experiments)
            rows = _read_csv(records / "tables" / "caat_ablation.csv")
            self.assertEqual(rows[0]["Params"], NOT_RECORDED)
            self.assertEqual(rows[0]["FLOPs"], NOT_RECORDED)

    def test_windows_and_linux_paths_normalize(self):
        self.assertEqual(normalized_path(r"C:\runs\x\log.txt"), "C:/runs/x/log.txt")
        self.assertEqual(normalized_path("/root/runs/x/log.txt"), "/root/runs/x/log.txt")

    def test_utf8_atomic_writes_use_lf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utf8.txt"
            atomic_write_text(path, "实验\r\n第二行\r\n")
            raw = path.read_bytes()
            self.assertIn("实验".encode("utf-8"), raw)
            self.assertNotIn(b"\r\n", raw)

    def test_distance_statistics_include_raw_percentiles(self):
        features = np.asarray([
            [0.0, 0.0], [0.1, 0.0], [0.3, 0.0],
            [1.0, 0.0], [1.2, 0.0],
        ])
        result = compute_distance_statistics(
            features, [1, 1, 1, 2, 2], [0, 0, 1, 0, 1]
        )
        self.assertIn("p95", result["same_id_cross_camera"])
        self.assertGreater(result["different_id"]["count"], 0)

    def test_anchor_statistics_are_from_controlled_batches(self):
        samples = [
            ("a.jpg", 1, 0), ("b.jpg", 1, 1),
            ("c.jpg", 2, 0), ("d.jpg", 2, 0),
        ]
        result = summarize_batches(samples, [0, 1, 2, 3], 4)
        self.assertEqual(result["total_anchor_count"], 4)
        self.assertEqual(result["valid_cross_camera_anchor_count"], 2)


if __name__ == "__main__":
    unittest.main()
