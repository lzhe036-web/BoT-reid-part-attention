import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.run_experiment import (
    TrainingInterrupted,
    _require_successful_smoke_before_formal,
    launch_training_subprocess,
)
from utils.dynamic_gating_evidence import (
    GatingEpochAccumulator,
    append_gating_epoch_record,
    read_gating_epoch_records,
)
from utils.dynamic_experiment_registry import (
    GATING_STAT_FIELDS,
    RUN_FIELDS,
    DynamicExperimentEvidenceError,
    build_dynamic_checkpoint_manifest,
    initialize_dynamic_run,
    migrate_unified_schema,
    read_json,
    refresh_experiments_markdown,
    register_dynamic_run_state,
    select_dynamic_checkpoint,
    seal_dynamic_run_evidence,
    transition_dynamic_run,
    validate_seed42_reproducibility,
)
from utils.multigranularity_signatures import STATIC_BASELINE_BRANCH, STATIC_BASELINE_SHA
import utils.dynamic_experiment_registry as registry


def feature_evidence():
    return {
        "feature_reference_commit": STATIC_BASELINE_SHA,
        "feature_reference_signature_sha256": "a" * 64,
        "current_feature_signature_sha256": "a" * 64,
        "feature_compatibility_status": "compatible",
        "mismatched_components": [],
        "components": {},
        "fusion_gating_signature": {
            "reference_sha256": "b" * 64,
            "current_sha256": "c" * 64,
            "status": "expected_experiment_difference",
        },
    }


def lineage():
    return {
        "branch": "exp/c2-l03-multi-granularity-dynamic-gating",
        "commit": "d" * 40, "parent_branch": STATIC_BASELINE_BRANCH,
        "parent_commit": STATIC_BASELINE_SHA, "merge_base": STATIC_BASELINE_SHA,
    }


class ConsoleTeeTest(unittest.TestCase):
    def test_stdout_stderr_unicode_and_zero_exit_are_teed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "console.log"
            code = (
                "import sys; print('stdout 中文', flush=True); "
                "print('stderr Ω', file=sys.stderr, flush=True)"
            )
            with mock.patch("tools.run_experiment.sys.stdout") as terminal:
                return_code, _runtime = launch_training_subprocess(
                    [sys.executable, "-c", code], os.environ.copy(), path
                )
            content = path.read_text(encoding="utf-8")
            self.assertEqual(return_code, 0)
            self.assertIn("stdout 中文", content)
            self.assertIn("stderr Ω", content)
            self.assertTrue(terminal.write.called)
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                registry.sha256_file(path),
            )

    def test_nonzero_exit_and_traceback_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "console.log"
            return_code, _runtime = launch_training_subprocess(
                [sys.executable, "-c", "raise RuntimeError('synthetic failure')"],
                os.environ.copy(), path,
            )
            self.assertNotEqual(return_code, 0)
            content = path.read_text(encoding="utf-8")
            self.assertIn("Traceback", content)
            self.assertIn("synthetic failure", content)

    def test_each_run_uses_an_independent_console_log(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name / "console.log" for name in ("a", "b")]
            for index, path in enumerate(paths):
                launch_training_subprocess(
                    [sys.executable, "-c", "print({})".format(index)],
                    os.environ.copy(), path,
                )
            self.assertNotEqual(paths[0], paths[1])
            self.assertNotEqual(paths[0].read_bytes(), paths[1].read_bytes())

    def test_interrupt_terminates_child_and_closes_preserved_log(self):
        class FakePipe(object):
            def __init__(self):
                self.calls = 0
                self.closed = False

            def read(self, _size=-1):
                self.calls += 1
                if self.calls == 1:
                    raise KeyboardInterrupt()
                return b"final child diagnostic\n"

            def close(self):
                self.closed = True

        class FakeProcess(object):
            def __init__(self):
                self.stdout = FakePipe()
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

        process = FakeProcess()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "console.log"
            with mock.patch("tools.run_experiment.subprocess.Popen", return_value=process):
                with self.assertRaises(TrainingInterrupted):
                    launch_training_subprocess(
                        [sys.executable, "synthetic.py"], os.environ.copy(), path
                    )
            self.assertTrue(process.terminated)
            self.assertTrue(process.stdout.closed)
            self.assertIn("final child diagnostic", path.read_text(encoding="utf-8"))


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.records = self.root / "experiment_records"
        self.experiments = self.root / "EXPERIMENTS.md"
        self.experiments.write_text(
            "# Experiments\n\nManual explanation | stays.\n",
            encoding="utf-8", newline="\n",
        )
        self.config = self.root / "config.yml"
        self.config.write_text("SEED: 42\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def make_run(self, run_kind="smoke"):
        output = self.root / "output-{}".format(run_kind)
        run_dir, manifest = initialize_dynamic_run(
            self.records, self.experiments,
            "EXP-{}".format(run_kind.upper()), run_kind,
            self.config, "SEED: 42\n", output, lineage(), feature_evidence(),
            ["python", "train.py"],
        )
        return run_dir, manifest

    @staticmethod
    def successful_manifest(run_dir):
        manifest = read_json(run_dir / "run_manifest.json")
        accumulator = GatingEpochAccumulator(1.0)
        accumulator.update([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        manifest["gating_statistics"] = accumulator.summary()
        manifest["metrics"] = {
            "rank1_percent": 95.1, "rank5_percent": 98.0,
            "rank10_percent": 99.0, "map_percent": 88.2,
            "best_epoch": 1, "selected_epoch": 1,
        }
        manifest["status"] = "success"
        manifest["notes"] = "pipe | newline\nkept"
        status = read_json(run_dir / "run_status.json")
        status["status"] = "success"
        registry.atomic_write_json(run_dir / "run_status.json", status)
        return manifest

    def test_initialized_failed_incomplete_and_interrupted_enter_all_runs(self):
        run_dir, _ = self.make_run("smoke")
        for status in (
                "running", "training_complete", "finalizing", "failed",
                "incomplete", "interrupted"):
            transition_dynamic_run(run_dir, status, error=status)
            markdown = self.experiments.read_text(encoding="utf-8")
            self.assertIn(status, markdown)
        formal_section = self.experiments.read_text(encoding="utf-8").split(
            "## Formal Results", 1
        )[1].split("<!-- AUTO-EXPERIMENT-RESULTS:END -->", 1)[0]
        self.assertNotIn("EXP-SMOKE", formal_section)

    def test_control_file_hashes_refresh_for_every_state(self):
        run_dir, _ = self.make_run("smoke")
        states = (
            "initialized", "running", "training_complete", "finalizing",
            "failed", "incomplete", "interrupted",
        )
        observed = []
        for state in states:
            if state != "initialized":
                transition_dynamic_run(run_dir, state, error=state)
            with (self.records / "runs.csv").open(
                    encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["run_manifest_sha256"],
                registry.sha256_file(run_dir / "run_manifest.json"),
            )
            self.assertEqual(
                row["run_status_sha256"],
                registry.sha256_file(run_dir / "run_status.json"),
            )
            observed.append(row["run_manifest_sha256"])
            with (self.records / "evidence_manifest.tsv").open(
                    encoding="utf-8", newline="") as handle:
                evidence = list(csv.DictReader(handle, delimiter="\t"))
            controls = {
                item["artifact_type"]: item for item in evidence
                if item["artifact_type"] in ("run_manifest", "run_status")
            }
            self.assertEqual(set(controls), {"run_manifest", "run_status"})
            self.assertEqual(controls["run_manifest"]["sha256"],
                             row["run_manifest_sha256"])
            self.assertEqual(controls["run_status"]["sha256"],
                             row["run_status_sha256"])
        self.assertEqual(len(set(observed)), len(observed))

    def test_required_state_history_includes_real_finalizing_phase(self):
        run_dir, _ = self.make_run("smoke")
        for state in ("running", "training_complete", "finalizing"):
            transition_dynamic_run(run_dir, state)
        manifest = read_json(run_dir / "run_manifest.json")
        self.assertEqual(
            [item["status"] for item in manifest["state_history"]],
            ["initialized", "running", "training_complete", "finalizing"],
        )
        self.assertEqual(manifest["ended_at_utc"], "not_recorded")

    def test_formal_success_enters_both_sections_smoke_success_only_all_runs(self):
        smoke_dir, _ = self.make_run("smoke")
        smoke = self.successful_manifest(smoke_dir)
        registry.atomic_write_json(smoke_dir / "run_manifest.json", smoke)
        register_dynamic_run_state(smoke_dir)
        formal_dir, _ = self.make_run("formal")
        formal = self.successful_manifest(formal_dir)
        registry.atomic_write_json(formal_dir / "run_manifest.json", formal)
        register_dynamic_run_state(formal_dir)
        markdown = self.experiments.read_text(encoding="utf-8")
        formal_section = markdown.split("## Formal Results", 1)[1].split(
            "<!-- AUTO-EXPERIMENT-RESULTS:END -->", 1
        )[0]
        self.assertIn("EXP-FORMAL", formal_section)
        self.assertNotIn("EXP-SMOKE", formal_section)

    def test_all_dynamic_scalars_and_artifact_indexes_are_in_markdown(self):
        run_dir, _ = self.make_run("formal")
        manifest = self.successful_manifest(run_dir)
        detail = run_dir / "gating_samples.tsv"
        detail.write_text("p2\tp4\tp6\n", encoding="utf-8")
        manifest["gating_samples"] = {
            "path": str(detail), "size_bytes": detail.stat().st_size,
            "sha256": registry.sha256_file(detail),
            "source_checkpoint_sha256": "e" * 64,
            "selection_rule": "stable | deterministic",
        }
        registry.atomic_write_json(run_dir / "run_manifest.json", manifest)
        register_dynamic_run_state(run_dir)
        markdown = self.experiments.read_text(encoding="utf-8")
        for field in GATING_STAT_FIELDS:
            self.assertIn(field, markdown)
        self.assertIn("gating_samples.tsv", markdown)
        self.assertIn("e" * 64, markdown)
        self.assertIn("stable \\| deterministic", markdown)

    def test_checkpoint_evidence_contains_every_real_checkpoint(self):
        run_dir, _ = self.make_run("formal")
        manifest = self.successful_manifest(run_dir)
        checkpoint_manifest = run_dir / "checkpoint_manifest.tsv"
        checkpoint_manifest.write_text(
            "epoch\tglobal_iteration\trelative_path\tfile_size\tsha256\tselected\n"
            "40\t100\ta.pt\t5\t{}\tfalse\n"
            "80\t200\tb.pt\t7\t{}\ttrue\n".format("a" * 64, "b" * 64),
            encoding="utf-8", newline="\n",
        )
        manifest["checkpoint_manifest"] = {
            "path": str(checkpoint_manifest),
            "size_bytes": checkpoint_manifest.stat().st_size,
            "sha256": registry.sha256_file(checkpoint_manifest),
        }
        registry.atomic_write_json(run_dir / "run_manifest.json", manifest)
        register_dynamic_run_state(run_dir)
        checkpoint_section = self.experiments.read_text(encoding="utf-8").split(
            "## Checkpoint Evidence", 1
        )[1]
        self.assertIn("a.pt", checkpoint_section)
        self.assertIn("b.pt", checkpoint_section)

    def test_missing_checkpoint_uses_sentinel_and_makes_no_fake_row(self):
        self.make_run("smoke")
        markdown = self.experiments.read_text(encoding="utf-8")
        run_section = markdown.split("## Run Registry / All Recorded Runs", 1)[1]
        self.assertIn("not_recorded", run_section)
        checkpoint_section = markdown.split("## Checkpoint Evidence", 1)[1]
        self.assertNotIn(".pt", checkpoint_section)

    def test_generation_is_idempotent_and_preserves_manual_text(self):
        self.make_run("smoke")
        first = refresh_experiments_markdown(self.experiments, self.records)
        first_hash = hashlib.sha256(first.encode("utf-8")).hexdigest()
        second = refresh_experiments_markdown(self.experiments, self.records)
        self.assertEqual(first_hash, hashlib.sha256(second.encode("utf-8")).hexdigest())
        self.assertIn("Manual explanation | stays.", second)
        with (self.records / "runs.csv").open(encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_transaction_failure_keeps_existing_markdown_intact(self):
        run_dir, _ = self.make_run("smoke")
        before = self.experiments.read_bytes()
        manifest = read_json(run_dir / "run_manifest.json")
        manifest["status"] = "failed"
        registry.atomic_write_json(run_dir / "run_manifest.json", manifest)
        status = read_json(run_dir / "run_status.json")
        status["status"] = "failed"
        registry.atomic_write_json(run_dir / "run_status.json", status)
        real_replace = registry.os.replace
        calls = {"count": 0}

        def fail_before_markdown(source, destination):
            calls["count"] += 1
            if calls["count"] == 1:
                raise OSError("synthetic transaction failure")
            return real_replace(source, destination)

        with mock.patch.object(registry.os, "replace", side_effect=fail_before_markdown):
            with self.assertRaises(OSError):
                register_dynamic_run_state(run_dir)
        self.assertEqual(self.experiments.read_bytes(), before)

    def test_v1_schema_migration_preserves_rows_and_unknown_fields(self):
        path = self.root / "legacy.csv"
        path.write_text("run_id,legacy_only\nR1,value\n", encoding="utf-8")
        fields, rows = migrate_unified_schema(path, RUN_FIELDS)
        self.assertIn("legacy_only", fields)
        self.assertEqual(rows[0]["legacy_only"], "value")
        with path.open(encoding="utf-8") as handle:
            migrated = list(csv.DictReader(handle))
        self.assertEqual(migrated[0]["legacy_only"], "value")
        self.assertEqual(migrated[0]["status"], "not_recorded")

    def test_no_lowercase_experiment_markdown_is_created(self):
        run_dir, manifest = self.make_run("smoke")
        self.assertTrue((run_dir / "evidence_manifest.tsv").is_file())
        self.assertEqual(
            manifest["run_evidence_manifest"]["sha256"],
            registry.sha256_file(run_dir / "evidence_manifest.tsv"),
        )
        self.assertFalse((self.root / "experiment.md").exists())
        self.assertFalse((self.root / "Experiment.md").exists())

    def test_repository_schema_headers_match_recorder_fields(self):
        repository_root = Path(__file__).resolve().parents[1]
        with (repository_root / "experiment_records" / "runs.csv").open(
                encoding="utf-8", newline="") as handle:
            self.assertEqual(tuple(next(csv.reader(handle))), tuple(RUN_FIELDS))

    def test_failed_run_console_log_enters_evidence_manifest(self):
        run_dir, _ = self.make_run("smoke")
        (run_dir / "console.log").write_text(
            "stdout\nstderr\n", encoding="utf-8", newline="\n"
        )
        transition_dynamic_run(run_dir, "failed", return_code=7, error="failure")
        with (self.records / "evidence_manifest.tsv").open(
                encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        console_rows = [row for row in rows if row["artifact_type"] == "console_log"]
        self.assertEqual(len(console_rows), 1)
        self.assertEqual(console_rows[0]["status"], "failed")
        self.assertEqual(console_rows[0]["sha256"], registry.sha256_file(run_dir / "console.log"))

    def test_success_sealing_fails_closed_without_console_log(self):
        run_dir, _ = self.make_run("smoke")
        with self.assertRaisesRegex(
                DynamicExperimentEvidenceError, "console.log"):
            seal_dynamic_run_evidence(
                run_dir, {}, {}, {}, [], {}, {},
                run_dir / "missing-summary.json",
                run_dir / "missing-samples.tsv", {}, 1.0,
            )


class GatingEpochStatisticsTest(unittest.TestCase):
    def test_sample_weighted_statistics_are_saved_and_read(self):
        accumulator = GatingEpochAccumulator(1.0)
        accumulator.update([[0.1, 0.2, 0.7]])
        accumulator.update([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        summary = accumulator.summary()
        self.assertEqual(summary["gating_sample_count"], 3)
        self.assertAlmostEqual(summary["p2_mean"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["applied_w2_mean"], 1.0)
        self.assertAlmostEqual(
            sum(summary[key] for key in (
                "dominant_k2_ratio", "dominant_k4_ratio", "dominant_k6_ratio"
            )), 1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = append_gating_epoch_record(directory, 1, 17, 17, summary)
            rows = read_gating_epoch_records(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["global_iteration"], 17)
        self.assertEqual(rows[0]["gating_sample_count"], 3)


class Seed42ReceiptTest(unittest.TestCase):
    def test_complete_seed42_chain_is_required(self):
        receipt = {
            "seed": 42,
            "seed_chain": {
                "source_config_seed": 42, "resolved_config_seed": 42,
                "applied_training_seed": 42,
                "reproducibility_metadata_seed": 42,
            },
            "random_state": {
                "seed": 42, "python_random_seed": 42, "numpy_seed": 42,
                "torch_cpu_seed": 42, "torch_cuda_manual_seed_all_seed": 42,
                "python_random_seeded": True, "numpy_seeded": True,
                "torch_cpu_seeded": True, "cuda_available": True,
                "torch_cuda_manual_seed_all_called": True,
                "torch_cuda_all_seeded": True, "cudnn_deterministic": True,
                "cudnn_benchmark": False, "pythonhashseed": "42",
                "cublas_workspace_config": ":4096:8",
            },
            "data_loader_worker_seeding": {
                "enabled": True, "num_workers": 8,
                "scheme": "torch.initial_seed() modulo 2**32 -> Python random and NumPy",
            },
            "random_identity_sampler": {"base_seed": 42},
            "data_loader_generators": {
                "stream_seeds": {"train": 42, "query": 43, "gallery": 44}
            },
            "configuration": {
                "source_file_sha256": "a" * 64,
                "resolved_file_sha256": "b" * 64,
            },
        }
        environment = {
            "gpu_count": 1, "pythonhashseed": "42",
            "cublas_workspace_config": ":4096:8",
        }
        self.assertTrue(validate_seed42_reproducibility(
            receipt, environment, "a" * 64, "b" * 64
        ))
        receipt["random_state"]["cudnn_benchmark"] = True
        with self.assertRaisesRegex(DynamicExperimentEvidenceError, "benchmark"):
            validate_seed42_reproducibility(
                receipt, environment, "a" * 64, "b" * 64
            )


class CheckpointEvidenceTest(unittest.TestCase):
    def test_checkpoint_mapping_uses_epoch_evidence_not_multiplication(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "log.txt").write_text(
                "EPOCH_EVIDENCE epoch=1 global_iteration=17 epoch_length=17\n",
                encoding="utf-8",
            )
            checkpoint = output / "resnet50_checkpoint_17.pt"
            checkpoint.write_bytes(b"checkpoint")
            validations = [{
                "epoch": 1, "global_iteration": 17, "rank1_percent": 1.0,
                "rank5_percent": 2.0, "rank10_percent": 3.0, "map_percent": 4.0,
            }]
            rows = build_dynamic_checkpoint_manifest(output, validations)
            selected, record = select_dynamic_checkpoint(rows, validations)
            self.assertEqual(selected["epoch"], 1)
            self.assertEqual(selected["global_iteration"], 17)
            self.assertEqual(selected["epoch_length"], 17)
            self.assertEqual(record["epoch"], 1)

    def test_missing_epoch_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "log.txt").write_text("ordinary log\n", encoding="utf-8")
            (output / "resnet50_checkpoint_17.pt").write_bytes(b"checkpoint")
            with self.assertRaisesRegex(DynamicExperimentEvidenceError, "EPOCH_EVIDENCE"):
                build_dynamic_checkpoint_manifest(output, [])


class FormalReadinessTest(unittest.TestCase):
    def test_csv_success_without_manifests_cannot_unlock_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            records = Path(directory)
            path = records / "runs.csv"
            path.write_text(
                "experiment_id,run_kind,status,feature_compatibility_status,"
                "gating_temperature,gating_sample_count\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    DynamicExperimentEvidenceError, "smoke"):
                _require_successful_smoke_before_formal(records)
            path.write_text(
                path.read_text(encoding="utf-8")
                + "C2-L03-MGDG-T1-S42-SMOKE,smoke,success,compatible,1.0,64\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    DynamicExperimentEvidenceError, "complete current candidate"):
                _require_successful_smoke_before_formal(records)


if __name__ == "__main__":
    unittest.main()
