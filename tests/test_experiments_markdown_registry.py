# encoding: utf-8
"""Tests for atomic, idempotent EXPERIMENTS.md recorder sections."""

from __future__ import absolute_import

import csv
import tempfile
import unittest
from pathlib import Path

from utils.experiment_recording import (
    AUTO_CHECKPOINTS_END,
    AUTO_CHECKPOINTS_START,
    AUTO_RESULTS_END,
    AUTO_RESULTS_START,
    AUTO_RUNS_END,
    AUTO_RUNS_START,
    MAIN_FIELDS,
    SCHEMA_VERSION,
    EvidenceError,
    _register_run_state,
    atomic_write_json,
    ensure_record_layout,
    initialize_run,
    sha256_file,
    update_experiments_markdown,
    upsert_csv,
    write_tsv,
)


def create_run(records, run_id, run_kind, status, notes="notes",
               checkpoints=0):
    run_dir = Path(records) / "runs" / run_id
    run_dir.mkdir()
    output = Path(records).parent / ("output-" + run_id)
    output.mkdir()
    source = run_dir / "config_source.yml"
    resolved = run_dir / "config_resolved.yml"
    source.write_text("SEED: 42\n", encoding="utf-8")
    resolved.write_text("SEED: 42\n", encoding="utf-8")
    console = run_dir / "console.log"
    console.write_text("console {}\n".format(run_id), encoding="utf-8")
    training = output / "log.txt"
    training.write_text("training {}\n".format(run_id), encoding="utf-8")
    feature = run_dir / "feature_compatibility.json"
    atomic_write_json(feature, {
        "feature_reference_commit": "b" * 40,
        "feature_reference_signature_sha256": "c" * 64,
        "current_feature_signature_sha256": "c" * 64,
        "feature_compatibility_status": "compatible",
        "mismatched_components": [],
    })
    checkpoint_rows = []
    for epoch in range(1, checkpoints + 1):
        checkpoint = output / "checkpoint_{}.pt".format(epoch)
        checkpoint.write_bytes(("checkpoint-{}".format(epoch)).encode("ascii"))
        checkpoint_rows.append({
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "filename": checkpoint.name,
            "path": str(checkpoint.resolve()),
            "size_bytes": checkpoint.stat().st_size,
            "epoch": epoch,
            "global_iteration": epoch * 17,
            "global_iteration_source": "ignite_engine_epoch_evidence",
            "sha256": sha256_file(checkpoint),
            "selected": epoch == checkpoints,
        })
    if checkpoint_rows:
        write_tsv(
            run_dir / "checkpoint_manifest.tsv",
            tuple(checkpoint_rows[0].keys()), checkpoint_rows,
        )
    artifact = run_dir / "artifact_hashes.tsv"
    artifact.write_text(
        "artifact_type\tpath\tsize_bytes\tsha256\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_id": "EXP-{}".format(run_id),
        "experiment_family": "c2l03_soft_min_alignment",
        "run_kind": run_kind,
        "method": "C2-L03 + Soft-Min Alignment",
        "method_family": "part_alignment",
        "method_variant": "soft_min",
        "dataset": "market1501",
        "branch": "exp/c2l03-soft-min-alignment",
        "commit_id": "a" * 40,
        "parent_branch": "exp/c2l03-hard-shortest-path-alignment",
        "parent_commit": "b" * 40,
        "config_file": "configs/soft.yml",
        "config_source_sha256": sha256_file(source),
        "config_resolved_sha256": sha256_file(resolved),
        "seed": 42,
        "lambda": 0.3,
        "cross_camera_positive_lambda": 0.3,
        "pcc_enabled": True,
        "pcc_parts": 6,
        "pcc_lambda": 0.1,
        "pcc_mode": "soft_min",
        "alignment_strategy": "soft_min",
        "alignment_mode": "soft_min",
        "alignment_temperature": 0.1,
        "gating_mode": "not_applicable",
        "gating_temperature": "not_applicable",
        "feature_reference_commit": "b" * 40,
        "feature_reference_signature_sha256": "c" * 64,
        "current_feature_signature_sha256": "c" * 64,
        "feature_compatibility_status": "compatible",
        "feature_compatibility_evidence_path": str(feature.resolve()),
        "feature_compatibility_evidence_sha256": sha256_file(feature),
        "output_dir": str(output.resolve()),
        "training_log_path": str(training.resolve()),
        "training_log_sha256": sha256_file(training),
        "console_log_path": str(console.resolve()),
        "console_log_sha256": sha256_file(console),
        "artifact_manifest_path": str(artifact.resolve()),
        "artifact_manifest_sha256": sha256_file(artifact),
        "start_time": "2026-08-15T00:00:0{}Z".format(len(run_id) % 10),
        "notes": notes,
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    status_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "phase": "complete" if status == "success" else status,
        "training_runtime_seconds": 12.5,
        "training_end_time": "2026-08-15T00:01:00Z",
        "updated_at_utc": "2026-08-15T00:01:00Z",
    }
    atomic_write_json(run_dir / "run_status.json", status_payload)
    _register_run_state(run_dir, status_payload)
    return run_dir, manifest, checkpoint_rows


class ExperimentsMarkdownRegistryTest(unittest.TestCase):
    def test_initialize_run_immediately_registers_running_status(self):
        from config import cfg

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "experiment_records"
            experiments = root / "EXPERIMENTS.md"
            experiments.write_text("# Experiments\n", encoding="utf-8")
            source_config = Path(__file__).resolve().parents[1] / "configs" / (
                "softmax_triplet_c2l03_soft_min_alignment_autodl.yml"
            )
            resolved = cfg.clone()
            resolved.merge_from_file(str(source_config))
            resolved.defrost()
            resolved.OUTPUT_DIR = str(root / "output")
            resolved.freeze()
            signature = "c" * 64
            run_dir, _ = initialize_run(
                records, "INIT-SOFT", "c2l03_soft_min_alignment",
                "initialized-run", str(source_config), resolved,
                str(root / "output"),
                {"branch": "exp/c2l03-soft-min-alignment",
                 "commit": "a" * 40},
                "initial", ["runner"],
                "exp/c2l03-soft-min-alignment", run_kind="smoke",
                parent_branch="exp/c2l03-hard-shortest-path-alignment",
                parent_commit="b" * 40,
                feature_compatibility={
                    "feature_reference_commit": "b" * 40,
                    "feature_current_commit": "a" * 40,
                    "feature_reference_signature_sha256": signature,
                    "current_feature_signature_sha256": signature,
                    "feature_compatibility_status": "compatible",
                    "mismatched_components": [], "components": {},
                },
                experiments_path=experiments,
            )
            rows = []
            with (records / "runs.csv").open(
                    "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["status"], "running")
            markdown = experiments.read_text(encoding="utf-8")
            self.assertIn("initialized-run", markdown)
            self.assertIn("running", markdown)
            self.assertTrue((run_dir / "feature_compatibility.json").is_file())

    def test_all_runs_formal_isolation_checkpoints_and_full_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = ensure_record_layout(root / "experiment_records")
            experiments = root / "EXPERIMENTS.md"
            experiments.write_text(
                "# Experiments\n\nHuman-maintained | explanation.\n",
                encoding="utf-8",
            )
            _, formal, checkpoints = create_run(
                records, "formal-success", "formal", "success",
                notes="pipe|line\nnext", checkpoints=3,
            )
            create_run(records, "smoke-success", "smoke", "success")
            create_run(records, "formal-failed", "formal", "failed")
            create_run(records, "formal-incomplete", "formal", "incomplete")
            create_run(records, "formal-interrupted", "formal", "interrupted")
            upsert_csv(records / "tables" / "main_results.csv", MAIN_FIELDS, {
                "schema_version": SCHEMA_VERSION,
                "run_id": "formal-success",
                "experiment_id": formal["experiment_id"],
                "run_kind": "formal",
                "status": "success",
                "seed": 42,
                "commit": formal["commit_id"],
                "checkpoint": checkpoints[-1]["path"],
                "checkpoint_sha256": checkpoints[-1]["sha256"],
            })
            update_experiments_markdown(experiments, records)
            first = experiments.read_text(encoding="utf-8")
            update_experiments_markdown(experiments, records)
            second = experiments.read_text(encoding="utf-8")
            self.assertEqual(first, second)
            self.assertIn("Human-maintained | explanation.", first)
            for marker in (
                    AUTO_RUNS_START, AUTO_RUNS_END, AUTO_RESULTS_START,
                    AUTO_RESULTS_END, AUTO_CHECKPOINTS_START,
                    AUTO_CHECKPOINTS_END):
                self.assertIn(marker, first)
            all_runs = first.split(AUTO_RUNS_START, 1)[1].split(
                AUTO_RUNS_END, 1
            )[0]
            for run_id in (
                    "formal-success", "smoke-success", "formal-failed",
                    "formal-incomplete", "formal-interrupted"):
                self.assertIn(run_id, all_runs)
            self.assertEqual(
                sum(
                    1 for line in all_runs.splitlines()
                    if "| formal-success |" in line
                ),
                1,
            )
            self.assertIn("console_log_sha256", all_runs)
            self.assertIn("feature_reference_signature_sha256", all_runs)
            self.assertIn("artifact_manifest_sha256", all_runs)
            self.assertIn("pipe\\|line<br>next", all_runs)
            self.assertIn("not_recorded", all_runs)
            formal_section = first.split(AUTO_RESULTS_START, 1)[1].split(
                AUTO_RESULTS_END, 1
            )[0]
            self.assertIn("formal-success", formal_section)
            self.assertNotIn("smoke-success", formal_section)
            self.assertNotIn("formal-failed", formal_section)
            checkpoint_section = first.split(
                AUTO_CHECKPOINTS_START, 1
            )[1].split(AUTO_CHECKPOINTS_END, 1)[0]
            checkpoint_lines = [
                line for line in checkpoint_section.splitlines()
                if ".pt" in line and line.startswith("|")
            ]
            self.assertEqual(len(checkpoint_lines), 3)
            self.assertIn("ignite_epoch", checkpoint_section)
            self.assertIn("global_iteration", checkpoint_section)
            self.assertIn(checkpoints[-1]["sha256"], checkpoint_section)
            self.assertFalse((root / "experiment.md").exists())
            self.assertFalse((root / "Experiment.md").exists())

    def test_generation_failure_preserves_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = ensure_record_layout(root / "experiment_records")
            experiments = root / "EXPERIMENTS.md"
            original = "# Experiments\n\nmanual\n\n{}\nbroken\n".format(
                AUTO_RUNS_START
            )
            experiments.write_text(original, encoding="utf-8")
            with self.assertRaises(EvidenceError):
                update_experiments_markdown(experiments, records)
            self.assertEqual(
                experiments.read_text(encoding="utf-8"), original
            )


if __name__ == "__main__":
    unittest.main()
