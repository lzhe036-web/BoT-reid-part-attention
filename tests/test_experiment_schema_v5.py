import csv
import hashlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

import utils.dynamic_experiment_registry as registry
from tools.generate_experiment_tables import main as generate_experiment_tables
from utils.experiment_schema import (
    AUTO_CHECKPOINTS_START,
    AUTO_RESULTS_START,
    AUTO_RUNS_START,
    EVIDENCE_FIELDS,
    GATING_STAT_FIELDS,
    RUN_FIELDS,
    SCHEMA_VERSION,
)


SOFT_V4_COMMIT = "b1c1cd3eabfca0bb1a08b90b3712523a3e7e719b"
REPO_ROOT = Path(__file__).resolve().parents[1]


class SchemaV5MigrationTest(unittest.TestCase):
    def migrate_fixture(self, header, row, fields=RUN_FIELDS, delimiter=","):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(
                buffer, fieldnames=header, delimiter=delimiter,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(row)
            path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")
            migrated_fields, migrated_rows = registry.migrate_unified_schema(
                path, fields, delimiter=delimiter
            )
            with path.open("r", encoding="utf-8", newline="") as handle:
                persisted = list(csv.DictReader(handle, delimiter=delimiter))
            return migrated_fields, migrated_rows, persisted

    def test_v1_fixture_keeps_version_and_unknown_column(self):
        fields, _rows, persisted = self.migrate_fixture(
            ["run_id", "legacy_only"],
            {"run_id": "v1", "legacy_only": "keep"},
        )
        self.assertIn("legacy_only", fields)
        self.assertEqual(persisted[0]["schema_version"], "1")
        self.assertEqual(persisted[0]["legacy_only"], "keep")
        self.assertEqual(persisted[0]["run_manifest_sha256"], "not_recorded")

    def test_v2_fixture_keeps_version_and_hard_metric(self):
        _fields, _rows, persisted = self.migrate_fixture(
            ["schema_version", "run_id", "hard_alignment_loss", "notes"],
            {"schema_version": "2", "run_id": "v2",
             "hard_alignment_loss": "0.2", "notes": "keep-v2"},
        )
        self.assertEqual(persisted[0]["schema_version"], "2")
        self.assertEqual(persisted[0]["hard_alignment_loss"], "0.2")
        self.assertEqual(persisted[0]["notes"], "keep-v2")

    def test_v3_fixture_keeps_version_and_marks_v5_evidence_missing(self):
        _fields, _rows, persisted = self.migrate_fixture(
            ["schema_version", "run_id", "status", "notes"],
            {"schema_version": "3", "run_id": "v3",
             "status": "success", "notes": "keep-v3"},
        )
        self.assertEqual(persisted[0]["schema_version"], "3")
        self.assertEqual(persisted[0]["candidate_protocol_signature_sha256"],
                         "not_recorded")
        self.assertEqual(persisted[0]["run_status_sha256"], "not_recorded")

    def test_real_soft_v4_header_fixture_migrates_without_relabelling(self):
        source = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show",
             "{}:experiment_records/runs.csv".format(SOFT_V4_COMMIT)],
            stderr=subprocess.PIPE,
        ).decode("utf-8", errors="strict")
        header = next(csv.reader(io.StringIO(source)))
        self.assertIn("commit_id", header)
        row = {field: "" for field in header}
        row.update({
            "schema_version": "4", "run_id": "soft-v4",
            "commit_id": "a" * 40, "console_log_sha256": "b" * 64,
            "GPU": "Synthetic GPU", "runtime": "12.5",
            "Rank-1": "91.0", "Rank-5": "96.0", "Rank-10": "98.0",
            "mAP": "82.0", "config_file": "soft.yml",
            "checkpoint": "checkpoint.pt", "checkpoint_sha256": "c" * 64,
            "soft_alignment_loss": "0.125", "notes": "real-v4-layout",
        })
        fields, _rows, persisted = self.migrate_fixture(header, row)
        self.assertEqual(persisted[0]["schema_version"], "4")
        self.assertEqual(persisted[0]["commit"], "a" * 40)
        self.assertEqual(persisted[0]["gpu"], "Synthetic GPU")
        self.assertEqual(persisted[0]["runtime_seconds"], "12.5")
        self.assertEqual(persisted[0]["rank1_percent"], "91.0")
        self.assertEqual(persisted[0]["rank5_percent"], "96.0")
        self.assertEqual(persisted[0]["rank10_percent"], "98.0")
        self.assertEqual(persisted[0]["map_percent"], "82.0")
        self.assertEqual(persisted[0]["source_config_path"], "soft.yml")
        self.assertEqual(
            persisted[0]["selected_checkpoint_path"], "checkpoint.pt"
        )
        self.assertEqual(
            persisted[0]["selected_checkpoint_sha256"], "c" * 64
        )
        self.assertEqual(persisted[0]["soft_alignment_loss"], "0.125")
        self.assertEqual(persisted[0]["notes"], "real-v4-layout")
        self.assertIn("soft_alignment_loss", fields)

    def test_illegal_empty_authority_fields_are_filtered_stably(self):
        fields = registry._authority_fields(
            ("run_id",), ("", "   ", None, "future_metric"),
            [{"\n": "bad", "future_metric": "1"}],
        )
        self.assertEqual(fields, ("run_id", "future_metric"))

    def test_v5_fixture_stays_v5(self):
        _fields, _rows, persisted = self.migrate_fixture(
            ["schema_version", "run_id", "status"],
            {"schema_version": str(SCHEMA_VERSION), "run_id": "v5",
             "status": "running"},
        )
        self.assertEqual(persisted[0]["schema_version"], "5")

    def test_v4_evidence_fixture_preserves_hash(self):
        source = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show",
             "{}:experiment_records/evidence_manifest.tsv".format(SOFT_V4_COMMIT)],
            stderr=subprocess.PIPE,
        ).decode("utf-8", errors="strict")
        header = next(csv.reader(io.StringIO(source), delimiter="\t"))
        row = {field: "" for field in header}
        row.update({"schema_version": "4", "run_id": "v4",
                    "artifact_type": "console_log", "path": "console.log",
                    "size_bytes": "10", "sha256": "c" * 64})
        _fields, _rows, persisted = self.migrate_fixture(
            header, row, EVIDENCE_FIELDS, delimiter="\t"
        )
        self.assertEqual(persisted[0]["schema_version"], "4")
        self.assertEqual(persisted[0]["sha256"], "c" * 64)


class CommonMarkdownMarkerMigrationTest(unittest.TestCase):
    def test_header_only_legacy_sections_are_removed_without_rows(self):
        content = (
            "Manual stays.\n\n"
            "<!-- AUTO-DYNAMIC-GATING-RUNS:START -->\n"
            "## Run Registry / All Recorded Runs\n\n"
            "| run_id | status |\n|---|---|\n"
            "<!-- AUTO-DYNAMIC-GATING-RUNS:END -->\n"
        )
        migrated = registry._remove_legacy_dynamic_sections(content, set())
        self.assertIn("Manual stays.", migrated)
        self.assertNotIn("AUTO-DYNAMIC", migrated)

    def test_unmigrated_real_legacy_row_fails_closed(self):
        content = (
            "Manual stays.\n"
            "<!-- AUTO-DYNAMIC-GATING-RUNS:START -->\n"
            "| run_id | status |\n|---|---|\n| real-run | success |\n"
            "<!-- AUTO-DYNAMIC-GATING-RUNS:END -->\n"
        )
        with self.assertRaises(registry.DynamicExperimentEvidenceError):
            registry._remove_legacy_dynamic_sections(content, set())

    def test_migrated_legacy_row_can_be_replaced_by_common_markers(self):
        content = (
            "Manual stays.\n"
            "<!-- AUTO-DYNAMIC-GATING-RUNS:START -->\n"
            "| run_id | status |\n|---|---|\n| real-run | success |\n"
            "<!-- AUTO-DYNAMIC-GATING-RUNS:END -->\n"
        )
        migrated = registry._remove_legacy_dynamic_sections(
            content, {"real-run"}
        )
        self.assertNotIn("AUTO-DYNAMIC", migrated)

    def test_repository_has_only_one_common_generated_marker_set(self):
        content = (REPO_ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")
        self.assertEqual(content.count(AUTO_RUNS_START), 1)
        self.assertEqual(content.count(AUTO_RESULTS_START), 1)
        self.assertEqual(content.count(AUTO_CHECKPOINTS_START), 1)
        self.assertNotIn("AUTO-DYNAMIC-GATING", content)


class UnifiedHistoricalMarkdownEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.records = self.root / "experiment_records"
        (self.records / "tables").mkdir(parents=True)
        self.experiments = self.root / "EXPERIMENTS.md"
        self.experiments.write_text(
            "# Experiments\n\nHuman-maintained explanation stays.\n",
            encoding="utf-8", newline="\n",
        )
        source = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show",
             "{}:experiment_records/runs.csv".format(SOFT_V4_COMMIT)],
            stderr=subprocess.PIPE,
        ).decode("utf-8", errors="strict")
        self.v4_header = next(csv.reader(io.StringIO(source)))

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_csv(path, fields, rows, delimiter=","):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, delimiter=delimiter,
                lineterminator="\n", extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def markdown_rows(section):
        lines = [
            line for line in section.splitlines()
            if line.startswith("|")
        ]
        headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
        rows = []
        for line in lines[2:]:
            values = [cell.strip() for cell in line.strip("|").split("|")]
            rows.append(dict(zip(headers, values)))
        return headers, {row["run_id"]: row for row in rows}

    def test_real_v4_alignment_and_future_fields_reach_both_markdown_sections(self):
        historical_fields = self.v4_header + ["future_alignment_metric"]
        soft_row = {field: "" for field in historical_fields}
        soft_row.update({
            "schema_version": "4",
            "run_id": "soft-v4-formal-run",
            "experiment_id": "SOFT-V4-FORMAL",
            "experiment_family": "c2l03_soft_min_alignment",
            "run_kind": "formal", "status": "success",
            "method": "Soft-Min Alignment",
            "method_family": "part_alignment",
            "method_variant": "soft_min",
            "dataset": "Market1501", "branch": "soft-branch",
            "commit_id": "a" * 40, "parent_branch": "hard-branch",
            "parent_commit": "b" * 40, "config_file": "soft.yml",
            "seed": "42", "lambda": "0.1",
            "cross_camera_positive_lambda": "0.3",
            "pcc_lambda": "0.1", "pcc_enabled": "true",
            "pcc_parts": "6", "pcc_mode": "soft_min",
            "alignment_strategy": "monotonic_right_down",
            "alignment_mode": "soft_min", "alignment_temperature": "0.1",
            "multigranular_feature_signature": "canonical-json",
            "multigranular_feature_signature_sha256": "c" * 64,
            "GPU": "Fixture GPU", "runtime": "60.5",
            "Rank-1": "91.1", "Rank-5": "96.2", "Rank-10": "98.3",
            "mAP": "82.4", "best_epoch": "80", "selected_epoch": "80",
            "valid_pcc_pair_count": "9",
            "mean_fixed_index_part_distance": "0.250",
            "hard_alignment_loss": "0.300",
            "valid_alignment_pair_count": "11",
            "mean_hard_path_cost": "3.300",
            "mean_path_absolute_offset": "0.400",
            "soft_alignment_loss": "0.125",
            "mean_soft_path_cost": "1.375",
            "future_alignment_metric": "future-kept",
            "notes": "real-v4-header-fixture",
        })
        runs_path = self.records / "runs.csv"
        formal_path = self.records / "tables" / "main_results.csv"
        evidence_path = self.records / "evidence_manifest.tsv"
        self.write_csv(runs_path, historical_fields, [soft_row])
        self.write_csv(formal_path, historical_fields, [soft_row])
        self.write_csv(evidence_path, EVIDENCE_FIELDS, [], delimiter="\t")

        self.assertEqual(generate_experiment_tables([
            "--records-root", str(self.records),
            "--experiments", str(self.experiments),
        ]), 0)

        with runs_path.open(encoding="utf-8", newline="") as handle:
            migrated_fields = list(next(csv.reader(handle)))
            migrated_rows = list(csv.DictReader(handle, fieldnames=migrated_fields))
        self.assertEqual(len(migrated_rows), 1)
        self.assertEqual(migrated_rows[0]["schema_version"], "4")

        dynamic_row = {field: "not_recorded" for field in migrated_fields}
        dynamic_row.update({
            "schema_version": "5", "run_id": "dynamic-v5-formal-run",
            "experiment_id": "DYNAMIC-V5-FORMAL", "run_kind": "formal",
            "status": "success", "method_family": "multi_granularity_feature",
            "method_variant": "per_sample_dynamic_gating",
            "gating_mode": "per_sample_dynamic_gating",
            "gating_input": "global", "gating_temperature": "1.0",
            "gating_normalization": "scaled_softmax", "scale_order": "2,4,6",
            "gating_sample_count": "256", "p2_mean": "0.2",
            "p4_mean": "0.3", "p6_mean": "0.5",
        })
        all_rows = migrated_rows + [dynamic_row]
        self.write_csv(runs_path, migrated_fields, all_rows)
        self.write_csv(formal_path, migrated_fields, all_rows)

        self.assertEqual(generate_experiment_tables([
            "--records-root", str(self.records),
            "--experiments", str(self.experiments),
        ]), 0)
        first_payloads = {
            path: path.read_bytes() for path in (
                runs_path, formal_path, evidence_path, self.experiments
            )
        }
        first_hashes = {
            path: hashlib.sha256(payload).hexdigest()
            for path, payload in first_payloads.items()
        }
        self.assertEqual(generate_experiment_tables([
            "--records-root", str(self.records),
            "--experiments", str(self.experiments),
        ]), 0)
        for path, payload in first_payloads.items():
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), first_hashes[path]
            )

        markdown = self.experiments.read_text(encoding="utf-8")
        run_section = markdown.split(
            "## Run Registry / All Recorded Runs", 1
        )[1].split("<!-- AUTO-EXPERIMENT-RUNS:END -->", 1)[0]
        formal_section = markdown.split(
            "## Formal Results", 1
        )[1].split("<!-- AUTO-EXPERIMENT-RESULTS:END -->", 1)[0]
        run_headers, run_rows = self.markdown_rows(run_section)
        formal_headers, formal_rows = self.markdown_rows(formal_section)
        required_alignment_values = {
            "valid_pcc_pair_count": "9",
            "mean_fixed_index_part_distance": "0.250",
            "hard_alignment_loss": "0.300",
            "valid_alignment_pair_count": "11",
            "mean_hard_path_cost": "3.300",
            "mean_path_absolute_offset": "0.400",
            "soft_alignment_loss": "0.125",
            "mean_soft_path_cost": "1.375",
            "dataset": "Market1501", "pcc_mode": "soft_min",
            "alignment_strategy": "monotonic_right_down",
            "future_alignment_metric": "future-kept",
        }
        for field, value in required_alignment_values.items():
            self.assertIn(field, run_headers)
            self.assertIn(field, formal_headers)
            self.assertEqual(run_rows["soft-v4-formal-run"][field], value)
            self.assertEqual(formal_rows["soft-v4-formal-run"][field], value)
        for field in GATING_STAT_FIELDS:
            self.assertIn(field, run_headers)
            self.assertIn(field, formal_headers)
        expected_dynamic_values = {
            "method_variant": "per_sample_dynamic_gating",
            "gating_mode": "per_sample_dynamic_gating",
            "gating_input": "global",
            "gating_normalization": "scaled_softmax",
            "scale_order": "2,4,6", "gating_sample_count": "256",
            "p2_mean": "0.2", "p4_mean": "0.3", "p6_mean": "0.5",
        }
        for field, value in expected_dynamic_values.items():
            self.assertEqual(run_rows["dynamic-v5-formal-run"][field], value)
            self.assertEqual(formal_rows["dynamic-v5-formal-run"][field], value)
        self.assertIn("Human-maintained explanation stays.", markdown)
        checkpoint_section = markdown.split(
            "## Checkpoint Evidence", 1
        )[1].split("<!-- AUTO-CHECKPOINT-EVIDENCE:END -->", 1)[0]
        self.assertNotIn(".pt", checkpoint_section)
        self.assertNotIn("AUTO-DYNAMIC-GATING", markdown)
        with runs_path.open(encoding="utf-8", newline="") as handle:
            persisted = list(csv.DictReader(handle))
        self.assertEqual(len(persisted), 2)
        self.assertEqual(persisted[0]["selected_checkpoint_path"], "not_recorded")
        self.assertEqual(persisted[0]["selected_checkpoint_sha256"], "not_recorded")


if __name__ == "__main__":
    unittest.main()
