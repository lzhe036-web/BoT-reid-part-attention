import csv
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

import utils.dynamic_experiment_registry as registry
from utils.experiment_schema import (
    AUTO_CHECKPOINTS_START,
    AUTO_RESULTS_START,
    AUTO_RUNS_START,
    EVIDENCE_FIELDS,
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
            "soft_alignment_loss": "0.125", "notes": "real-v4-layout",
        })
        fields, _rows, persisted = self.migrate_fixture(header, row)
        self.assertEqual(persisted[0]["schema_version"], "4")
        self.assertEqual(persisted[0]["commit"], "a" * 40)
        self.assertEqual(persisted[0]["soft_alignment_loss"], "0.125")
        self.assertEqual(persisted[0]["notes"], "real-v4-layout")
        self.assertIn("soft_alignment_loss", fields)

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


if __name__ == "__main__":
    unittest.main()
