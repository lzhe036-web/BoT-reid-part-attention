import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.dynamic_experiment_registry import (
    DynamicExperimentEvidenceError,
    _git,
    _porcelain_relative_path,
    validate_dynamic_runtime_worktree,
)


class GitPorcelainParsingTest(unittest.TestCase):
    def test_git_preserves_tracked_modified_leading_space(self):
        with mock.patch.object(
                subprocess, "check_output",
                return_value=b" M EXPERIMENTS.md\n"):
            output = _git(".", [
                "status", "--porcelain=v1", "--untracked-files=all"
            ])
        self.assertEqual(output, " M EXPERIMENTS.md")
        self.assertEqual(
            _porcelain_relative_path(output), "EXPERIMENTS.md"
        )

    def test_staged_modified_path_is_parsed(self):
        self.assertEqual(
            _porcelain_relative_path("M  EXPERIMENTS.md"),
            "EXPERIMENTS.md",
        )

    def test_untracked_path_is_parsed(self):
        self.assertEqual(
            _porcelain_relative_path("?? some_file.txt"), "some_file.txt"
        )

    def test_rename_uses_destination_path(self):
        self.assertEqual(
            _porcelain_relative_path("R  old_name -> new_name"), "new_name"
        )

    def test_only_recorder_evidence_changes_are_allowed(self):
        porcelain = (
            " M EXPERIMENTS.md\n"
            " M experiment_records/runs.csv\n"
            " M experiment_records/evidence_manifest.tsv"
        )
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            run_dir = repo / "experiment_records" / "runs" / "RUN-1"
            output_dir = repo / "output"
            with mock.patch(
                    "utils.dynamic_experiment_registry._git",
                    return_value=porcelain):
                self.assertTrue(validate_dynamic_runtime_worktree(
                    repo, run_dir, output_dir
                ))

    def test_source_modification_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            with mock.patch(
                    "utils.dynamic_experiment_registry._git",
                    return_value=" M modeling/baseline.py"):
                with self.assertRaisesRegex(
                        DynamicExperimentEvidenceError,
                        "modeling/baseline.py"):
                    validate_dynamic_runtime_worktree(
                        repo,
                        repo / "experiment_records" / "runs" / "RUN-1",
                        repo / "output",
                    )


if __name__ == "__main__":
    unittest.main()
