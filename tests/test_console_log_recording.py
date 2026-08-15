# encoding: utf-8
"""Synthetic subprocess tests for the unified live stdout/stderr tee."""

from __future__ import absolute_import

import csv
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tools.run_experiment import (
    TrainingInterrupted,
    _launch_training_subprocess,
)
from utils.experiment_recording import (
    SCHEMA_VERSION,
    EvidenceError,
    atomic_write_json,
    ensure_record_layout,
    finalize_run,
    read_json,
    record_console_log_evidence,
    record_run_failure,
    sha256_file,
)


def read_rows(path, delimiter=","):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def make_run(root, run_id, status="running"):
    root = Path(root)
    records = ensure_record_layout(root / "experiment_records")
    run_dir = records / "runs" / run_id
    run_dir.mkdir()
    experiments = root / "EXPERIMENTS.md"
    experiments.write_text("# Experiments\n\nmanual text\n", encoding="utf-8")
    atomic_write_json(run_dir / "run_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_id": "SYNTHETIC-CONSOLE",
        "experiment_family": "synthetic",
        "run_kind": "smoke",
        "method": "synthetic",
        "method_family": "part_alignment",
        "method_variant": "soft_min",
        "dataset": "market1501",
        "branch": "exp/test",
        "commit_id": "a" * 40,
        "parent_branch": "exp/hard",
        "parent_commit": "b" * 40,
        "seed": 42,
        "alignment_mode": "soft_min",
        "alignment_temperature": 0.1,
        "gating_mode": "not_applicable",
        "gating_temperature": "not_applicable",
        "output_dir": str(root / ("output-" + run_id)),
        "console_log_path": str(run_dir / "console.log"),
        "experiments_path": str(experiments),
        "start_time": "2026-08-15T00:00:00Z",
        "notes": "synthetic",
    })
    atomic_write_json(run_dir / "run_status.json", {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "phase": "synthetic",
        "updated_at_utc": "2026-08-15T00:00:01Z",
    })
    return records, run_dir, experiments


class InterruptingStream(io.StringIO):
    def __init__(self):
        super(InterruptingStream, self).__init__()
        self.interrupted = False

    def write(self, value):
        result = super(InterruptingStream, self).write(value)
        if value and not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt()
        return result


class TimestampStream(io.StringIO):
    def __init__(self):
        super(TimestampStream, self).__init__()
        self.writes = []

    def write(self, value):
        if value:
            self.writes.append((time.monotonic(), value))
        return super(TimestampStream, self).write(value)


class ConsoleLogRecordingTest(unittest.TestCase):
    def test_stdout_stderr_unicode_are_teed_and_exit_zero_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "console.log"
            terminal = io.StringIO()
            command = [
                sys.executable, "-u", "-c",
                "import sys; print('标准输出', flush=True); "
                "print('错误输出', file=sys.stderr, flush=True)",
            ]
            completed = _launch_training_subprocess(
                command, dict(os.environ), path, terminal_stream=terminal
            )
            self.assertEqual(completed.returncode, 0)
            content = path.read_text(encoding="utf-8")
            self.assertIn("标准输出", content)
            self.assertIn("错误输出", content)
            self.assertLess(content.index("标准输出"), content.index("错误输出"))
            self.assertEqual(
                content,
                terminal.getvalue().replace("\r\n", "\n").replace("\r", "\n"),
            )
            self.assertGreater(path.stat().st_size, 0)

    def test_nonzero_exit_and_traceback_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "console.log"
            completed = _launch_training_subprocess(
                [sys.executable, "-u", "-c", "raise RuntimeError('boom')"],
                dict(os.environ), path, terminal_stream=io.StringIO(),
            )
            self.assertNotEqual(completed.returncode, 0)
            content = path.read_text(encoding="utf-8")
            self.assertIn("Traceback", content)
            self.assertIn("RuntimeError: boom", content)

    def test_output_is_forwarded_before_the_child_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "console.log"
            terminal = TimestampStream()
            command = [
                sys.executable, "-u", "-c",
                "import time; print('early', flush=True); time.sleep(0.6); "
                "print('late', flush=True)",
            ]
            started = time.monotonic()
            _launch_training_subprocess(
                command, dict(os.environ), path, terminal_stream=terminal
            )
            ended = time.monotonic()
            early_times = [
                timestamp for timestamp, text in terminal.writes
                if "early" in text
            ]
            self.assertTrue(early_times)
            self.assertLess(early_times[0] - started, 0.5)
            self.assertGreater(ended - early_times[0], 0.4)

    def test_invalid_utf8_is_replaced_in_utf8_console_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "console.log"
            command = [
                sys.executable, "-u", "-c",
                "import os; os.write(1, b'bad-\\xff-byte\\n')",
            ]
            _launch_training_subprocess(
                command, dict(os.environ), path, terminal_stream=io.StringIO()
            )
            self.assertIn("bad-\ufffd-byte", path.read_text(encoding="utf-8"))

    def test_interrupt_terminates_child_and_closes_partial_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "console.log"
            command = [
                sys.executable, "-u", "-c",
                "import time; print('before interrupt', flush=True); time.sleep(30)",
            ]
            with self.assertRaises(TrainingInterrupted) as context:
                _launch_training_subprocess(
                    command, dict(os.environ), path,
                    terminal_stream=InterruptingStream()
                )
            self.assertIsNotNone(context.exception.returncode)
            self.assertIn(
                "before interrupt", path.read_text(encoding="utf-8")
            )
            with path.open("a", encoding="utf-8") as handle:
                handle.write("closed\n")

    def test_console_hash_and_all_terminal_states_enter_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, status in enumerate(
                    ("success", "failed", "incomplete"), start=1):
                records, run_dir, experiments = make_run(
                    directory, "run-{}".format(index), status=status
                )
                (run_dir / "console.log").write_text(
                    "{} console\n".format(status), encoding="utf-8"
                )
                evidence = record_console_log_evidence(run_dir)
                self.assertEqual(
                    evidence["sha256"], sha256_file(run_dir / "console.log")
                )
                if status != "success":
                    record_run_failure(run_dir, RuntimeError(status), status=status)
            rows = read_rows(
                records / "evidence_manifest.tsv", delimiter="\t"
            )
            console_rows = [
                row for row in rows if row["artifact_type"] == "console_log"
            ]
            self.assertEqual(len(console_rows), 3)
            self.assertEqual(len({row["path"] for row in console_rows}), 3)
            registry = read_rows(records / "runs.csv")
            self.assertEqual(
                {row["status"] for row in registry},
                {"success", "failed", "incomplete"},
            )
            markdown = experiments.read_text(encoding="utf-8")
            for status in ("success", "failed", "incomplete"):
                self.assertIn(status, markdown)

    def test_schema_v4_finalizer_requires_nonempty_matching_console_hash(self):
        from tests.test_experiment_recording import make_fixture

        for case in ("missing", "empty", "wrong_hash"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                records, run_dir, _output, experiments = make_fixture(root)
                manifest = read_json(run_dir / "run_manifest.json")
                manifest["schema_version"] = SCHEMA_VERSION
                console = run_dir / "console.log"
                if case == "empty":
                    console.write_text("", encoding="utf-8")
                    manifest["console_log_sha256"] = sha256_file(console)
                elif case == "wrong_hash":
                    console.write_text("real console\n", encoding="utf-8")
                    manifest["console_log_sha256"] = "0" * 64
                atomic_write_json(run_dir / "run_manifest.json", manifest)
                with self.assertRaises(EvidenceError) as context:
                    finalize_run(
                        run_dir, records, root, experiments,
                        run_analyses=False, verify_git=False,
                    )
                self.assertIn("Console log", str(context.exception))


if __name__ == "__main__":
    unittest.main()
