# encoding: utf-8
"""Regression tests for Ignite epoch/checkpoint evidence mapping."""

from __future__ import absolute_import

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ignite.engine import Engine

from engine.trainer import _attach_epoch_evidence_logging
from utils.experiment_recording import (
    EvidenceError,
    build_checkpoint_manifest,
    parse_training_log,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "test_c2l03_seed42_1epoch.sh"


def _training_log(epoch_rows, denominator=183, include_evidence=True):
    lines = [
        "2026-08-07 10:00:00 reid_baseline.train INFO: Start training",
    ]
    for epoch, global_iteration, epoch_length in epoch_rows:
        lines.extend([
            (
                "2026-08-07 10:10:00 reid_baseline.train INFO: "
                "Epoch[{}] Iteration[20/{}] loss_total: 1.0, loss_id: 0.4, "
                "loss_triplet: 0.5, loss_camera_triplet: 0.0, "
                "loss_cross_camera_positive: 0.1, "
                "cross_camera_positive_count: 60.0, Acc: 0.8, "
                "Base Lr: 3.50e-04"
            ).format(epoch, denominator),
        ])
        if include_evidence:
            lines.append(
                "2026-08-07 10:10:01 reid_baseline.train INFO: "
                "EPOCH_EVIDENCE epoch={} global_iteration={} "
                "epoch_length={}".format(
                    epoch, global_iteration, epoch_length
                )
            )
        lines.extend([
            "2026-08-07 10:10:02 reid_baseline.train INFO: "
            "Validation Results - Epoch: {}".format(epoch),
            "2026-08-07 10:10:03 reid_baseline.train INFO: mAP: 2.6%",
            "2026-08-07 10:10:04 reid_baseline.train INFO: "
            "CMC curve, Rank-1  :6.8%",
            "2026-08-07 10:10:05 reid_baseline.train INFO: "
            "CMC curve, Rank-5  :15.3%",
            "2026-08-07 10:10:06 reid_baseline.train INFO: "
            "CMC curve, Rank-10 :21.1%",
        ])
    return "\n".join(lines) + "\n"


def _parse(directory, epoch_rows, denominator=183, include_evidence=True):
    log_path = Path(directory) / "log.txt"
    log_path.write_text(
        _training_log(
            epoch_rows,
            denominator=denominator,
            include_evidence=include_evidence,
        ),
        encoding="utf-8",
    )
    return parse_training_log(log_path)


def _checkpoint(directory, iteration):
    path = Path(directory) / "resnet50_checkpoint_{}.pt".format(iteration)
    path.write_bytes(b"synthetic checkpoint")
    return path


class EpochCheckpointEvidenceTest(unittest.TestCase):
    def test_real_183_log_denominator_binds_checkpoint_186_to_epoch_one(self):
        with tempfile.TemporaryDirectory() as directory:
            log_info = _parse(directory, [(1, 186, 186)], denominator=183)
            _checkpoint(directory, 186)
            rows, selected = build_checkpoint_manifest(
                directory,
                log_info["validations"],
                selected_epoch=1,
                destination=Path(directory) / "checkpoint_manifest.tsv",
            )
            self.assertEqual(rows[0]["epoch"], 1)
            self.assertEqual(rows[0]["global_iteration"], 186)
            self.assertEqual(rows[0]["schema_version"], 3)
            self.assertEqual(Path(rows[0]["path"]).name, selected["filename"])
            self.assertEqual(
                rows[0]["global_iteration_source"],
                "ignite_engine_epoch_evidence",
            )
            self.assertEqual(selected["filename"], "resnet50_checkpoint_186.pt")

    def test_real_183_log_denominator_rejects_checkpoint_185(self):
        with tempfile.TemporaryDirectory() as directory:
            log_info = _parse(directory, [(1, 186, 186)], denominator=183)
            _checkpoint(directory, 185)
            with self.assertRaisesRegex(EvidenceError, "does not exactly match"):
                build_checkpoint_manifest(
                    directory,
                    log_info["validations"],
                    selected_epoch=1,
                    destination=Path(directory) / "checkpoint_manifest.tsv",
                )

    def test_120_epoch_checkpoint_iterations_bind_exact_epochs(self):
        epoch_rows = [
            (40, 7440, 186),
            (80, 14880, 186),
            (120, 22320, 186),
        ]
        with tempfile.TemporaryDirectory() as directory:
            log_info = _parse(directory, epoch_rows, denominator=183)
            for _epoch, global_iteration, _epoch_length in epoch_rows:
                _checkpoint(directory, global_iteration)
            rows, selected = build_checkpoint_manifest(
                directory,
                log_info["validations"],
                selected_epoch=120,
                destination=Path(directory) / "checkpoint_manifest.tsv",
            )
            self.assertEqual(
                {
                    row["global_iteration"]: row["epoch"] for row in rows
                },
                {7440: 40, 14880: 80, 22320: 120},
            )
            self.assertEqual(selected["epoch"], 120)

    def test_authoritative_epoch_length_overrides_legacy_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            log_info = _parse(directory, [(1, 186, 186)], denominator=183)
            validation = log_info["validations"][0]
            self.assertEqual(validation["epoch_length"], 186)
            self.assertEqual(validation["iterations_per_epoch"], 186)
            self.assertEqual(validation["global_iteration"], 186)
            self.assertEqual(
                validation["global_iteration_source"],
                "ignite_engine_epoch_evidence",
            )

    def test_log_without_epoch_evidence_uses_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            log_info = _parse(
                directory,
                [(1, 0, 0), (2, 0, 0)],
                denominator=100,
                include_evidence=False,
            )
            self.assertEqual(
                log_info["global_iteration_source"],
                "legacy_log_denominator_inference",
            )
            self.assertEqual(
                [row["global_iteration"] for row in log_info["validations"]],
                [100, 200],
            )
            _checkpoint(directory, 200)
            rows, _selected = build_checkpoint_manifest(
                directory,
                log_info["validations"],
                selected_epoch=2,
                destination=Path(directory) / "checkpoint_manifest.tsv",
            )
            self.assertEqual(rows[0]["epoch"], 2)

    def test_duplicate_global_iteration_epoch_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(EvidenceError, "maps to multiple epochs"):
                _parse(
                    directory,
                    [(1, 186, 186), (2, 186, 186)],
                    denominator=183,
                )

    def test_checkpoint_without_exact_epoch_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            log_info = _parse(
                directory,
                [(40, 7440, 186), (80, 14880, 186)],
                denominator=183,
            )
            _checkpoint(directory, 22320)
            with self.assertRaisesRegex(EvidenceError, "does not exactly match"):
                build_checkpoint_manifest(
                    directory,
                    log_info["validations"],
                    selected_epoch=80,
                    destination=Path(directory) / "checkpoint_manifest.tsv",
                )

    def test_engine_handler_logs_state_and_resets_epoch_counter(self):
        logger = mock.Mock()
        trainer = Engine(lambda _engine, batch: batch)
        epoch_log_state = _attach_epoch_evidence_logging(trainer, logger)
        trainer.run([1, 2, 3], max_epochs=2)
        messages = [call.args[0] for call in logger.info.call_args_list]
        self.assertEqual(messages, [
            "EPOCH_EVIDENCE epoch=1 global_iteration=3 epoch_length=3",
            "EPOCH_EVIDENCE epoch=2 global_iteration=6 epoch_length=3",
        ])
        self.assertEqual(epoch_log_state["iteration"], 3)

    def test_smoke_script_supports_v2_output_without_changing_default(self):
        script = SMOKE_SCRIPT.read_text(encoding="utf-8")
        default_output = (
            "/root/autodl-tmp/experiments/BoT/"
            "c2l03_seed42_market1501_smoke_1epoch"
        )
        self.assertIn(
            'SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-' + default_output + '}"',
            script,
        )
        self.assertIn('OUTPUT_DIR "${SMOKE_OUTPUT_DIR}"', script)


if __name__ == "__main__":
    unittest.main()
