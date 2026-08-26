import csv
import datetime as dt
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from yacs.config import CfgNode

from tools.recover_g2_global_local_experiment import (
    EXPECTED_BRANCH,
    EXPERIMENT_ID,
    G2RecoveryError,
    recover,
)
from utils.config_serialization import serialize_cfg_node_yaml
from utils.experiment_schema import GATING_STAT_FIELDS, SCHEMA_VERSION


COMMIT = "f" * 40


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class G2RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "formal_g2"
        self.output.mkdir()
        self.records = self.root / "experiment_records"
        self.experiments = self.root / "EXPERIMENTS.md"
        self.experiments.write_text("# Experiments\n", encoding="utf-8")
        self.console = self.root / "formal_g2.console.log"
        self.console.write_text("machine-captured console\n", encoding="utf-8")
        self.config = self.root / "g2.yml"
        self._build_fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def _statistics(self, epoch):
        values = {
            "gating_temperature": 1.0,
            "gating_sample_count": 11904,
            "mean_gate_entropy": 1.0 + epoch / 10000.0,
        }
        for label, mean in (("2", 0.4), ("4", 0.3), ("6", 0.3)):
            values["p{}_mean".format(label)] = mean
            values["p{}_std".format(label)] = 0.01
            values["p{}_min".format(label)] = mean - 0.02
            values["p{}_max".format(label)] = mean + 0.02
            values["applied_w{}_mean".format(label)] = mean * 3.0
            values["applied_w{}_std".format(label)] = 0.03
            values["dominant_k{}_ratio".format(label)] = (
                0.5 if label == "2" else 0.25
            )
        self.assertEqual(set(GATING_STAT_FIELDS), set(values))
        return values

    def _build_fixture(self):
        config = {
            "SEED": 42,
            "MODEL": {
                "MULTI_GRANULARITY_DYNAMIC_GATING": True,
                "MULTI_GRANULARITY_GATING_INPUT": "concat_global_local",
                "MULTI_GRANULARITY_GATING_TAU": 1.0,
                "MULTI_GRANULARITY_GATING_NORMALIZATION": "scaled_softmax",
                "MULTI_GRANULARITY_PART_SCALES": [2, 4, 6],
                "CROSS_CAMERA_POSITIVE_MODE": "mean",
                "CROSS_CAMERA_POSITIVE_LAMBDA": 0.3,
            },
            "DATASETS": {"NAMES": "market1501"},
            "SOLVER": {
                "MAX_EPOCHS": 120,
                "CHECKPOINT_PERIOD": 40,
                "EVAL_PERIOD": 40,
                "MARGIN": 0.3,
            },
            "OUTPUT_DIR": str(self.output.resolve()),
        }
        config_text = yaml.safe_dump(config, sort_keys=False)
        self.config.write_text(config_text, encoding="utf-8")
        resolved = self.output / "config_resolved.yml"
        resolved.write_text(
            serialize_cfg_node_yaml(CfgNode(config)), encoding="utf-8"
        )

        (self.output / "log.txt").write_text("formal training log\n", encoding="utf-8")
        validations = []
        metrics = {
            40: (93.0, 84.0),
            80: (95.2, 88.1),
            120: (95.2, 87.9),
        }
        for epoch in (40, 80, 120):
            rank1, mean_ap = metrics[epoch]
            validations.append({
                "epoch": epoch,
                "global_iteration": epoch * 186,
                "timestamp_utc": "2026-08-27T00:00:00Z",
                "rank1_percent": rank1,
                "rank5_percent": 98.0,
                "rank10_percent": 99.0,
                "map_percent": mean_ap,
                "re_ranking": "no",
                "neck_feat": "after",
                "feat_norm": "yes",
            })
        validation_path = self.output / "validation_history.jsonl"
        validation_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in validations),
            encoding="utf-8",
        )

        gate_records = []
        for epoch in range(1, 121):
            row = {
                "epoch": epoch,
                "global_iteration": epoch * 186,
                "epoch_length": 186,
            }
            row.update(self._statistics(epoch))
            gate_records.append(row)
        gate_path = self.output / "dynamic_gating_epoch_stats.jsonl"
        gate_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in gate_records),
            encoding="utf-8",
        )

        checkpoint_rows = []
        for epoch in (40, 80, 120):
            iteration = epoch * 186
            checkpoint = self.output / "resnet50_checkpoint_{}.pt".format(iteration)
            checkpoint.write_bytes("checkpoint-{}\n".format(epoch).encode("ascii"))
            checkpoint_rows.append({
                "epoch": epoch,
                "global_iteration": iteration,
                "epoch_length": 186,
                "filename": checkpoint.name,
                "path": str(checkpoint.resolve()),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": _sha(checkpoint),
                "selected": "true" if epoch == 80 else "false",
                "artifact_type": "model_checkpoint",
                "relative_path": checkpoint.name,
                "file_size": checkpoint.stat().st_size,
            })
        checkpoint_manifest = self.output / "checkpoint_manifest.tsv"
        with checkpoint_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(checkpoint_rows[0]),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(checkpoint_rows)
        selected_checkpoint = self.output / checkpoint_rows[1]["relative_path"]
        selected_sha = _sha(selected_checkpoint)

        analysis_dir = self.output / "g2_gating_analysis"
        analysis_dir.mkdir()
        samples = analysis_dir / "gating_samples.tsv"
        samples.write_text(
            "stable_sample_key\tp2\tp4\tp6\nq:1\t0.4\t0.3\t0.3\n",
            encoding="utf-8",
        )
        summary = analysis_dir / "dynamic_gating_summary.json"
        selection_rule = "sha256(stable_sample_key) ascending; first 256 query+gallery samples"
        _write_json(summary, {
            "schema_version": SCHEMA_VERSION,
            "source_checkpoint_path": str(selected_checkpoint.resolve()),
            "source_checkpoint_sha256": selected_sha,
            "selection_rule": selection_rule,
            "selected_sample_count": 1,
            "training_epoch_statistics": gate_records[79],
            "deterministic_sample_statistics": self._statistics(80),
            "gating_samples": {
                "path": str(samples.resolve()),
                "size_bytes": samples.stat().st_size,
                "sha256": _sha(samples),
                "source_checkpoint_sha256": selected_sha,
                "selection_rule": selection_rule,
            },
        })
        analysis_artifacts = {
            "controller_block_norms_csv": analysis_dir / "controller.csv",
            "controller_block_norms_png": analysis_dir / "controller.png",
            "test_gate_samples_tsv": samples,
            "test_weight_summary_csv": analysis_dir / "weights.csv",
            "test_weight_distribution_png": analysis_dir / "weights.png",
            "dynamic_gating_summary_json": summary,
        }
        for name, path in analysis_artifacts.items():
            if not path.exists():
                path.write_bytes((name + "\n").encode("ascii"))
        analysis_manifest = self.output / "g2_gating_analysis_manifest.json"
        _write_json(analysis_manifest, {
            "analysis_type": "G2 global-plus-local Dynamic Gating observation",
            "config_path": str(self.config.resolve()),
            "config_sha256": _sha(self.config),
            "checkpoint_path": str(selected_checkpoint.resolve()),
            "checkpoint_sha256": selected_sha,
            "epoch_statistics_path": str(gate_path.resolve()),
            "epoch_statistics_sha256": _sha(gate_path),
            "gating_input": "concat([g, z2, z4, z6])",
            "files": {
                name: {"path": str(path.resolve()), "sha256": _sha(path)}
                for name, path in analysis_artifacts.items()
            },
        })

        result = {
            "experiment": "G2 global-plus-local Dynamic Gating",
            "branch": EXPECTED_BRANCH,
            "commit": COMMIT,
            "seed": 42,
            "gating_input": "concat([g, z2, z4, z6])",
            "gate_outputs": ["w2", "w4", "w6"],
            "checkpoint_selection_rule": (
                "highest Rank-1; if tied, highest mAP; if still tied, earliest epoch"
            ),
            "selected_checkpoint": {
                "path": str(selected_checkpoint.resolve()),
                "sha256": selected_sha,
                "epoch": 80,
                "global_iteration": 80 * 186,
            },
            "metrics": {
                "rank1_percent": 95.2,
                "rank5_percent": 98.0,
                "rank10_percent": 99.0,
                "map_percent": 88.1,
            },
            "selected_epoch_gate_statistics": gate_records[79],
            "test_gate_weight_distribution": [],
            "controller_input_block_coefficient_statistics": [],
            "evidence": {
                "config": str(self.config.resolve()),
                "config_sha256": _sha(self.config),
                "validation_history": str(validation_path.resolve()),
                "validation_history_sha256": _sha(validation_path),
                "epoch_gate_statistics": str(gate_path.resolve()),
                "epoch_gate_statistics_sha256": _sha(gate_path),
                "analysis_manifest": str(analysis_manifest.resolve()),
                "analysis_manifest_sha256": _sha(analysis_manifest),
            },
        }
        result_path = self.output / "g2_formal_result.json"
        _write_json(result_path, result)

        _write_json(self.output / "reproducibility.json", {
            "schema_version": 1,
            "created_at_utc": "2026-08-27T00:00:00Z",
            "seed": 42,
            "seed_applied_before_data_loading": True,
            "seed_chain": {
                "source_config_seed": 42,
                "resolved_config_seed": 42,
                "applied_training_seed": 42,
                "reproducibility_metadata_seed": 42,
            },
            "configuration": {
                "source_file": str(self.config.resolve()),
                "source_file_sha256": _sha(self.config),
                "resolved_file": "config_resolved.yml",
                "resolved_file_sha256": _sha(resolved),
            },
            "command": ["tools/train.py", "--config_file", str(self.config.resolve())],
            "code": {
                "repository": str(self.root),
                "commit": COMMIT,
                "branch": EXPECTED_BRANCH,
                "dirty": False,
            },
            "environment": {
                "hostname": "autodl-test",
                "python_version": "3.10.0",
                "torch_version": "2.0.0",
                "cuda_version": "11.8",
                "gpu_names": ["NVIDIA GeForce RTX 4090"],
            },
        })

    def _recover(self):
        with mock.patch(
                "tools.recover_g2_global_local_experiment._lineage",
                return_value={
                    "parent_branch": "parent",
                    "parent_commit": "a" * 40,
                    "merge_base": "a" * 40,
                }):
            return recover(
                self.config,
                self.output,
                self.console,
                self.records,
                self.experiments,
                started_at_utc="2026-08-27T00:00:00Z",
                ended_at_utc="2026-08-27T01:00:00Z",
                runtime_seconds=3600,
            )

    def test_recovery_registers_g2_once_and_is_idempotent(self):
        run_dir, first_row, created = self._recover()
        self.assertTrue(created)
        self.assertTrue(run_dir.is_dir())
        self.assertEqual(first_row["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(first_row["gating_input"], "concat_global_local")
        self.assertEqual(float(first_row["rank1_percent"]), 95.2)
        self.assertEqual(float(first_row["map_percent"]), 88.1)
        self.assertEqual(int(first_row["selected_epoch"]), 80)

        second_dir, second_row, created = self._recover()
        self.assertFalse(created)
        self.assertEqual(second_dir, run_dir)
        self.assertEqual(second_row["run_id"], first_row["run_id"])

        with (self.records / "runs.csv").open("r", encoding="utf-8", newline="") as handle:
            run_rows = list(csv.DictReader(handle))
        with (self.records / "tables" / "main_results.csv").open(
                "r", encoding="utf-8", newline="") as handle:
            formal_rows = list(csv.DictReader(handle))
        self.assertEqual(len(run_rows), 1)
        self.assertEqual(len(formal_rows), 1)
        markdown = self.experiments.read_text(encoding="utf-8")
        self.assertGreaterEqual(markdown.count(first_row["run_id"]), 2)
        self.assertIn("concat_global_local", markdown)

    def test_missing_console_log_fails_without_registering_result(self):
        self.console.unlink()
        with self.assertRaisesRegex(G2RecoveryError, "console log"):
            self._recover()
        self.assertFalse((self.records / "runs.csv").exists())
        self.assertNotIn(EXPERIMENT_ID, self.experiments.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
