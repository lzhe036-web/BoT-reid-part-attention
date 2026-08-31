import csv
import json
import math
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml
from yacs.config import CfgNode

from config import cfg
from tests.test_recover_g2_global_local_experiment import G2RecoveryTest, _sha, _write_json
from tools.analyze_g2_global_local_gating import _block_rows, _load_configuration
from tools.g2_dynamic_gating_profiles import (
    G2_LOCAL_ONLY_PROFILE,
    G2_WITHOUT_Z2_PROFILE,
)
from tools.recover_g2_global_local_experiment import G2RecoveryError, recover
from utils.config_serialization import serialize_cfg_node_yaml
from utils.dynamic_gating_evidence import dynamic_gating_sample_fields, gating_stat_fields


REPO_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_global_local_autodl.yml"
)
LOCAL_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_local_only_autodl.yml"
)
WITHOUT_Z2_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_without_z2_autodl.yml"
)


class G2LocalOnlyAnalyzerTest(unittest.TestCase):
    def _configuration(self):
        configuration = cfg.clone()
        configuration.merge_from_file(str(LOCAL_CONFIG))
        configuration.freeze()
        return configuration

    def test_local_only_controller_blocks_are_exactly_z2_z4_z6(self):
        state = {
            "multi_granularity_dynamic_gate.controller.weight": torch.randn(3, 768)
        }
        rows, boundaries = _block_rows(
            state, self._configuration(), "a" * 64,
            profile=G2_LOCAL_ONLY_PROFILE,
        )
        self.assertEqual(boundaries, (("z2", 0, 256), ("z4", 256, 512), ("z6", 512, 768)))
        self.assertEqual({row["input_block"] for row in rows}, {"z2", "z4", "z6"})
        self.assertNotIn("g", {row["input_block"] for row in rows})
        self.assertEqual(len(rows), 9)

    def test_local_only_analyzer_rejects_wrong_controller_width_or_mode(self):
        with self.assertRaisesRegex(ValueError, "must be 768"):
            _block_rows(
                {"multi_granularity_dynamic_gate.controller.weight": torch.randn(3, 2816)},
                self._configuration(), "b" * 64, profile=G2_LOCAL_ONLY_PROFILE,
            )
        with self.assertRaisesRegex(ValueError, "concat_local"):
            _load_configuration(GLOBAL_CONFIG, profile=G2_LOCAL_ONLY_PROFILE)

    def test_local_only_formal_scripts_are_independent_and_fail_closed(self):
        smoke = (REPO_ROOT / "scripts" / "test_g2_local_only_gating_1epoch_autodl.sh").read_text(encoding="utf-8")
        formal = (REPO_ROOT / "scripts" / "train_g2_local_only_seed42_autodl.sh").read_text(encoding="utf-8")
        for text in (smoke, formal):
            self.assertIn('EXPECTED_BRANCH="codex/g2-local-only"', text)
            self.assertIn("g2_local_only_tau1_seed42_market1501", text)
            self.assertIn("PYTHONHASHSEED=42", text)
            self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", text)
            self.assertNotIn("g2_global_local_tau1_seed42_market1501", text)
        self.assertIn('git rev-parse --verify "origin/${EXPECTED_BRANCH}"', formal)
        self.assertIn("finalize_g2_local_only_experiment.py", formal)
        self.assertIn("recover_g2_local_only_experiment.py", formal)


class G2LocalOnlyRecoveryTest(G2RecoveryTest):
    """Reuse the full machine-evidence fixture, changing only variant identity."""

    def _build_fixture(self):
        super(G2LocalOnlyRecoveryTest, self)._build_fixture()

        source = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        source["MODEL"]["MULTI_GRANULARITY_GATING_INPUT"] = "concat_local"
        self.config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
        resolved = self.output / "config_resolved.yml"
        resolved.write_text(serialize_cfg_node_yaml(CfgNode(source)), encoding="utf-8")

        reproducibility_path = self.output / "reproducibility.json"
        reproducibility = json.loads(reproducibility_path.read_text(encoding="utf-8"))
        reproducibility["configuration"]["source_file_sha256"] = _sha(self.config)
        reproducibility["configuration"]["resolved_file_sha256"] = _sha(resolved)
        reproducibility["code"]["branch"] = G2_LOCAL_ONLY_PROFILE.expected_branch
        _write_json(reproducibility_path, reproducibility)

        analysis_manifest_path = self.output / "g2_gating_analysis_manifest.json"
        analysis = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
        analysis["analysis_type"] = "G2-local-only Dynamic Gating observation"
        analysis["config_sha256"] = _sha(self.config)
        analysis["gating_input"] = "concat([z2,z4,z6])"
        pdf = self.output / "g2_gating_analysis" / "weights.pdf"
        pdf.write_bytes(b"fixture pdf\n")
        analysis["files"]["test_weight_distribution_pdf"] = {
            "path": str(pdf.resolve()), "sha256": _sha(pdf)
        }
        _write_json(analysis_manifest_path, analysis)

        global_result = self.output / "g2_formal_result.json"
        result = json.loads(global_result.read_text(encoding="utf-8"))
        result["experiment"] = G2_LOCAL_ONLY_PROFILE.experiment_label
        result["branch"] = G2_LOCAL_ONLY_PROFILE.expected_branch
        result["gating_input"] = G2_LOCAL_ONLY_PROFILE.gating_input_semantics
        result["evidence"]["config_sha256"] = _sha(self.config)
        result["evidence"]["analysis_manifest_sha256"] = _sha(analysis_manifest_path)
        _write_json(
            self.output / G2_LOCAL_ONLY_PROFILE.formal_result_filename, result
        )

    def _recover(self):
        with mock.patch(
                "tools.recover_g2_global_local_experiment._lineage",
                return_value={
                    "parent_branch": G2_LOCAL_ONLY_PROFILE.expected_parent_branch,
                    "parent_commit": "a" * 40,
                    "merge_base": "a" * 40,
                }):
            return recover(
                self.config, self.output, self.console, self.records,
                self.experiments,
                started_at_utc="2026-08-27T00:00:00Z",
                ended_at_utc="2026-08-27T01:00:00Z",
                runtime_seconds=3600,
                profile=G2_LOCAL_ONLY_PROFILE,
            )

    def test_local_only_recovery_is_idempotent_and_registers_one_row(self):
        run_dir, row, created = self._recover()
        self.assertTrue(created)
        self.assertEqual(row["experiment_id"], G2_LOCAL_ONLY_PROFILE.experiment_id)
        self.assertEqual(row["gating_input"], "concat_local")
        self.assertEqual(row["method_variant"], G2_LOCAL_ONLY_PROFILE.method_variant)
        self.assertTrue((run_dir / G2_LOCAL_ONLY_PROFILE.formal_result_filename).is_file())
        _second_dir, second_row, created = self._recover()
        self.assertFalse(created)
        self.assertEqual(second_row["run_id"], row["run_id"])
        with (self.records / "runs.csv").open("r", encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_recovery_registers_g2_once_and_is_idempotent(self):
        """Override the inherited global-identity assertion for concat_local."""
        self.test_local_only_recovery_is_idempotent_and_registers_one_row()


class G2WithoutZ2AnalyzerTest(unittest.TestCase):
    def _configuration(self):
        configuration = cfg.clone()
        configuration.merge_from_file(str(WITHOUT_Z2_CONFIG))
        configuration.freeze()
        return configuration

    def test_without_z2_controller_blocks_are_exactly_z4_z6(self):
        state = {
            "multi_granularity_dynamic_gate.controller.weight": torch.randn(2, 512)
        }
        rows, boundaries = _block_rows(
            state, self._configuration(), "c" * 64,
            profile=G2_WITHOUT_Z2_PROFILE,
        )
        self.assertEqual(boundaries, (("z4", 0, 256), ("z6", 256, 512)))
        self.assertEqual({row["input_block"] for row in rows}, {"z4", "z6"})
        self.assertEqual({row["target_gate"] for row in rows}, {"w4", "w6"})
        self.assertEqual(len(rows), 4)

    def test_without_z2_analyzer_rejects_wrong_width_or_mode(self):
        with self.assertRaisesRegex(ValueError, "must be 512"):
            _block_rows(
                {"multi_granularity_dynamic_gate.controller.weight": torch.randn(2, 768)},
                self._configuration(), "d" * 64, profile=G2_WITHOUT_Z2_PROFILE,
            )
        with self.assertRaisesRegex(ValueError, "concat_z4_z6"):
            _load_configuration(LOCAL_CONFIG, profile=G2_WITHOUT_Z2_PROFILE)

    def test_without_z2_formal_scripts_are_independent_and_fail_closed(self):
        smoke = (REPO_ROOT / "scripts" / "test_g2_without_z2_gating_1epoch_autodl.sh").read_text(encoding="utf-8")
        formal = (REPO_ROOT / "scripts" / "train_g2_without_z2_seed42_autodl.sh").read_text(encoding="utf-8")
        for text in (smoke, formal):
            self.assertIn('EXPECTED_BRANCH="codex/g2-without-z2"', text)
            self.assertIn("g2_without_z2_tau1_seed42_market1501", text)
            self.assertIn("PYTHONHASHSEED=42", text)
            self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", text)
            self.assertNotIn("g2_local_only_tau1_seed42_market1501", text)
        self.assertIn('git rev-parse --verify "origin/${EXPECTED_BRANCH}"', formal)
        self.assertIn("finalize_g2_without_z2_experiment.py", formal)
        self.assertIn("recover_g2_without_z2_experiment.py", formal)


class G2WithoutZ2RecoveryTest(G2RecoveryTest):
    """Exercise strict recovery using only two-way z4/z6 machine evidence."""

    @staticmethod
    def _two_way_statistics(epoch):
        values = {
            "gating_temperature": 1.0,
            "gating_sample_count": 11904,
            "mean_gate_entropy": 0.67 + epoch / 100000.0,
        }
        for label, mean in (("4", 0.4), ("6", 0.6)):
            values["p{}_mean".format(label)] = mean
            values["p{}_std".format(label)] = 0.01
            values["p{}_min".format(label)] = mean - 0.02
            values["p{}_max".format(label)] = mean + 0.02
            values["applied_w{}_mean".format(label)] = mean
            values["applied_w{}_std".format(label)] = 0.01
            values["dominant_k{}_ratio".format(label)] = mean
        if set(values) != set(gating_stat_fields((4, 6))):
            raise AssertionError("Two-way gate schema drifted")
        return values

    def _build_fixture(self):
        super(G2WithoutZ2RecoveryTest, self)._build_fixture()

        source = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        source["MODEL"]["MULTI_GRANULARITY_GATING_INPUT"] = "concat_z4_z6"
        self.config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
        resolved = self.output / "config_resolved.yml"
        resolved.write_text(serialize_cfg_node_yaml(CfgNode(source)), encoding="utf-8")

        reproducibility_path = self.output / "reproducibility.json"
        reproducibility = json.loads(reproducibility_path.read_text(encoding="utf-8"))
        reproducibility["configuration"]["source_file_sha256"] = _sha(self.config)
        reproducibility["configuration"]["resolved_file_sha256"] = _sha(resolved)
        reproducibility["code"]["branch"] = G2_WITHOUT_Z2_PROFILE.expected_branch
        _write_json(reproducibility_path, reproducibility)

        gate_records = []
        for epoch in range(1, 121):
            row = {
                "epoch": epoch,
                "global_iteration": epoch * 186,
                "epoch_length": 186,
                "gating_scales": [4, 6],
            }
            row.update(self._two_way_statistics(epoch))
            gate_records.append(row)
        gate_path = self.output / "dynamic_gating_epoch_stats.jsonl"
        gate_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in gate_records),
            encoding="utf-8",
        )

        checkpoint = self.output / "resnet50_checkpoint_14880.pt"
        checkpoint_sha = _sha(checkpoint)
        analysis_dir = self.output / G2_WITHOUT_Z2_PROFILE.analysis_directory_name
        analysis_dir.mkdir()
        samples = analysis_dir / "gating_samples.tsv"
        sample_row = {
            "stable_sample_key": "q:1", "dataset_split": "query",
            "pid": 1, "camid": 2, "p4": 0.4, "p6": 0.6,
            "w4": 0.4, "w6": 0.6,
            "entropy": -(0.4 * math.log(0.4) + 0.6 * math.log(0.6)),
            "dominant_k": 6, "checkpoint_sha256": checkpoint_sha,
        }
        with samples.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=dynamic_gating_sample_fields((4, 6)),
                delimiter="\t", lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(sample_row)
        summary = analysis_dir / "dynamic_gating_summary.json"
        selection_rule = "sha256(stable_sample_key) ascending; first 256 query+gallery samples"
        _write_json(summary, {
            "schema_version": 5,
            "source_checkpoint_path": str(checkpoint.resolve()),
            "source_checkpoint_sha256": checkpoint_sha,
            "selection_rule": selection_rule,
            "selected_sample_count": 1,
            "training_epoch_statistics": gate_records[79],
            "deterministic_sample_statistics": self._two_way_statistics(80),
            "gating_samples": {
                "path": str(samples.resolve()), "size_bytes": samples.stat().st_size,
                "sha256": _sha(samples), "source_checkpoint_sha256": checkpoint_sha,
                "selection_rule": selection_rule,
            },
        })
        artifacts = {
            "controller_block_norms_csv": analysis_dir / "controller.csv",
            "controller_block_norms_png": analysis_dir / "controller.png",
            "test_gate_samples_tsv": samples,
            "test_weight_summary_csv": analysis_dir / "weights.csv",
            "test_weight_distribution_png": analysis_dir / "weights.png",
            "test_weight_distribution_pdf": analysis_dir / "weights.pdf",
            "dynamic_gating_summary_json": summary,
        }
        for name, path in artifacts.items():
            if not path.exists():
                path.write_bytes((name + "\n").encode("ascii"))
        analysis_manifest = self.output / G2_WITHOUT_Z2_PROFILE.analysis_manifest_filename
        _write_json(analysis_manifest, {
            "analysis_type": "G2-without-z2 Dynamic Gating observation",
            "config_path": str(self.config.resolve()), "config_sha256": _sha(self.config),
            "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
            "epoch_statistics_path": str(gate_path.resolve()),
            "epoch_statistics_sha256": _sha(gate_path),
            "gating_input": G2_WITHOUT_Z2_PROFILE.gating_input_semantics,
            "files": {
                name: {"path": str(path.resolve()), "sha256": _sha(path)}
                for name, path in artifacts.items()
            },
        })

        result = json.loads((self.output / "g2_formal_result.json").read_text(encoding="utf-8"))
        result["experiment"] = G2_WITHOUT_Z2_PROFILE.experiment_label
        result["branch"] = G2_WITHOUT_Z2_PROFILE.expected_branch
        result["gating_input"] = G2_WITHOUT_Z2_PROFILE.gating_input_semantics
        result["gate_outputs"] = ["w4", "w6"]
        result["selected_epoch_gate_statistics"] = gate_records[79]
        result["evidence"]["config_sha256"] = _sha(self.config)
        result["evidence"]["epoch_gate_statistics_sha256"] = _sha(gate_path)
        result["evidence"]["analysis_manifest"] = str(analysis_manifest.resolve())
        result["evidence"]["analysis_manifest_sha256"] = _sha(analysis_manifest)
        _write_json(self.output / G2_WITHOUT_Z2_PROFILE.formal_result_filename, result)

    def _recover(self):
        with mock.patch(
                "tools.recover_g2_global_local_experiment._lineage",
                return_value={
                    "parent_branch": G2_WITHOUT_Z2_PROFILE.expected_parent_branch,
                    "parent_commit": "a" * 40,
                    "merge_base": "a" * 40,
                }):
            return recover(
                self.config, self.output, self.console, self.records,
                self.experiments,
                started_at_utc="2026-08-27T00:00:00Z",
                ended_at_utc="2026-08-27T01:00:00Z",
                runtime_seconds=3600,
                profile=G2_WITHOUT_Z2_PROFILE,
            )

    def test_without_z2_recovery_is_idempotent_and_registers_two_way_schema(self):
        run_dir, row, created = self._recover()
        self.assertTrue(created)
        self.assertEqual(row["experiment_id"], G2_WITHOUT_Z2_PROFILE.experiment_id)
        self.assertEqual(row["gating_input"], "concat_z4_z6")
        self.assertEqual(row["scale_order"], "4,6")
        self.assertEqual(row["p2_mean"], "not_recorded")
        self.assertTrue((run_dir / G2_WITHOUT_Z2_PROFILE.formal_result_filename).is_file())
        _second_dir, second_row, created = self._recover()
        self.assertFalse(created)
        self.assertEqual(second_row["run_id"], row["run_id"])
        with (self.records / "runs.csv").open("r", encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_recovery_rejects_legacy_z2_epoch_evidence(self):
        gate_path = self.output / "dynamic_gating_epoch_stats.jsonl"
        rows = [json.loads(line) for line in gate_path.read_text(encoding="utf-8").splitlines()]
        rows[0]["p2_mean"] = 0.0
        gate_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(G2RecoveryError, "must not record z2"):
            self._recover()

    def test_recovery_registers_g2_once_and_is_idempotent(self):
        """Override the inherited global-identity assertion for concat_z4_z6."""
        self.test_without_z2_recovery_is_idempotent_and_registers_two_way_schema()
