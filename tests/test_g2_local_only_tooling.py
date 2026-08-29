import csv
import json
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml
from yacs.config import CfgNode

from config import cfg
from tests.test_recover_g2_global_local_experiment import G2RecoveryTest, _sha, _write_json
from tools.analyze_g2_global_local_gating import _block_rows, _load_configuration
from tools.g2_dynamic_gating_profiles import G2_LOCAL_ONLY_PROFILE
from tools.recover_g2_global_local_experiment import recover
from utils.config_serialization import serialize_cfg_node_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_global_local_autodl.yml"
)
LOCAL_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
    "g2_local_only_autodl.yml"
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
