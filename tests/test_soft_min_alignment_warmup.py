# encoding: utf-8
"""Protocol, loss-gate, evidence, and runner tests for warmup20."""

from __future__ import absolute_import

import subprocess
import unittest
from pathlib import Path

import torch
import yaml

from config import cfg
from layers import (
    effective_pcc_lambda,
    make_loss,
    validate_pcc_warmup_epochs,
)
from tools.run_experiment import _build_config_comparison
from tools.validate_c2l03_soft_min_alignment_warmup import (
    BASELINE_BRANCH,
    BASELINE_CONFIG,
    BASELINE_SHA,
    EXPECTED_DIFFERENCES,
    WARMUP_CONFIG,
)
from utils.experiment_recording import (
    MAIN_FIELDS,
    RUN_FIELDS,
    SCHEMA_VERSION,
    SOFT_ALIGNMENT_WARMUP_FIELDS,
    SOFT_WARMUP_BASELINE_EXPERIMENT_ID,
    SOFT_WARMUP_FAMILY,
    TABLE_SCHEMAS,
    _soft_alignment_warmup_table_eligible,
    validate_local_alignment_warmup_evidence,
)
from utils.reproducibility import resolved_config_text


REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH = "exp/c2l03-soft-min-alignment-warmup20-tau0p2-lambda0p05"


def _load(path):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(path))
    local_cfg.freeze()
    return local_cfg


class WarmupProtocolTest(unittest.TestCase):
    def test_branch_is_directly_based_on_fixed_remote_sha(self):
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=str(REPO_ROOT), text=True
        ).strip()
        baseline = subprocess.check_output(
            ["git", "rev-parse", BASELINE_BRANCH],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
        merge_base = subprocess.check_output(
            ["git", "merge-base", BASELINE_BRANCH, "HEAD"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
        self.assertEqual(branch, BRANCH)
        self.assertEqual(baseline, BASELINE_SHA)
        self.assertEqual(merge_base, BASELINE_SHA)

    def test_resolved_config_diff_is_exact(self):
        baseline = _load(BASELINE_CONFIG)
        candidate = _load(WARMUP_CONFIG)
        evidence = _build_config_comparison(
            BASELINE_CONFIG, baseline, WARMUP_CONFIG, candidate,
            EXPECTED_DIFFERENCES,
        )
        self.assertEqual(evidence["status"], "pass")
        self.assertEqual(
            set(evidence["observed_differences"]), set(EXPECTED_DIFFERENCES)
        )
        self.assertEqual(candidate.SEED, baseline.SEED)
        self.assertEqual(candidate.SEED, 42)
        self.assertEqual(candidate.DATASETS.NAMES, baseline.DATASETS.NAMES)
        self.assertEqual(candidate.MODEL.PCC_SOFTMIN_TAU, 0.2)
        self.assertEqual(candidate.MODEL.PCC_LAMBDA, 0.05)
        self.assertEqual(candidate.MODEL.PCC_PARTS, 6)
        self.assertEqual(candidate.MODEL.PCC_WARMUP_EPOCHS, 20)
        self.assertEqual(baseline.MODEL.PCC_WARMUP_EPOCHS, 0)
        self.assertEqual(candidate.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA, 0.3)
        self.assertEqual(candidate.SOLVER.MAX_EPOCHS, 120)
        self.assertEqual(
            candidate.SOLVER.WARMUP_ITERS, baseline.SOLVER.WARMUP_ITERS
        )

    def test_formal_script_has_fixed_identity_and_machine_diff_gate(self):
        script = REPO_ROOT / "scripts" / (
            "train_c2l03_soft_min_alignment_tau0p2_lambda0p05_"
            "warmup20_autodl.sh"
        )
        text = script.read_text(encoding="utf-8")
        self.assertIn('EXPECTED_BRANCH="{}"'.format(BRANCH), text)
        self.assertIn('PARENT_COMMIT="{}"'.format(BASELINE_SHA), text)
        self.assertIn("--expected-config-difference MODEL.PCC_WARMUP_EPOCHS", text)
        self.assertIn("--expected-config-difference OUTPUT_DIR", text)
        self.assertIn("--run-kind formal", text)
        self.assertNotIn("MODEL.PCC_LAMBDA 0.05", text)
        self.assertNotIn("--required-smoke-experiment-id", text)


class WarmupLossGateTest(unittest.TestCase):
    def test_effective_lambda_boundaries_and_validation(self):
        expected = {1: 0.0, 20: 0.0, 21: 0.05, 120: 0.05}
        for epoch, value in expected.items():
            self.assertEqual(effective_pcc_lambda(0.05, 20, epoch), value)
        self.assertEqual(effective_pcc_lambda(0.05, 0, 1), 0.05)
        for invalid in (-1, 1.5, True, "invalid"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_pcc_warmup_epochs(invalid)

    def test_gate_removes_only_local_gradient_and_default_zero_is_legacy(self):
        candidate = _load(WARMUP_CONFIG).clone()
        candidate.defrost()
        candidate.MODEL.IF_LABELSMOOTH = "off"
        candidate.freeze()
        torch.manual_seed(20260824)
        score_source = torch.randn(4, 2)
        feature_source = torch.randn(4, 16)
        local_source = torch.randn(4, 6, 16)
        pids = torch.tensor([0, 0, 1, 1])
        camids = torch.tensor([0, 1, 0, 1])

        def evaluate(local_cfg, epoch):
            score = score_source.clone().requires_grad_(True)
            feature = feature_source.clone().requires_grad_(True)
            local = local_source.clone().requires_grad_(True)
            loss_fn = make_loss(local_cfg, 2)
            loss_fn.set_epoch(epoch)
            output = loss_fn(score, feature, pids, camids, local)
            output["loss_total"].backward()
            return output, score.grad, feature.grad, local.grad

        for epoch in (1, 20):
            output, score_grad, feature_grad, local_grad = evaluate(
                candidate, epoch
            )
            self.assertEqual(output["pcc_effective_lambda"], 0.0)
            self.assertTrue(torch.count_nonzero(score_grad).item() > 0)
            self.assertTrue(torch.count_nonzero(feature_grad).item() > 0)
            self.assertTrue(
                local_grad is None
                or torch.count_nonzero(local_grad).item() == 0
            )
        for epoch in (21, 120):
            output, _, _, local_grad = evaluate(candidate, epoch)
            self.assertEqual(output["pcc_effective_lambda"], 0.05)
            self.assertIsNotNone(local_grad)
            self.assertTrue(torch.count_nonzero(local_grad).item() > 0)

        baseline = _load(BASELINE_CONFIG).clone()
        baseline.defrost()
        baseline.MODEL.IF_LABELSMOOTH = "off"
        baseline.freeze()
        output, _, _, local_grad = evaluate(baseline, 1)
        self.assertEqual(baseline.MODEL.PCC_WARMUP_EPOCHS, 0)
        self.assertEqual(output["pcc_effective_lambda"], 0.05)
        self.assertTrue(torch.count_nonzero(local_grad).item() > 0)


class WarmupEvidenceTest(unittest.TestCase):
    def test_full_epoch_evidence_and_boundary_projection(self):
        configuration = _load(WARMUP_CONFIG)
        gates = []
        for epoch in range(1, 121):
            gates.append({
                "epoch": epoch,
                "configured_lambda": 0.05,
                "effective_lambda": 0.0 if epoch <= 20 else 0.05,
                "warmup_epochs": 20,
                "active": epoch <= 20,
                "alignment_temperature": 0.2,
                "id_loss_enabled": True,
                "triplet_loss_enabled": True,
                "cross_camera_positive_enabled": True,
            })
        evidence = validate_local_alignment_warmup_evidence(
            {"local_alignment_gates": gates},
            yaml.safe_load(resolved_config_text(configuration)),
        )
        self.assertEqual(evidence["status"], "pass")
        self.assertEqual(
            [row["epoch"] for row in evidence["boundary_evidence"]],
            [1, 20, 21, 120],
        )
        self.assertEqual(evidence["all_epoch_gate_count"], 120)
        tampered = list(gates)
        tampered[20] = dict(tampered[20], effective_lambda=0.0)
        with self.assertRaisesRegex(RuntimeError, "epoch 21"):
            validate_local_alignment_warmup_evidence(
                {"local_alignment_gates": tampered},
                yaml.safe_load(resolved_config_text(configuration)),
            )

    def test_machine_table_schema_and_eligibility_are_explicit(self):
        self.assertEqual(SCHEMA_VERSION, 5)
        self.assertEqual(
            TABLE_SCHEMAS["soft_alignment_warmup_comparison"],
            SOFT_ALIGNMENT_WARMUP_FIELDS,
        )
        self.assertIn("local_alignment_warmup_epochs", MAIN_FIELDS)
        self.assertIn("local_alignment_warmup_epochs", RUN_FIELDS)
        warmup = {
            "experiment_family": SOFT_WARMUP_FAMILY,
            "experiment_id": (
                "C2-L03-SOFTMIN-T0P2-LP0P05-WARMUP20-S42"
            ),
            "run_kind": "formal", "status": "success",
            "alignment_mode": "soft_min", "alignment_temperature": 0.2,
            "pcc_lambda": 0.05, "local_alignment_warmup_epochs": 20,
        }
        baseline = dict(
            warmup,
            experiment_family="c2l03_soft_min_alignment_lambda_sweep_tau0p2",
            experiment_id=SOFT_WARMUP_BASELINE_EXPERIMENT_ID,
            local_alignment_warmup_epochs=0,
        )
        self.assertTrue(_soft_alignment_warmup_table_eligible(warmup))
        self.assertTrue(_soft_alignment_warmup_table_eligible(baseline))
        self.assertFalse(_soft_alignment_warmup_table_eligible(
            dict(warmup, pcc_lambda=0.1)
        ))


if __name__ == "__main__":
    unittest.main()
