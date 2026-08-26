# encoding: utf-8
"""Algorithm and protocol tests for the windowed Soft-Min sweep."""

from __future__ import absolute_import

import itertools
import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from layers.part_correspondence_consistency import (
    soft_min_path_costs,
    validate_softmin_window,
    windowed_soft_min_path_costs,
)
from utils.experiment_recording import (
    EvidenceError,
    NOT_APPLICABLE,
    STRICT_MANIFEST_REQUIRED_FIELDS,
    WINDOWED_SOFT_ALIGNMENT_FAMILY,
    _windowed_soft_alignment_table_eligible,
    experiment_identity,
    parse_training_log,
    validate_strict_manifest_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2l03_soft_min_alignment_tau0p2_lambda0p05_autodl.yml"
)
WINDOW_CONFIGS = {
    1: REPO_ROOT / "configs" / (
        "softmax_triplet_c2l03_windowed_soft_min_alignment_"
        "tau0p2_lambda0p05_w1_autodl.yml"
    ),
    2: REPO_ROOT / "configs" / (
        "softmax_triplet_c2l03_windowed_soft_min_alignment_"
        "tau0p2_lambda0p05_w2_autodl.yml"
    ),
}


def _changed_leaf_paths(left, right, prefix=""):
    paths = set()
    for key in set(left) | set(right):
        path = "{}.{}".format(prefix, key) if prefix else str(key)
        if key not in left or key not in right:
            paths.add(path)
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            paths.update(_changed_leaf_paths(left[key], right[key], path))
        elif left[key] != right[key]:
            paths.add(path)
    return paths


def _enumerate_band_path_costs(matrix, window):
    parts = int(matrix.size(0))
    costs = []
    for down_positions in itertools.combinations(
            range(2 * parts - 2), parts - 1):
        down_positions = set(down_positions)
        row = column = 0
        valid = abs(row - column) <= window
        cost = matrix[row, column]
        for move_index in range(2 * parts - 2):
            if move_index in down_positions:
                row += 1
            else:
                column += 1
            valid = valid and abs(row - column) <= window
            cost = cost + matrix[row, column]
        if valid:
            costs.append(cost)
    return torch.stack(costs)


class WindowedSoftMinAlignmentTest(unittest.TestCase):
    def test_dynamic_program_matches_enumerated_band_paths(self):
        torch.manual_seed(86)
        matrices = torch.randn(3, 4, 4, dtype=torch.float64)
        tau = 0.31
        for window in (1, 2):
            expected = torch.stack([
                -tau * torch.logsumexp(
                    -_enumerate_band_path_costs(matrix, window) / tau,
                    dim=0,
                ) for matrix in matrices
            ])
            actual = windowed_soft_min_path_costs(matrices, tau, window)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-11))

    def test_k6_band_counts_and_valid_windows(self):
        for window, expected_cells in ((1, 16), (2, 24)):
            cells = [
                (row, column) for row in range(6) for column in range(6)
                if abs(row - column) <= window
            ]
            self.assertEqual(len(cells), expected_cells)
            self.assertEqual(validate_softmin_window(window, 6), window)

    def test_unrestricted_window_matches_forward_and_gradient(self):
        torch.manual_seed(87)
        unrestricted = torch.randn(2, 4, 4, dtype=torch.float64,
                                   requires_grad=True)
        windowed = unrestricted.detach().clone().requires_grad_(True)
        expected = soft_min_path_costs(unrestricted, 0.2).sum()
        actual = windowed_soft_min_path_costs(windowed, 0.2, 3).sum()
        expected.backward()
        actual.backward()
        self.assertTrue(torch.allclose(expected.detach(), actual.detach()))
        self.assertTrue(torch.allclose(unrestricted.grad, windowed.grad))

    def test_out_of_band_elements_have_no_effect_or_gradient(self):
        torch.manual_seed(88)
        matrix = torch.randn(1, 5, 5, dtype=torch.float64, requires_grad=True)
        output = windowed_soft_min_path_costs(matrix, 0.2, 1).sum()
        output.backward()
        mask = torch.tensor([
            [abs(row - column) <= 1 for column in range(5)]
            for row in range(5)
        ], dtype=torch.bool)
        self.assertTrue(torch.equal(
            matrix.grad[0][~mask], torch.zeros_like(matrix.grad[0][~mask])
        ))
        perturbed = matrix.detach().clone()
        perturbed[0][~mask] += 1.0e6
        self.assertTrue(torch.allclose(
            output.detach(),
            windowed_soft_min_path_costs(perturbed, 0.2, 1).sum(),
        ))

    def test_w1_w2_gradients_are_finite(self):
        for window in (1, 2):
            local = torch.randn(2, 6, 6, dtype=torch.float64,
                                requires_grad=True)
            loss = windowed_soft_min_path_costs(local, 0.2, window).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(local.grad).all())

    def test_invalid_windows_fail_closed(self):
        for window in (0, -1, 1.0, True, "1"):
            with self.subTest(window=window):
                with self.assertRaises(ValueError):
                    validate_softmin_window(window, 6)
        self.assertEqual(validate_softmin_window(0, 1), 0)

    def test_configs_have_only_declared_protocol_differences(self):
        base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        outputs = set()
        for window, path in WINDOW_CONFIGS.items():
            candidate = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(
                _changed_leaf_paths(base, candidate),
                {"MODEL.PCC_MODE", "MODEL.PCC_SOFTMIN_WINDOW", "OUTPUT_DIR"},
            )
            self.assertEqual(candidate["MODEL"]["PCC_SOFTMIN_WINDOW"], window)
            identity = experiment_identity(candidate)
            self.assertEqual(identity["alignment_mode"], "windowed_soft_min")
            self.assertEqual(identity["alignment_window"], window)
            self.assertEqual(identity["alignment_temperature"], 0.2)
            self.assertEqual(candidate["SOLVER"]["MAX_EPOCHS"], 120)
            self.assertEqual(candidate["SEED"], 42)
            self.assertEqual(candidate["MODEL"]["PCC_LAMBDA"], 0.05)
            self.assertEqual(
                candidate["MODEL"]["CROSS_CAMERA_POSITIVE_LAMBDA"], 0.3
            )
            outputs.add(candidate["OUTPUT_DIR"])
        self.assertEqual(len(outputs), 2)

    def test_window_specific_runners_are_isolated(self):
        branch = "exp/c2l03-windowed-soft-min-alignment-window-sweep-tau0p2-lambda0p05"
        parent = "67b7bbf528a0a6279a3f9ab86aed43ad91b1ef63"
        for window in (1, 2):
            suffix = "w{}_autodl.sh".format(window)
            expected_id = "C2-L03-WSOFTMIN-W{}-T0P2-LP0P05-S42".format(window)
            for prefix, run_kind in (("test_", "smoke"), ("train_", "formal")):
                path = REPO_ROOT / "scripts" / (
                    prefix + "c2l03_windowed_soft_min_alignment_" + suffix
                )
                text = path.read_text(encoding="utf-8")
                self.assertIn('EXPECTED_BRANCH="{}"'.format(branch), text)
                self.assertIn(parent, text)
                self.assertIn("--run-kind {}".format(run_kind), text)
                self.assertIn(expected_id, text)
                self.assertIn("--expected-commit", text)
                self.assertIn("--feature-reference-commit", text)
                if run_kind == "formal":
                    self.assertIn(expected_id + "-SMOKE", text)
                    self.assertNotIn(
                        "C2-L03-WSOFTMIN-W{}-T0P2-LP0P05-S42-SMOKE".format(
                            1 if window == 2 else 2
                        ),
                        text,
                    )

    def test_strict_manifest_accepts_only_empty_raw_porcelain(self):
        manifest = {field: "evidence" for field in STRICT_MANIFEST_REQUIRED_FIELDS}
        manifest.update({
            "upstream": "not_recorded",
            "git_status_porcelain_raw": "",
            "git_staged_diff_empty": True,
            "git_unstaged_diff_empty": True,
            "git_operations_in_progress": [],
            "git_preflight_clean": True,
            "git_status_preflight": [],
        })
        validate_strict_manifest_preflight(manifest)
        absent = dict(manifest)
        del absent["git_status_porcelain_raw"]
        with self.assertRaisesRegex(EvidenceError, "Strict experiment manifest lacks"):
            validate_strict_manifest_preflight(absent)
        dirty = dict(manifest)
        dirty["git_status_porcelain_raw"] = " M layers/x.py"
        with self.assertRaisesRegex(EvidenceError, "Raw preflight porcelain"):
            validate_strict_manifest_preflight(dirty)

    def test_parser_and_windowed_table_filter_are_window_aware(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "log.txt"
            log.write_text("\n".join([
                "2026-08-27 10:00:00 INFO: Epoch[1] Iteration[1/1] loss_total: 1.0, loss_id: 0.2, loss_triplet: 0.3, Base Lr: 1e-4",
                "2026-08-27 10:00:01 INFO: Windowed Soft Alignment Epoch Summary - Epoch: 1 window: 2 alignment_temperature: 0.2 windowed_soft_alignment_loss: 0.3 valid_alignment_pair_count: 10 mean_windowed_soft_path_cost: 3.3",
                "2026-08-27 10:00:02 INFO: EPOCH_EVIDENCE epoch=1 global_iteration=1 epoch_length=1",
                "2026-08-27 10:00:03 INFO: Validation Results - Epoch: 1",
                "2026-08-27 10:00:04 INFO: mAP: 80.0%",
                "2026-08-27 10:00:05 INFO: CMC curve, Rank-1  :90.0%",
                "2026-08-27 10:00:06 INFO: CMC curve, Rank-5  :96.0%",
                "2026-08-27 10:00:07 INFO: CMC curve, Rank-10 :98.0%",
                "",
            ]), encoding="utf-8")
            parsed = parse_training_log(log)
            self.assertEqual(parsed["alignment_window"], 2)
            self.assertEqual(parsed["valid_alignment_pair_count"], 10)
            self.assertAlmostEqual(
                parsed["windowed_soft_alignment_loss"], 0.3
            )
        manifest = {
            "experiment_family": WINDOWED_SOFT_ALIGNMENT_FAMILY,
            "run_kind": "formal", "status": "success",
            "alignment_mode": "windowed_soft_min",
            "alignment_temperature": 0.2, "pcc_lambda": 0.05,
            "alignment_window": 1,
        }
        self.assertTrue(_windowed_soft_alignment_table_eligible(manifest))
        manifest["alignment_window"] = 3
        self.assertFalse(_windowed_soft_alignment_table_eligible(manifest))


if __name__ == "__main__":
    unittest.main()
