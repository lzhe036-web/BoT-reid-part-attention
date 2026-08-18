# encoding: utf-8
"""Protocol and fail-closed smoke-gate tests for the Soft-Min tau sweep."""

from __future__ import absolute_import

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from config import cfg
from utils.experiment_recording import sha256_file
from utils.reproducibility import resolved_config_text
from utils.smoke_gate import SmokeGateError, validate_formal_smoke_gate


REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH = "exp/c2l03-soft-min-alignment-tau-sweep"
HARD_SHA = "6b46f2c3747124b97d59ed5cf987f33efb82282b"
FAMILY = "c2l03_soft_min_alignment_tau_sweep"
BASE_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2l03_soft_min_alignment_autodl.yml"
)
SWEEP_CASES = (
    {
        "tag": "tau0p05",
        "tau": 0.05,
        "config": "softmax_triplet_c2l03_soft_min_alignment_tau0p05_autodl.yml",
        "output": "/root/autodl-tmp/experiments/BoT/c2l03_soft_min_alignment_tau0p05_seed42_market1501",
        "smoke_id": "C2-L03-SOFTMIN-T0P05-S42-SMOKE",
        "formal_id": "C2-L03-SOFTMIN-T0P05-S42",
    },
    {
        "tag": "tau0p2",
        "tau": 0.2,
        "config": "softmax_triplet_c2l03_soft_min_alignment_tau0p2_autodl.yml",
        "output": "/root/autodl-tmp/experiments/BoT/c2l03_soft_min_alignment_tau0p2_seed42_market1501",
        "smoke_id": "C2-L03-SOFTMIN-T0P2-S42-SMOKE",
        "formal_id": "C2-L03-SOFTMIN-T0P2-S42",
    },
)


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


def _git(repo, *args):
    return subprocess.check_output(
        ["git"] + list(args), cwd=str(repo), text=True
    ).strip()


def _resolved_configuration(config_path):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(config_path))
    local_cfg.freeze()
    return yaml.safe_load(resolved_config_text(local_cfg))


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class SoftMinTauSweepProtocolTest(unittest.TestCase):
    def test_candidate_configs_change_only_tau_and_output_dir(self):
        baseline = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        for case in SWEEP_CASES:
            path = REPO_ROOT / "configs" / case["config"]
            candidate = yaml.safe_load(path.read_text(encoding="utf-8"))
            with self.subTest(tau=case["tau"]):
                self.assertEqual(
                    _changed_leaf_paths(baseline, candidate),
                    {"MODEL.PCC_SOFTMIN_TAU", "OUTPUT_DIR"},
                )
                self.assertEqual(candidate["MODEL"]["PCC_SOFTMIN_TAU"], case["tau"])
                self.assertEqual(candidate["MODEL"]["PCC_LAMBDA"], 0.1)
                self.assertEqual(candidate["MODEL"]["PCC_PARTS"], 6)
                self.assertEqual(candidate["MODEL"]["PCC_MODE"], "soft_min")
                self.assertEqual(candidate["SEED"], 42)
                self.assertEqual(candidate["SOLVER"]["MAX_EPOCHS"], 120)
                self.assertEqual(candidate["OUTPUT_DIR"], case["output"])

    def test_runners_are_isolated_and_formals_require_matching_smoke(self):
        seen_ids = set()
        seen_outputs = set()
        for case in SWEEP_CASES:
            smoke = REPO_ROOT / "scripts" / (
                "test_c2l03_soft_min_alignment_{}_1epoch.sh".format(case["tag"])
            )
            formal = REPO_ROOT / "scripts" / (
                "train_c2l03_soft_min_alignment_{}_autodl.sh".format(case["tag"])
            )
            smoke_text = smoke.read_text(encoding="utf-8")
            formal_text = formal.read_text(encoding="utf-8")
            with self.subTest(tau=case["tau"]):
                for text in (smoke_text, formal_text):
                    self.assertIn('EXPECTED_BRANCH="{}"'.format(BRANCH), text)
                    self.assertIn(HARD_SHA, text)
                    self.assertIn("detached HEAD", text)
                    self.assertIn("dirty Git worktree", text)
                    self.assertIn("non-empty", text)
                    self.assertIn("python tools/run_experiment.py", text)
                    self.assertIn("--experiment-family {}".format(FAMILY), text)
                    self.assertIn('--feature-reference-commit "${PARENT_COMMIT}"', text)
                self.assertIn("--experiment-id {}".format(case["smoke_id"]), smoke_text)
                self.assertIn("--run-kind smoke", smoke_text)
                self.assertIn("SOLVER.MAX_EPOCHS 1", smoke_text)
                self.assertIn("SOLVER.CHECKPOINT_PERIOD 1", smoke_text)
                self.assertIn("SOLVER.EVAL_PERIOD 1", smoke_text)
                self.assertIn("--experiment-id {}".format(case["formal_id"]), formal_text)
                self.assertIn("--run-kind formal", formal_text)
                self.assertIn(
                    "--required-smoke-experiment-id {}".format(case["smoke_id"]),
                    formal_text,
                )
                self.assertNotIn("  SOLVER.", formal_text)
                self.assertNotIn("  MODEL.PCC_SOFTMIN_TAU", formal_text)
                self.assertIn(case["output"], formal_text)
            seen_ids.update((case["smoke_id"], case["formal_id"]))
            seen_outputs.add(case["output"])
        self.assertEqual(len(seen_ids), 4)
        self.assertEqual(len(seen_outputs), 2)


class FormalSmokeGateTest(unittest.TestCase):
    def _make_evidence(self, root):
        repo = root / "repo"
        repo.mkdir()
        _git(repo, "init")
        _git(repo, "config", "user.name", "Smoke Gate Test")
        _git(repo, "config", "user.email", "smoke-gate@example.invalid")
        (repo / "model.py").write_text("MODEL_VERSION = 1\n", encoding="utf-8")
        _git(repo, "add", "model.py")
        _git(repo, "commit", "-m", "implementation")
        smoke_commit = _git(repo, "rev-parse", "HEAD")

        case = SWEEP_CASES[0]
        config_path = REPO_ROOT / "configs" / case["config"]
        formal_configuration = _resolved_configuration(config_path)
        smoke_configuration = copy.deepcopy(formal_configuration)
        smoke_configuration["SOLVER"]["MAX_EPOCHS"] = 1
        smoke_configuration["SOLVER"]["CHECKPOINT_PERIOD"] = 1
        smoke_configuration["SOLVER"]["EVAL_PERIOD"] = 1
        smoke_configuration["OUTPUT_DIR"] = str(root / "external-output")

        run_dir = repo / "experiment_records" / "runs" / "matching-smoke"
        run_dir.mkdir(parents=True)
        source = run_dir / "config_source.yml"
        source.write_bytes(config_path.read_bytes())
        resolved = run_dir / "config_resolved.yml"
        resolved.write_text(
            yaml.safe_dump(smoke_configuration, sort_keys=True), encoding="utf-8"
        )
        console = run_dir / "console.log"
        console.write_text("real smoke console\n", encoding="utf-8")

        output = root / "external-output"
        output.mkdir()
        training_log = output / "log.txt"
        training_log.write_text("Epoch[1] Iteration[1]\n", encoding="utf-8")
        checkpoint = output / "resnet50_checkpoint_1.pt"
        checkpoint.write_bytes(b"real-one-epoch-checkpoint")
        checkpoint_sha = sha256_file(checkpoint)
        (run_dir / "checkpoint_manifest.tsv").write_text(
            "epoch\tglobal_iteration\tpath\tsha256\tselected\n"
            "1\t1\t{}\t{}\tTrue\n".format(checkpoint, checkpoint_sha),
            encoding="utf-8",
        )
        _write_json(run_dir / "metrics_summary.json", {
            "selected_epoch": 1,
            "checkpoint_sha256": checkpoint_sha,
        })
        feature = {
            "feature_reference_commit": HARD_SHA,
            "feature_reference_signature_sha256": "c" * 64,
            "current_feature_signature_sha256": "c" * 64,
            "feature_compatibility_status": "compatible",
        }
        manifest = {
            "run_id": "matching-smoke",
            "run_kind": "smoke",
            "experiment_id": case["smoke_id"],
            "experiment_family": FAMILY,
            "branch": BRANCH,
            "expected_branch": BRANCH,
            "commit_id": smoke_commit,
            "config_source_sha256": sha256_file(source),
            "config_resolved_sha256": sha256_file(resolved),
            "console_log_sha256": sha256_file(console),
            "training_log_path": str(training_log),
            "training_log_sha256": sha256_file(training_log),
            "seed": 42,
            "pcc_lambda": 0.1,
            "pcc_parts": 6,
            "pcc_mode": "soft_min",
            "alignment_temperature": case["tau"],
        }
        manifest.update(feature)
        _write_json(run_dir / "run_manifest.json", manifest)
        _write_json(run_dir / "run_status.json", {
            "status": "success",
            "phase": "complete",
            "training_exit_code": 0,
            "training_runtime_seconds": 1.0,
            "selected_epoch": 1,
        })
        _git(repo, "add", "experiment_records")
        _git(repo, "commit", "-m", "record smoke evidence")
        current_commit = _git(repo, "rev-parse", "HEAD")
        return (
            repo, config_path, formal_configuration, current_commit,
            case, feature,
        )

    def test_allows_only_pure_evidence_descendant_and_rejects_model_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (repo, config_path, configuration, current_commit,
             case, feature) = self._make_evidence(root)
            evidence = validate_formal_smoke_gate(
                repo_root=repo,
                records_root=repo / "experiment_records",
                formal_config_path=config_path,
                formal_configuration=configuration,
                current_commit=current_commit,
                expected_branch=BRANCH,
                expected_experiment_id=case["smoke_id"],
                expected_experiment_family=FAMILY,
                feature_compatibility=feature,
            )
            self.assertEqual(evidence["selected_epoch"], 1)
            self.assertEqual(evidence["smoke_commit"], _git(repo, "rev-parse", "HEAD~1"))
            self.assertTrue(all(
                path.startswith("experiment_records/")
                for path in evidence["post_smoke_changed_paths"]
            ))

            (repo / "model.py").write_text("MODEL_VERSION = 2\n", encoding="utf-8")
            _git(repo, "add", "model.py")
            _git(repo, "commit", "-m", "change model after smoke")
            changed_commit = _git(repo, "rev-parse", "HEAD")
            with self.assertRaisesRegex(SmokeGateError, "non-evidence files"):
                validate_formal_smoke_gate(
                    repo_root=repo,
                    records_root=repo / "experiment_records",
                    formal_config_path=config_path,
                    formal_configuration=configuration,
                    current_commit=changed_commit,
                    expected_branch=BRANCH,
                    expected_experiment_id=case["smoke_id"],
                    expected_experiment_family=FAMILY,
                    feature_compatibility=feature,
                )


if __name__ == "__main__":
    unittest.main()
