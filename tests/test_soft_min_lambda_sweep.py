# encoding: utf-8
"""Strict protocol, evidence-gate, and table tests for the tau=0.2 lambda sweep."""

from __future__ import absolute_import

import copy
import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from config import cfg
from tools.run_experiment import _effective_run_opts, _formal_gpu_lock
from tools.validate_c2l03_soft_min_alignment import parse_args
from utils.experiment_recording import (
    SCHEMA_VERSION,
    RUN_FIELDS,
    SOFT_ALIGNMENT_LAMBDA_FIELDS,
    SOFT_LAMBDA_SWEEP_FAMILY,
    TABLE_SCHEMAS,
    _lambda_table_eligible,
    _soft_alignment_lambda_table_eligible,
    config_protocol_signature,
    atomic_write_json,
    ensure_record_layout,
    git_implementation_signature,
    sha256_file,
    update_experiments_markdown,
    upsert_csv,
)
from utils.reproducibility import resolved_config_text
from utils.smoke_gate import SmokeGateError, validate_formal_smoke_gate


REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH = "exp/c2l03-soft-min-alignment-lambda-sweep-tau0p2"
PARENT_BRANCH = "exp/c2l03-soft-min-alignment-tau-sweep"
PARENT_SHA = "734e335034ac1cb935d9e63f0f00736c16821f13"
FEATURE_SHA = "6b46f2c3747124b97d59ed5cf987f33efb82282b"
BASE_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2l03_soft_min_alignment_tau0p2_autodl.yml"
)
CASES = (
    {
        "tag": "lambda0p05", "lambda": 0.05, "lp": "0P05",
        "output_tag": "lambda0p05",
    },
    {
        "tag": "lambda0p1", "lambda": 0.1, "lp": "0P1",
        "output_tag": "lambda0p1",
    },
    {
        "tag": "lambda0p3", "lambda": 0.3, "lp": "0P3",
        "output_tag": "lambda0p3",
    },
)


def _config_path(case):
    return REPO_ROOT / "configs" / (
        "softmax_triplet_c2l03_soft_min_alignment_tau0p2_"
        "{}_autodl.yml".format(case["tag"])
    )


def _resolved(path):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(path))
    local_cfg.freeze()
    return yaml.safe_load(resolved_config_text(local_cfg))


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


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class LambdaSweepProtocolTest(unittest.TestCase):
    def test_configs_have_structural_deep_diff_only(self):
        baseline = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        outputs = set()
        for case in CASES:
            candidate = yaml.safe_load(
                _config_path(case).read_text(encoding="utf-8")
            )
            expected_changes = {"OUTPUT_DIR"}
            if case["lambda"] != baseline["MODEL"]["PCC_LAMBDA"]:
                expected_changes.add("MODEL.PCC_LAMBDA")
            with self.subTest(alignment_lambda=case["lambda"]):
                self.assertEqual(
                    _changed_leaf_paths(baseline, candidate), expected_changes
                )
                self.assertEqual(candidate["MODEL"]["PCC_SOFTMIN_TAU"], 0.2)
                self.assertEqual(
                    candidate["MODEL"]["PCC_LAMBDA"], case["lambda"]
                )
                self.assertEqual(candidate["MODEL"]["PCC_PARTS"], 6)
                self.assertEqual(candidate["SEED"], 42)
                self.assertEqual(
                    candidate["MODEL"]["CROSS_CAMERA_POSITIVE_LAMBDA"], 0.3
                )
                self.assertEqual(candidate["SOLVER"]["MAX_EPOCHS"], 120)
                self.assertTrue(candidate["OUTPUT_DIR"].endswith(
                    "{}_seed42_market1501".format(case["output_tag"])
                ))
                outputs.add(candidate["OUTPUT_DIR"])
        self.assertEqual(len(outputs), 3)

    def test_validator_cli_is_parameterized_and_backward_compatible(self):
        defaults = parse_args([])
        self.assertEqual(defaults.expected_tau, 0.1)
        self.assertEqual(defaults.expected_lambda, 0.1)
        parsed = parse_args([
            "--config-file", str(_config_path(CASES[0])),
            "--expected-tau", "0.2", "--expected-lambda", "0.05",
        ])
        self.assertEqual(parsed.expected_tau, 0.2)
        self.assertEqual(parsed.expected_lambda, 0.05)

    def test_six_runners_are_unique_and_do_not_override_protocol_fields(self):
        ids = set()
        outputs = set()
        for case in CASES:
            smoke_id = "C2-L03-SOFTMIN-T0P2-LP{}-S42-SMOKE".format(
                case["lp"]
            )
            formal_id = smoke_id[:-6]
            smoke_path = REPO_ROOT / "scripts" / (
                "test_c2l03_soft_min_alignment_tau0p2_{}_autodl.sh".format(
                    case["tag"]
                )
            )
            formal_path = REPO_ROOT / "scripts" / (
                "train_c2l03_soft_min_alignment_tau0p2_{}_autodl.sh".format(
                    case["tag"]
                )
            )
            smoke = smoke_path.read_text(encoding="utf-8")
            formal = formal_path.read_text(encoding="utf-8")
            for run_kind, text in (("smoke", smoke), ("formal", formal)):
                self.assertIn('EXPECTED_BRANCH="{}"'.format(BRANCH), text)
                self.assertIn('PARENT_COMMIT="{}"'.format(PARENT_SHA), text)
                self.assertIn(
                    'FEATURE_REFERENCE_COMMIT="{}"'.format(FEATURE_SHA), text
                )
                self.assertIn("python tools/run_experiment.py", text)
                self.assertNotIn("python tools/train.py", text)
                self.assertIn("--run-kind {}".format(run_kind), text)
                self.assertIn(
                    "--experiment-family {}".format(
                        SOFT_LAMBDA_SWEEP_FAMILY
                    ), text,
                )
                invocation = text.split("python tools/run_experiment.py", 1)[1]
                for forbidden in (
                        "MODEL.PCC_SOFTMIN_TAU", "MODEL.PCC_LAMBDA", "SEED",
                        "SOLVER.MAX_EPOCHS", "OUTPUT_DIR \""):
                    self.assertNotIn(forbidden, invocation)
                self.assertIn("non-empty", text)
            self.assertIn("--experiment-id {}".format(smoke_id), smoke)
            self.assertIn("--experiment-id {}".format(formal_id), formal)
            self.assertIn(
                "--required-smoke-experiment-id {}".format(smoke_id), formal
            )
            ids.update((smoke_id, formal_id))
            outputs.add(yaml.safe_load(
                _config_path(case).read_text(encoding="utf-8")
            )["OUTPUT_DIR"])
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(outputs), 3)

    def test_unified_runner_derives_only_standard_smoke_overrides(self):
        formal_output = "/tmp/formal"
        self.assertEqual(
            _effective_run_opts("smoke", formal_output),
            [
                "SOLVER.MAX_EPOCHS", "1",
                "SOLVER.CHECKPOINT_PERIOD", "1",
                "SOLVER.EVAL_PERIOD", "1",
                "OUTPUT_DIR", "/tmp/formal_smoke",
            ],
        )
        self.assertEqual(_effective_run_opts("formal", formal_output), [])

    def test_formal_gpu_lock_rejects_concurrent_formal_but_not_smoke(self):
        with _formal_gpu_lock("formal"):
            with self.assertRaisesRegex(RuntimeError, "GPU lock"):
                with _formal_gpu_lock("formal"):
                    pass
            with _formal_gpu_lock("smoke") as lock:
                self.assertEqual(lock, "not_recorded")

    def test_parent_algorithms_model_descriptor_data_and_eval_are_unchanged(self):
        protected = (
            "layers", "modeling", "data", "engine", "solver",
            "tools/train.py", "tools/test.py",
        )
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", PARENT_SHA, "--"] + list(protected),
            cwd=str(REPO_ROOT), text=True,
        ).strip()
        self.assertEqual(changed, "")


class StrictLambdaSmokeGateTest(unittest.TestCase):
    def _make_evidence(self, root, case):
        repo = root / "repo"
        repo.mkdir()
        _git(repo, "init")
        _git(repo, "config", "user.name", "Lambda Gate Test")
        _git(repo, "config", "user.email", "lambda-gate@example.invalid")
        (repo / "implementation.py").write_text(
            "IMPLEMENTATION_VERSION = 1\n", encoding="utf-8"
        )
        _git(repo, "add", "implementation.py")
        _git(repo, "commit", "-m", "implementation")
        smoke_commit = _git(repo, "rev-parse", "HEAD")
        implementation_signature = git_implementation_signature(
            repo, smoke_commit
        )

        formal_configuration = _resolved(_config_path(case))
        formal_configuration["OUTPUT_DIR"] = str(root / "formal-output")
        formal_config_path = root / "formal.yml"
        formal_config_path.write_text(
            yaml.safe_dump(formal_configuration, sort_keys=True),
            encoding="utf-8",
        )
        smoke_configuration = copy.deepcopy(formal_configuration)
        smoke_configuration["SOLVER"]["MAX_EPOCHS"] = 1
        smoke_configuration["SOLVER"]["CHECKPOINT_PERIOD"] = 1
        smoke_configuration["SOLVER"]["EVAL_PERIOD"] = 1
        smoke_configuration["OUTPUT_DIR"] = "{}_smoke".format(
            formal_configuration["OUTPUT_DIR"]
        )

        run_id = "strict-smoke-{}".format(case["tag"])
        run_dir = repo / "experiment_records" / "runs" / run_id
        run_dir.mkdir(parents=True)
        source = run_dir / "config_source.yml"
        source.write_bytes(formal_config_path.read_bytes())
        resolved = run_dir / "config_resolved.yml"
        resolved.write_text(
            yaml.safe_dump(smoke_configuration, sort_keys=True),
            encoding="utf-8",
        )
        console = run_dir / "console.log"
        console.write_text("combined stdout/stderr\n", encoding="utf-8")

        output = Path(smoke_configuration["OUTPUT_DIR"])
        output.mkdir()
        training_log = output / "log.txt"
        training_log.write_text("real epoch 1 training log\n", encoding="utf-8")
        checkpoint = output / "resnet50_checkpoint_1.pt"
        checkpoint.write_bytes(b"real-one-epoch-checkpoint")
        checkpoint_sha = sha256_file(checkpoint)
        checkpoint_manifest = run_dir / "checkpoint_manifest.tsv"
        checkpoint_manifest.write_text(
            "epoch\tglobal_iteration\tpath\tsha256\tselected\n"
            "1\t1\t{}\t{}\tTrue\n".format(checkpoint, checkpoint_sha),
            encoding="utf-8",
        )
        _write_json(run_dir / "metrics_summary.json", {
            "best_epoch": 1,
            "selected_epoch": 1,
            "checkpoint_sha256": checkpoint_sha,
        })
        feature = {
            "feature_reference_commit": FEATURE_SHA,
            "feature_reference_signature_sha256": "c" * 64,
            "current_feature_signature_sha256": "c" * 64,
            "feature_compatibility_status": "compatible",
        }
        _write_json(run_dir / "feature_compatibility.json", feature)
        dataset_manifest = {
            "dataset": "market1501",
            "sampler": "softmax_triplet",
            "batch_size": 64,
            "num_instance": 4,
            "dataset_manifest_sha256": "d" * 64,
        }
        _write_json(run_dir / "dataset_manifest.json", dataset_manifest)
        _write_json(run_dir / "environment.json", {
            "gpus": [{"name": "Synthetic GPU"}],
            "git_branch": BRANCH,
            "git_commit": smoke_commit,
        })
        (run_dir / "environment_packages.txt").write_text(
            "torch==synthetic\n", encoding="utf-8"
        )
        _write_json(run_dir / "reproducibility.json", {
            "status": "complete", "applied_seed": 42,
            "cudnn_deterministic": True, "cudnn_benchmark": False,
        })
        _write_json(run_dir / "model_manifest.json", {
            "implementation_signature_sha256": implementation_signature,
        })
        status = {
            "status": "success", "phase": "complete",
            "training_exit_code": 0, "training_runtime_seconds": 1.5,
            "selected_epoch": 1,
        }
        _write_json(run_dir / "run_status.json", status)

        protocol_signature = config_protocol_signature(formal_configuration)
        smoke_id = "C2-L03-SOFTMIN-T0P2-LP{}-S42-SMOKE".format(case["lp"])
        manifest = {
            "run_id": run_id,
            "run_kind": "smoke",
            "experiment_id": smoke_id,
            "experiment_family": SOFT_LAMBDA_SWEEP_FAMILY,
            "branch": BRANCH,
            "expected_branch": BRANCH,
            "commit_id": smoke_commit,
            "parent_branch": PARENT_BRANCH,
            "parent_commit": PARENT_SHA,
            "merge_base": PARENT_SHA,
            "git_preflight_clean": True,
            "git_status_preflight": [],
            "config_source_sha256": sha256_file(source),
            "config_source_size_bytes": source.stat().st_size,
            "config_resolved_sha256": sha256_file(resolved),
            "config_resolved_size_bytes": resolved.stat().st_size,
            "console_log_sha256": sha256_file(console),
            "training_log_path": str(training_log),
            "training_log_sha256": sha256_file(training_log),
            "seed": 42,
            "cross_camera_positive_lambda": 0.3,
            "pcc_lambda": case["lambda"],
            "pcc_parts": 6,
            "pcc_mode": "soft_min",
            "alignment_temperature": 0.2,
            "protocol_signature_sha256": protocol_signature,
            "implementation_signature_sha256": implementation_signature,
            "dataset_manifest_sha256": dataset_manifest[
                "dataset_manifest_sha256"
            ],
        }
        manifest.update(feature)

        artifact_paths = {
            "source_config": source,
            "resolved_config": resolved,
            "console_log": console,
            "dataset_manifest": run_dir / "dataset_manifest.json",
            "environment": run_dir / "environment.json",
            "environment_packages": run_dir / "environment_packages.txt",
            "feature_compatibility": run_dir / "feature_compatibility.json",
            "metrics_summary": run_dir / "metrics_summary.json",
            "model_manifest": run_dir / "model_manifest.json",
            "reproducibility": run_dir / "reproducibility.json",
            "checkpoint_manifest": checkpoint_manifest,
            "run_status": run_dir / "run_status.json",
            "training_log": training_log,
            "selected_checkpoint": checkpoint,
        }
        artifact_manifest = run_dir / "artifact_hashes.tsv"
        with artifact_manifest.open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("artifact_type", "path", "size_bytes", "sha256"),
                delimiter="\t", lineterminator="\n",
            )
            writer.writeheader()
            for artifact_type, path in artifact_paths.items():
                stored_path = (
                    path.name if path.parent == run_dir else str(path)
                )
                writer.writerow({
                    "artifact_type": artifact_type,
                    "path": stored_path,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        manifest["artifact_manifest_size_bytes"] = artifact_manifest.stat().st_size
        manifest["artifact_manifest_sha256"] = sha256_file(artifact_manifest)
        _write_json(run_dir / "run_manifest.json", manifest)

        _git(repo, "add", "experiment_records")
        _git(repo, "commit", "-m", "record smoke evidence")
        current_commit = _git(repo, "rev-parse", "HEAD")
        expected = {
            "repo_root": repo,
            "records_root": repo / "experiment_records",
            "formal_config_path": formal_config_path,
            "formal_configuration": formal_configuration,
            "current_commit": current_commit,
            "expected_branch": BRANCH,
            "expected_experiment_id": smoke_id,
            "expected_experiment_family": SOFT_LAMBDA_SWEEP_FAMILY,
            "feature_compatibility": feature,
            "expected_parent_branch": PARENT_BRANCH,
            "expected_parent_commit": PARENT_SHA,
            "expected_merge_base": PARENT_SHA,
            "expected_protocol_signature": protocol_signature,
            "expected_implementation_signature": implementation_signature,
            "expected_dataset_manifest": dataset_manifest,
        }
        return run_dir, expected

    def test_matching_complete_smoke_unlocks_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            _, expected = self._make_evidence(Path(directory), CASES[0])
            evidence = validate_formal_smoke_gate(**expected)
            self.assertEqual(evidence["selected_epoch"], 1)
            self.assertEqual(
                evidence["protocol_signature_sha256"],
                expected["expected_protocol_signature"],
            )
            self.assertEqual(
                evidence["implementation_signature_sha256"],
                expected["expected_implementation_signature"],
            )

    def test_cross_lambda_smoke_cannot_unlock_other_formals(self):
        with tempfile.TemporaryDirectory() as directory:
            _, expected = self._make_evidence(Path(directory), CASES[0])
            for other in CASES[1:]:
                other_config = _resolved(_config_path(other))
                other_config["OUTPUT_DIR"] = str(
                    Path(directory) / "formal-other-{}".format(other["tag"])
                )
                other_path = Path(directory) / "{}.yml".format(other["tag"])
                other_path.write_text(
                    yaml.safe_dump(other_config, sort_keys=True),
                    encoding="utf-8",
                )
                attempt = dict(expected)
                attempt.update({
                    "formal_config_path": other_path,
                    "formal_configuration": other_config,
                    "expected_experiment_id": (
                        "C2-L03-SOFTMIN-T0P2-LP{}-S42-SMOKE".format(
                            other["lp"]
                        )
                    ),
                    "expected_protocol_signature": config_protocol_signature(
                        other_config
                    ),
                })
                with self.subTest(alignment_lambda=other["lambda"]):
                    with self.assertRaisesRegex(
                            SmokeGateError, "no smoke evidence"):
                        validate_formal_smoke_gate(**attempt)

    def test_failed_incomplete_interrupted_or_missing_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, expected = self._make_evidence(Path(directory), CASES[1])
            status_path = run_dir / "run_status.json"
            original = json.loads(status_path.read_text(encoding="utf-8"))
            for status in ("failed", "incomplete", "interrupted"):
                changed = dict(original)
                changed["status"] = status
                _write_json(status_path, changed)
                with self.subTest(status=status):
                    with self.assertRaises(SmokeGateError):
                        validate_formal_smoke_gate(**expected)
            _write_json(status_path, original)
            (run_dir / "environment.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(SmokeGateError):
                validate_formal_smoke_gate(**expected)

    def test_hash_parent_clean_and_signature_tampering_is_rejected(self):
        fields = (
            ("parent_commit", "e" * 40),
            ("git_preflight_clean", False),
            ("protocol_signature_sha256", "e" * 64),
            ("implementation_signature_sha256", "e" * 64),
            ("dataset_manifest_sha256", "e" * 64),
        )
        for field, value in fields:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    run_dir, expected = self._make_evidence(
                        Path(directory), CASES[2]
                    )
                    manifest_path = run_dir / "run_manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest[field] = value
                    _write_json(manifest_path, manifest)
                    with self.assertRaises(SmokeGateError):
                        validate_formal_smoke_gate(**expected)


class LambdaTableSchemaTest(unittest.TestCase):
    def test_view_schema_is_explicit_and_uses_pcc_lambda(self):
        self.assertEqual(SCHEMA_VERSION, 4)
        self.assertEqual(
            TABLE_SCHEMAS["soft_alignment_lambda_sensitivity"],
            SOFT_ALIGNMENT_LAMBDA_FIELDS,
        )
        for field in (
                "schema_version", "run_id", "experiment_id", "run_kind",
                "status", "alignment_mode", "alignment_temperature",
                "pcc_lambda", "alignment_lambda", "parts", "seed",
                "rank1", "rank5", "rank10", "map", "best_epoch",
                "runtime", "checkpoint", "checkpoint_sha256", "commit",
                "output_dir"):
            self.assertIn(field, SOFT_ALIGNMENT_LAMBDA_FIELDS)
        self.assertNotIn("lambda", SOFT_ALIGNMENT_LAMBDA_FIELDS)

    def test_new_view_is_header_only_before_real_experiments(self):
        csv_path = REPO_ROOT / "experiment_records" / "tables" / (
            "soft_alignment_lambda_sensitivity.csv"
        )
        markdown_path = csv_path.with_suffix(".md")
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows, [])
        self.assertEqual(
            markdown_path.read_text(encoding="utf-8").count("\n"), 2
        )

    def test_markdown_registry_preserves_authoritative_machine_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = ensure_record_layout(root / "experiment_records")
            experiments = root / "EXPERIMENTS.md"
            experiments.write_text("# Experiments\n", encoding="utf-8")
            row = {field: "not_recorded" for field in RUN_FIELDS}
            row.update({
                "schema_version": SCHEMA_VERSION,
                "run_id": "authoritative-run",
                "experiment_id": "AUTHORITATIVE",
                "run_kind": "formal",
                "status": "success",
                "source_config_path": "/recorded/source.yml",
            })
            upsert_csv(records / "runs.csv", RUN_FIELDS, row)
            run_dir = records / "runs" / "authoritative-run"
            run_dir.mkdir(parents=True)
            atomic_write_json(run_dir / "run_manifest.json", {
                "run_id": "authoritative-run",
                "config_file": "/host-local/wrong.yml",
            })
            atomic_write_json(run_dir / "run_status.json", {
                "status": "success",
            })
            update_experiments_markdown(experiments, records)
            generated = experiments.read_text(encoding="utf-8")
            self.assertIn("/recorded/source.yml", generated)
            self.assertNotIn("/host-local/wrong.yml", generated)

    def test_view_filter_accepts_only_declared_formal_soft_min_matrix(self):
        manifest = {
            "experiment_family": SOFT_LAMBDA_SWEEP_FAMILY,
            "experiment_id": "C2-L03-SOFTMIN-T0P2-LP0P05-S42",
            "run_kind": "formal",
            "alignment_mode": "soft_min",
            "alignment_temperature": 0.2,
            "pcc_lambda": 0.05,
            "lambda": 0.3,
        }
        self.assertTrue(_soft_alignment_lambda_table_eligible(manifest))
        self.assertFalse(_lambda_table_eligible(manifest))
        for field, invalid in (
                ("run_kind", "smoke"),
                ("status", "failed"),
                ("alignment_mode", "hard_shortest_path"),
                ("alignment_temperature", 0.1),
                ("pcc_lambda", 0.2),
                ("experiment_family", "another_family")):
            candidate = dict(manifest)
            candidate[field] = invalid
            with self.subTest(field=field):
                self.assertFalse(
                    _soft_alignment_lambda_table_eligible(candidate)
                )


if __name__ == "__main__":
    unittest.main()
