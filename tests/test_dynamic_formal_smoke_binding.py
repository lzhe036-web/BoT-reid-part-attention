import csv
import hashlib
import io
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import utils.dynamic_experiment_registry as registry
from utils.dynamic_gating_evidence import (
    DYNAMIC_GATING_SAMPLE_FIELDS,
    DYNAMIC_GATING_SELECTION_RULE,
    GatingEpochAccumulator,
)
from utils.experiment_schema import SCHEMA_VERSION
from utils.multigranularity_signatures import (
    STATIC_BASELINE_BRANCH,
    STATIC_BASELINE_SHA,
)


SMOKE_ID = "C2-L03-MGDG-T1-S42-SMOKE"


def feature_evidence():
    return {
        "feature_reference_commit": STATIC_BASELINE_SHA,
        "feature_reference_signature_sha256": "b" * 64,
        "current_feature_signature_sha256": "b" * 64,
        "feature_compatibility_status": "compatible",
        "current_commit": "e" * 40,
        "mismatched_components": [],
        "components": {},
        "fusion_gating_signature": {
            "reference_sha256": "c" * 64,
            "current_sha256": "d" * 64,
            "status": "expected_experiment_difference",
        },
    }


def protocol(max_epochs, output_dir):
    return {
        "SEED": 42,
        "MODEL": {
            "MULTI_GRANULARITY_PART_SCALES": [2, 4, 6],
            "MULTI_GRANULARITY_GATING_TAU": 1.0,
            "MULTI_GRANULARITY_GATING_INPUT": "global",
            "MULTI_GRANULARITY_GATING_NORMALIZATION": "scaled_softmax",
        },
        "SOLVER": {
            "MAX_EPOCHS": max_epochs,
            "CHECKPOINT_PERIOD": 1 if max_epochs == 1 else 40,
            "EVAL_PERIOD": 1 if max_epochs == 1 else 40,
        },
        "OUTPUT_DIR": str(output_dir),
    }


class FormalSmokeFixture(object):
    def __init__(self, root):
        self.root = Path(root)
        self.records = self.root / "experiment_records"
        self.experiments = self.root / "EXPERIMENTS.md"
        self.experiments.write_text("# Experiments\n", encoding="utf-8")
        self.output = self.root / "smoke-output"
        self.source = self.root / "formal.yml"
        self.formal_protocol = protocol(120, self.root / "formal-output")
        self.source.write_text(
            yaml.safe_dump(self.formal_protocol, sort_keys=True), encoding="utf-8"
        )
        self.smoke_protocol = protocol(1, self.output)
        self.lineage = {
            "branch": "exp/c2-l03-multi-granularity-dynamic-gating",
            "commit": "e" * 40,
            "parent_branch": STATIC_BASELINE_BRANCH,
            "parent_commit": STATIC_BASELINE_SHA,
            "merge_base": STATIC_BASELINE_SHA,
        }
        self.feature = feature_evidence()
        protocol_sha = registry.candidate_protocol_signature(self.smoke_protocol)
        self.run_dir, _manifest = registry.initialize_dynamic_run(
            self.records, self.experiments, SMOKE_ID, "smoke", self.source,
            yaml.safe_dump(self.smoke_protocol, sort_keys=True), self.output,
            self.lineage, self.feature, ["python", "train.py"],
            candidate_protocol_signature_sha256=protocol_sha,
            implementation_signature_sha256="f" * 64,
        )
        self.expected_samples = self._write_gating_evidence()
        self._complete_success()

    def _write_gating_evidence(self):
        checkpoint = self.run_dir / "selected.pt"
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_sha = registry.sha256_file(checkpoint)
        keys = ["candidate-a", "candidate-b"]
        keys.sort(key=lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest())
        probabilities = ((0.2, 0.3, 0.5), (0.5, 0.3, 0.2))
        rows, expected = [], []
        for index, key in enumerate(keys):
            split = "query" if index == 0 else "gallery"
            pid, camid = index + 1, index + 2
            p = probabilities[index]
            entropy = -sum(value * math.log(value) for value in p)
            dominant = (2, 4, 6)[max(range(3), key=lambda item: p[item])]
            rows.append({
                "stable_sample_key": key, "dataset_split": split,
                "pid": pid, "camid": camid,
                "p2": p[0], "p4": p[1], "p6": p[2],
                "w2": 3 * p[0], "w4": 3 * p[1], "w6": 3 * p[2],
                "entropy": entropy, "dominant_k": dominant,
                "checkpoint_sha256": checkpoint_sha,
            })
            expected.append((key, split, "unused.jpg", pid, camid))
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer, fieldnames=DYNAMIC_GATING_SAMPLE_FIELDS,
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        samples = self.run_dir / "gating_samples.tsv"
        samples.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")
        accumulator = GatingEpochAccumulator(1.0)
        accumulator.update(probabilities)
        self.statistics = accumulator.summary()
        samples_evidence = {
            "path": str(samples.resolve()), "size_bytes": samples.stat().st_size,
            "sha256": registry.sha256_file(samples),
            "source_checkpoint_sha256": checkpoint_sha,
            "selection_rule": DYNAMIC_GATING_SELECTION_RULE,
        }
        summary = self.run_dir / "dynamic_gating_summary.json"
        registry.atomic_write_json(summary, {
            "schema_version": SCHEMA_VERSION,
            "source_checkpoint_sha256": checkpoint_sha,
            "selection_rule": DYNAMIC_GATING_SELECTION_RULE,
            "selected_sample_count": len(rows),
            "training_epoch_statistics": self.statistics,
            "deterministic_sample_statistics": self.statistics,
            "gating_samples": samples_evidence,
        })
        self.checkpoint = checkpoint
        self.samples = samples
        self.summary = summary
        return expected

    def _evidence(self, path):
        return registry._file_evidence(path)

    def _complete_success(self):
        manifest = registry.read_json(self.run_dir / "run_manifest.json")
        self.output.mkdir(parents=True, exist_ok=True)
        files = {
            "console_log": self.run_dir / "console.log",
            "training_log": self.output / "log.txt",
            "reproducibility": self.output / "reproducibility.json",
            "environment": self.run_dir / "environment.json",
            "dataset_manifest": self.run_dir / "dataset_manifest.json",
            "model_manifest": self.run_dir / "model_manifest.json",
        }
        for name, path in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n".format(name), encoding="utf-8")
        registry.atomic_write_json(files["dataset_manifest"], {"fixture": True})
        checkpoint_manifest = self.run_dir / "checkpoint_manifest.tsv"
        checkpoint_manifest.write_text(
            "epoch\tglobal_iteration\trelative_path\tfile_size\tsha256\tselected\n"
            "1\t17\tselected.pt\t{}\t{}\ttrue\n".format(
                self.checkpoint.stat().st_size,
                registry.sha256_file(self.checkpoint),
            ), encoding="utf-8", newline="\n",
        )
        resolved = self.run_dir / "config_resolved.yml"
        manifest.update({
            "status": "success", "gating_statistics": self.statistics,
            "metrics": {"selected_epoch": 1, "best_epoch": 1},
            "console_log": self._evidence(files["console_log"]),
            "training_log": self._evidence(files["training_log"]),
            "resolved_config": self._evidence(resolved),
            "reproducibility": self._evidence(files["reproducibility"]),
            "environment": self._evidence(files["environment"]),
            "dataset_manifest": self._evidence(files["dataset_manifest"]),
            "model_manifest": self._evidence(files["model_manifest"]),
            "checkpoint_manifest": self._evidence(checkpoint_manifest),
            "selected_checkpoint": self._evidence(self.checkpoint),
            "dynamic_gating_summary": self._evidence(self.summary),
            "gating_samples": self._evidence(self.samples),
        })
        for name in (
                "console_log", "training_log", "reproducibility", "environment",
                "dataset_manifest", "model_manifest", "checkpoint_manifest",
                "selected_checkpoint", "dynamic_gating_summary", "gating_samples",
                "resolved_config"):
            manifest.setdefault("artifacts", {})[name] = dict(manifest[name])
        manifest["artifacts"]["dynamic_gating_summary"].update({
            "source_checkpoint_sha256": manifest["selected_checkpoint"]["sha256"],
            "selection_rule": DYNAMIC_GATING_SELECTION_RULE,
        })
        manifest["artifacts"]["gating_samples"].update({
            "source_checkpoint_sha256": manifest["selected_checkpoint"]["sha256"],
            "selection_rule": DYNAMIC_GATING_SELECTION_RULE,
        })
        registry._refresh_partial_artifact_manifest(self.run_dir, manifest)
        registry.atomic_write_json(self.run_dir / "run_manifest.json", manifest)
        status = registry.read_json(self.run_dir / "run_status.json")
        status["status"] = "success"
        registry.atomic_write_json(self.run_dir / "run_status.json", status)
        registry.register_dynamic_run_state(self.run_dir)

    def mutate_manifest(self, callback):
        manifest = registry.read_json(self.run_dir / "run_manifest.json")
        callback(manifest)
        registry._refresh_partial_artifact_manifest(self.run_dir, manifest)
        registry.atomic_write_json(self.run_dir / "run_manifest.json", manifest)
        registry.register_dynamic_run_state(self.run_dir)

    def gate(self, current_protocol=None, current_source=None, current_feature=None):
        protocol_value = current_protocol or self.formal_protocol
        with mock.patch.object(
                registry, "implementation_signature", return_value="f" * 64), \
                mock.patch.object(
                    registry, "validate_smoke_commit_lineage", return_value=[]):
            return registry.validate_recorded_smoke_for_formal(
                self.records, self.root, SMOKE_ID, self.lineage,
                current_source or self.source, protocol_value,
                current_feature or self.feature,
                current_protocol_signature_sha256=registry.candidate_protocol_signature(
                    protocol_value
                ),
                current_implementation_signature_sha256="f" * 64,
                selection_resolver=lambda _cfg: list(self.expected_samples),
                dataset_validator=lambda _cfg, _manifest: None,
            )


class FormalSmokeBindingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = FormalSmokeFixture(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def assert_gate_fails(self, **kwargs):
        with self.assertRaises(registry.DynamicExperimentEvidenceError):
            self.fixture.gate(**kwargs)

    def test_same_candidate_smoke_unlocks_formal(self):
        result = self.fixture.gate()
        self.assertEqual(result["manifest"]["run_kind"], "smoke")

    def test_source_config_sha_change_fails(self):
        changed = self.fixture.root / "changed.yml"
        changed.write_text(self.fixture.source.read_text(encoding="utf-8") + "# changed\n")
        self.assert_gate_fails(current_source=changed)

    def test_normalized_protocol_change_fails(self):
        changed = json.loads(json.dumps(self.fixture.formal_protocol))
        changed["MODEL"]["MULTI_GRANULARITY_GATING_INPUT"] = "changed"
        self.assert_gate_fails(current_protocol=changed)

    def test_shared_feature_signature_change_fails(self):
        changed = json.loads(json.dumps(self.fixture.feature))
        changed["current_feature_signature_sha256"] = "1" * 64
        self.assert_gate_fails(current_feature=changed)

    def test_gating_signature_change_fails(self):
        changed = json.loads(json.dumps(self.fixture.feature))
        changed["fusion_gating_signature"]["current_sha256"] = "1" * 64
        self.assert_gate_fails(current_feature=changed)

    def test_parent_sha_change_fails(self):
        self.fixture.mutate_manifest(
            lambda manifest: manifest.update(parent_commit="1" * 40)
        )
        self.assert_gate_fails()

    def test_seed_tau_and_k_changes_fail(self):
        mutations = (
            lambda manifest: manifest.update(seed=7),
            lambda manifest: manifest.update(gating_temperature=2.0),
            lambda manifest: manifest.update(scale_order="2,6,4"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = FormalSmokeFixture(directory)
                    fixture.mutate_manifest(mutation)
                    with self.assertRaises(registry.DynamicExperimentEvidenceError):
                        fixture.gate()

    def test_any_artifact_hash_tamper_fails(self):
        path = self.fixture.run_dir / "console.log"
        path.write_text("tampered\n", encoding="utf-8")
        self.assert_gate_fails()

    def test_non_one_epoch_smoke_fails(self):
        resolved = self.fixture.run_dir / "config_resolved.yml"
        changed = protocol(2, self.fixture.output)
        resolved.write_text(yaml.safe_dump(changed), encoding="utf-8")
        self.fixture.mutate_manifest(
            lambda manifest: (
                manifest.update(resolved_config=registry._file_evidence(resolved)),
                manifest["artifacts"].update(
                    resolved_config_snapshot=registry._file_evidence(resolved)
                ),
            )
        )
        self.assert_gate_fails()


class EvidenceOnlyCommitLineageTest(unittest.TestCase):
    def test_source_change_fails_but_evidence_only_descendant_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.check_call(["git", "init", str(repo)], stdout=subprocess.DEVNULL)
            subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Test"])
            subprocess.check_call(["git", "-C", str(repo), "config", "user.email", "test@example.com"])
            (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            (repo / "EXPERIMENTS.md").write_text("# evidence\n", encoding="utf-8")
            subprocess.check_call(["git", "-C", str(repo), "add", "."])
            subprocess.check_call(["git", "-C", str(repo), "commit", "-m", "base"], stdout=subprocess.DEVNULL)
            base = registry._git(repo, ["rev-parse", "HEAD"])
            (repo / "EXPERIMENTS.md").write_text("# evidence updated\n", encoding="utf-8")
            subprocess.check_call(["git", "-C", str(repo), "add", "EXPERIMENTS.md"])
            subprocess.check_call(["git", "-C", str(repo), "commit", "-m", "evidence"], stdout=subprocess.DEVNULL)
            evidence = registry._git(repo, ["rev-parse", "HEAD"])
            self.assertEqual(
                registry.implementation_signature(repo, base),
                registry.implementation_signature(repo, evidence),
            )
            self.assertEqual(
                registry.validate_smoke_commit_lineage(repo, base, evidence),
                ["EXPERIMENTS.md"],
            )
            (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
            subprocess.check_call(["git", "-C", str(repo), "add", "source.py"])
            subprocess.check_call(["git", "-C", str(repo), "commit", "-m", "source"], stdout=subprocess.DEVNULL)
            current = registry._git(repo, ["rev-parse", "HEAD"])
            self.assertNotEqual(
                registry.implementation_signature(repo, base),
                registry.implementation_signature(repo, current),
            )
            with self.assertRaises(registry.DynamicExperimentEvidenceError):
                registry.validate_smoke_commit_lineage(repo, base, current)


if __name__ == "__main__":
    unittest.main()
