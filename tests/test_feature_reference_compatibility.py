# encoding: utf-8
"""Revision-bound multigranular feature compatibility evidence tests."""

from __future__ import absolute_import

import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from utils.multigranular_signature import (
    FeatureCompatibilityError,
    build_feature_compatibility_evidence,
    git_show_source,
)
from utils.experiment_recording import (
    SCHEMA_VERSION,
    atomic_write_json,
    finalize_run,
    read_json,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HARD_SHA = "6b46f2c3747124b97d59ed5cf987f33efb82282b"
SOFT_SHA = "2d80106350a8d344f1d09d4bbd9cfa8c24d6f5e5"
HARD_CONFIG_PATH = (
    "configs/softmax_triplet_c2l03_hard_shortest_path_alignment_autodl.yml"
)
SOFT_CONFIG_PATH = "configs/softmax_triplet_c2l03_soft_min_alignment_autodl.yml"


BASE_SOURCES = {
    "modeling/baseline.py": """
class Baseline:
    def forward(self, x):
        global_feat = self.gap(x)
        return global_feat

class PartAttentionHead:
    def __init__(self, channels, num_parts=6):
        self.channels = channels
        self.num_parts = num_parts

    def forward(self, feature_map):
        return feature_map
""",
    "layers/part_correspondence_consistency.py": """
def horizontal_part_bounds(height, num_parts):
    return [(i * height // num_parts, (i + 1) * height // num_parts) for i in range(num_parts)]

def build_local_part_descriptors(feature_map, num_parts=6):
    bounds = horizontal_part_bounds(feature_map.size(2), num_parts)
    return adaptive_average_pool(feature_map, bounds)

def build_cross_camera_positive_pairs(pids, camids):
    return same_pid_different_camera_upper_triangle(pids, camids)

def select_pair_local_features(local_features, pair_indices):
    return local_features[pair_indices[:, 0]], local_features[pair_indices[:, 1]]

def pairwise_local_distance_matrix(local_a, local_b):
    return euclidean_l2(local_a, local_b)
""",
    "modeling/backbones/resnet.py": """
class ResNet:
    def forward(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))
""",
}


PARAMETER_PAYLOAD = {
    "parameters": [
        {"name": "base.conv1.weight", "shape": [64, 3, 7, 7]},
        {"name": "part_attention_head.part_logits.weight", "shape": [6, 2048, 1, 1]},
        {"name": "bottleneck.weight", "shape": [2048]},
    ],
    "descriptor_dim": 2048,
}


def parameter_provider(repo_root, revision, configuration):
    return PARAMETER_PAYLOAD


class FeatureReferenceCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hard = yaml.safe_load(git_show_source(
            REPO_ROOT, HARD_SHA, HARD_CONFIG_PATH
        ))
        cls.soft = yaml.safe_load(git_show_source(
            REPO_ROOT, SOFT_SHA, SOFT_CONFIG_PATH
        ))

    def evidence(self, mutations=None, suffix=""):
        mutations = mutations or {}

        def provider(repo_root, revision, path):
            source = BASE_SOURCES[path]
            if str(revision).lower() == SOFT_SHA and path in mutations:
                old, new = mutations[path]
                source = source.replace(old, new)
            if str(revision).lower() == SOFT_SHA and suffix:
                if path == "layers/part_correspondence_consistency.py":
                    source += suffix
            return source

        return build_feature_compatibility_evidence(
            REPO_ROOT, HARD_SHA, SOFT_SHA, self.hard, self.soft,
            source_provider=provider,
            parameter_schema_provider=parameter_provider,
        )

    def test_real_soft_matches_fixed_hard_reference(self):
        evidence = build_feature_compatibility_evidence(
            REPO_ROOT, HARD_SHA, SOFT_SHA, self.hard, self.soft
        )
        self.assertEqual(evidence["feature_reference_commit"], HARD_SHA)
        self.assertEqual(evidence["feature_compatibility_status"], "compatible")
        self.assertEqual(evidence["mismatched_components"], [])
        self.assertEqual(
            evidence["feature_reference_signature_sha256"],
            evidence["current_feature_signature_sha256"],
        )

    def test_git_show_reads_requested_revision_without_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.check_call(["git", "init"], cwd=str(repo),
                                  stdout=subprocess.DEVNULL)
            subprocess.check_call(
                ["git", "config", "user.email", "fixture@example.invalid"],
                cwd=str(repo),
            )
            subprocess.check_call(
                ["git", "config", "user.name", "Fixture"], cwd=str(repo)
            )
            path = repo / "feature.py"
            path.write_text("VALUE = 'parent'\n", encoding="utf-8")
            subprocess.check_call(["git", "add", "feature.py"], cwd=str(repo))
            subprocess.check_call(
                ["git", "commit", "-m", "parent"], cwd=str(repo),
                stdout=subprocess.DEVNULL,
            )
            parent = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
            ).strip()
            path.write_text("VALUE = 'current'\n", encoding="utf-8")
            self.assertEqual(
                git_show_source(repo, parent, "feature.py"),
                "VALUE = 'parent'\n",
            )
            self.assertEqual(path.read_text(encoding="utf-8"), "VALUE = 'current'\n")

    def test_mode_and_tau_changes_do_not_change_feature_signature(self):
        evidence = self.evidence()
        self.assertEqual(evidence["feature_compatibility_status"], "compatible")

    def test_shared_component_changes_report_exact_component(self):
        cases = (
            ("modeling/baseline.py", "return global_feat", "return global_feat + 1", "baseline_forward"),
            ("modeling/baseline.py", "self.channels = channels", "self.channels = channels + 1", "part_attention_init"),
            ("modeling/baseline.py", "return feature_map", "return feature_map + 1", "part_attention_forward"),
            ("layers/part_correspondence_consistency.py", "return [(i * height // num_parts", "return [(i * height // num_parts + 1", "horizontal_part_bounds"),
            ("layers/part_correspondence_consistency.py", "return adaptive_average_pool(feature_map, bounds)", "return max_pool(feature_map, bounds)", "build_local_part_descriptors"),
            ("layers/part_correspondence_consistency.py", "return same_pid_different_camera_upper_triangle(pids, camids)", "return all_pairs(pids, camids)", "build_cross_camera_positive_pairs"),
            ("layers/part_correspondence_consistency.py", "return local_features[pair_indices[:, 0]], local_features[pair_indices[:, 1]]", "return local_features[pair_indices[:, 1]], local_features[pair_indices[:, 0]]", "select_pair_local_features"),
            ("layers/part_correspondence_consistency.py", "return euclidean_l2(local_a, local_b)", "return cosine_distance(local_a, local_b)", "pairwise_local_distance_matrix"),
            ("modeling/backbones/resnet.py", "self.layer1(x)", "self.stem(x)", "resnet50_backbone"),
        )
        for path, old, new, component in cases:
            with self.subTest(component=component):
                evidence = self.evidence({path: (old, new)})
                self.assertEqual(evidence["feature_compatibility_status"], "incompatible")
                self.assertIn(component, evidence["mismatched_components"])

    def test_alignment_and_gating_specific_additions_are_excluded(self):
        evidence = self.evidence(suffix="""
def soft_min_path_costs(distance, tau):
    return distance - tau

class FutureDynamicGating:
    def forward(self, value):
        return value
""")
        self.assertEqual(evidence["feature_compatibility_status"], "compatible")

    def test_invalid_or_missing_parent_fails_closed(self):
        with self.assertRaises(FeatureCompatibilityError):
            build_feature_compatibility_evidence(
                REPO_ROOT, "", SOFT_SHA, self.hard, self.soft,
                parameter_schema_provider=parameter_provider,
            )
        with self.assertRaises(FeatureCompatibilityError):
            git_show_source(REPO_ROOT, "f" * 40, "modeling/baseline.py")

    def test_schema_v4_soft_finalizer_accepts_bound_compatible_evidence(self):
        from tests.test_experiment_recording import make_fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, run_dir, _output, experiments = make_fixture(
                root, variant="pcc", pcc_mode="soft_min"
            )
            manifest_path = run_dir / "run_manifest.json"
            manifest = read_json(manifest_path)
            signature_sha = manifest[
                "multigranular_feature_signature_sha256"
            ]
            feature = {
                "schema_version": 1,
                "feature_reference_commit": manifest["parent_commit"],
                "feature_current_commit": manifest["commit_id"],
                "feature_reference_signature_sha256": signature_sha,
                "current_feature_signature_sha256": signature_sha,
                "feature_compatibility_status": "compatible",
                "mismatched_components": [],
                "components": {"baseline_forward": {
                    "reference_sha256": "d" * 64,
                    "current_sha256": "d" * 64,
                    "status": "compatible",
                }},
            }
            feature_path = run_dir / "feature_compatibility.json"
            atomic_write_json(feature_path, feature)
            console = run_dir / "console.log"
            console.write_text("complete console\n", encoding="utf-8")
            manifest.update({
                "schema_version": SCHEMA_VERSION,
                "feature_reference_commit": feature[
                    "feature_reference_commit"
                ],
                "feature_reference_signature_sha256": signature_sha,
                "current_feature_signature_sha256": signature_sha,
                "feature_compatibility_status": "compatible",
                "feature_compatibility_evidence_path": str(
                    feature_path.resolve()
                ),
                "feature_compatibility_evidence_sha256": sha256_file(
                    feature_path
                ),
                "console_log_path": str(console.resolve()),
                "console_log_sha256": sha256_file(console),
            })
            atomic_write_json(manifest_path, manifest)
            model_path = run_dir / "model_manifest.json"
            model = read_json(model_path)
            for field in (
                    "feature_reference_commit",
                    "feature_reference_signature_sha256",
                    "current_feature_signature_sha256",
                    "feature_compatibility_status",
                    "feature_compatibility_evidence_sha256"):
                model[field] = manifest[field]
            atomic_write_json(model_path, model)
            result = finalize_run(
                run_dir, records, root, experiments,
                run_analyses=False, verify_git=False,
            )
            self.assertEqual(result["status"]["status"], "success")


if __name__ == "__main__":
    unittest.main()
