import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import yaml

from utils.multigranularity_signatures import (
    DYNAMIC_CONFIG_PATH,
    STATIC_BASELINE_SHA,
    STATIC_CONFIG_PATH,
    FeatureCompatibilityError,
    build_feature_compatibility_evidence,
    git_show_source,
    require_feature_compatibility,
    revision_shared_signature,
)
import utils.multigranularity_signatures as signatures


REPO_ROOT = Path(__file__).resolve().parents[1]


class FeatureSignatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_sources = {
            path: git_show_source(REPO_ROOT, STATIC_BASELINE_SHA, path)
            for path in (
                "modeling/baseline.py", "modeling/backbones/resnet.py",
                "layers/triplet_loss.py",
            )
        }
        cls.current_sources = {
            path: (REPO_ROOT / path).read_text(encoding="utf-8")
            for path in cls.parent_sources
        }
        cls.static_config = yaml.safe_load(
            git_show_source(REPO_ROOT, STATIC_BASELINE_SHA, STATIC_CONFIG_PATH)
        )
        cls.dynamic_config = yaml.safe_load(
            (REPO_ROOT / DYNAMIC_CONFIG_PATH).read_text(encoding="utf-8")
        )
        parent_signature, _ = revision_shared_signature(
            REPO_ROOT, STATIC_BASELINE_SHA, STATIC_CONFIG_PATH,
            source_overrides=cls.parent_sources,
            config_override=cls.static_config,
            python_executable=sys.executable,
        )
        cls.parameter_schema = parent_signature["shared_parameter_schema"]

    def evidence(self, current_sources=None, current_config=None):
        return build_feature_compatibility_evidence(
            REPO_ROOT, STATIC_BASELINE_SHA, STATIC_BASELINE_SHA,
            reference_source_overrides=self.parent_sources,
            current_source_overrides=current_sources or self.current_sources,
            reference_config_override=self.static_config,
            current_config_override=current_config or self.dynamic_config,
            reference_parameter_schema=self.parameter_schema,
            current_parameter_schema=self.parameter_schema,
        )

    def test_current_dynamic_shared_signature_matches_fixed_parent(self):
        evidence = require_feature_compatibility(self.evidence())
        self.assertEqual(evidence["feature_reference_commit"], STATIC_BASELINE_SHA)
        self.assertEqual(
            evidence["feature_reference_signature_sha256"],
            evidence["current_feature_signature_sha256"],
        )

    def test_parent_source_is_read_with_git_show(self):
        with mock.patch.object(signatures, "_git", wraps=signatures._git) as git_call:
            text = git_show_source(
                REPO_ROOT, STATIC_BASELINE_SHA, "modeling/baseline.py"
            )
        self.assertIn("class Baseline", text)
        calls = [call.args[1] for call in git_call.call_args_list]
        self.assertIn(
            ["show", "{}:modeling/baseline.py".format(STATIC_BASELINE_SHA)],
            calls,
        )

    def test_gating_mode_or_temperature_does_not_change_shared_signature(self):
        modified = copy.deepcopy(self.dynamic_config)
        modified["MODEL"]["MULTI_GRANULARITY_GATING_TAU"] = 3.5
        evidence = self.evidence(current_config=modified)
        self.assertEqual(evidence["feature_compatibility_status"], "compatible")

    def test_shared_baseline_forward_change_is_detected_and_named(self):
        sources = dict(self.current_sources)
        sources["modeling/baseline.py"] = sources["modeling/baseline.py"].replace(
            "global_feat = self.gap(feature_map)",
            "global_feat = self.gap(feature_map) * 2.0",
        )
        evidence = self.evidence(sources)
        self.assertEqual(evidence["feature_compatibility_status"], "incompatible")
        self.assertIn("Baseline.forward_shared", evidence["mismatched_components"])

    def test_backbone_forward_change_is_detected(self):
        sources = dict(self.current_sources)
        before, separator, _tail = sources["modeling/backbones/resnet.py"].rpartition(
            "return x"
        )
        self.assertTrue(separator)
        sources["modeling/backbones/resnet.py"] = before + "return x * 1.01" + _tail
        evidence = self.evidence(sources)
        self.assertIn(
            "ResNet50.backbone_structure_forward", evidence["mismatched_components"]
        )

    def test_partition_pooling_change_is_detected(self):
        sources = dict(self.current_sources)
        sources["modeling/baseline.py"] = sources["modeling/baseline.py"].replace(
            "height * part_idx // scale",
            "(height * part_idx + 1) // scale",
            1,
        )
        evidence = self.evidence(sources)
        self.assertIn("horizontal_partition", evidence["mismatched_components"])

    def test_pair_rule_or_distance_change_is_detected(self):
        sources = dict(self.current_sources)
        sources["layers/triplet_loss.py"] = sources["layers/triplet_loss.py"].replace(
            "same_pid_mask", "same_pid_mask_modified", 1
        )
        evidence = self.evidence(sources)
        self.assertTrue(any(
            item.startswith("pair_distance.") or item == "same_pid_different_camera_pair_rule"
            for item in evidence["mismatched_components"]
        ))

    def test_adding_gating_specific_module_does_not_change_shared_signature(self):
        sources = dict(self.current_sources)
        sources["modeling/baseline.py"] += (
            "\nclass FutureGatingOnlyModule(object):\n"
            "    def forward(self, value):\n        return value\n"
        )
        evidence = self.evidence(sources)
        self.assertEqual(evidence["feature_compatibility_status"], "compatible")

    def test_invalid_or_missing_reference_commit_fails_closed(self):
        for value in ("deadbeef", "f" * 40):
            with self.subTest(value=value):
                with self.assertRaises(FeatureCompatibilityError):
                    build_feature_compatibility_evidence(
                        REPO_ROOT, value, STATIC_BASELINE_SHA,
                        reference_parameter_schema=self.parameter_schema,
                        current_parameter_schema=self.parameter_schema,
                    )

    def test_require_compatibility_reports_component_mismatches(self):
        sources = dict(self.current_sources)
        sources["modeling/baseline.py"] = sources["modeling/baseline.py"].replace(
            "projected_parts.mean(dim=1)", "projected_parts.sum(dim=1)"
        )
        with self.assertRaisesRegex(FeatureCompatibilityError, "local_descriptor_builder"):
            require_feature_compatibility(self.evidence(sources))


if __name__ == "__main__":
    unittest.main()
