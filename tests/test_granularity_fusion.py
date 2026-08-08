# encoding: utf-8
"""Tests for the Parallel-B static/dynamic granularity comparison."""

from __future__ import absolute_import

import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
import yaml

from config import cfg
from modeling.baseline import Baseline
from modeling.granularity_fusion import (
    GRANULARITY_LABELS,
    GranularityFusion,
    fusion_parameter_counts,
)
from tools.analyze_granularity_gating import (
    load_checkpoint_strict,
    write_gate_analysis_outputs,
)
from tools.run_experiment import _model_manifest
from utils.experiment_recording import (
    FUSION_FIELDS,
    MAIN_FIELDS,
    RUN_FIELDS,
    experiment_identity,
    sha256_file,
    upsert_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_CONFIG = (
    REPO_ROOT / "configs" /
    "softmax_triplet_c2l03_static_granularity_fusion_autodl.yml"
)
DYNAMIC_CONFIG = (
    REPO_ROOT / "configs" /
    "softmax_triplet_c2l03_dynamic_granularity_gating_autodl.yml"
)


class CountingBackbone(nn.Module):
    def __init__(self, height=13, width=5):
        super(CountingBackbone, self).__init__()
        self.height = height
        self.width = width
        self.calls = 0
        self.projection = nn.Conv2d(3, 2048, kernel_size=1, bias=False)

    def forward(self, inputs):
        self.calls += 1
        features = self.projection(inputs)
        return torch.nn.functional.adaptive_avg_pool2d(
            features, (self.height, self.width)
        )


def build_synthetic_model(mode="dynamic", num_classes=8):
    model = Baseline(
        num_classes=num_classes,
        last_stride=1,
        model_path="",
        neck="bnneck",
        neck_feat="after",
        model_name="resnet50",
        pretrain_choice="none",
        part_attention=True,
        part_attention_parts=6,
        part_correspondence_consistency=False,
        pcc_parts=6,
        multi_granularity_local=True,
        multi_granularity_scales=(2, 4, 6),
        multi_granularity_dim=256,
        multi_granularity_aggregation="mean",
        multi_granularity_fusion=True,
        multi_granularity_fusion_mode=mode,
        multi_granularity_fusion_dim=256,
        dynamic_gating_hidden_dim=256,
    )
    model.base = CountingBackbone()
    return model


def nested_differences(left, right, prefix=""):
    differences = []
    keys = set(left) | set(right)
    for key in sorted(keys):
        path = "{}.{}".format(prefix, key) if prefix else key
        if key not in left or key not in right:
            differences.append(path)
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            differences.extend(nested_differences(left[key], right[key], path))
        elif left[key] != right[key]:
            differences.append(path)
    return differences


class GranularityFusionTest(unittest.TestCase):
    def test_four_components_have_identical_shapes(self):
        module = GranularityFusion(8, fusion_dim=4, hidden_dim=5,
                                   mode="dynamic")
        components = module.build_components(
            torch.randn(3, 8), tuple(torch.randn(3, 4) for _ in range(3))
        )
        self.assertEqual(tuple(components.shape), (3, 4, 4))
        for index in range(4):
            self.assertEqual(tuple(components[:, index].shape), (3, 4))

    def test_static_weights_are_normalized_shared_and_trainable(self):
        module = GranularityFusion(8, fusion_dim=4, hidden_dim=5,
                                   mode="static")
        fused, details = module(
            torch.randn(3, 8), tuple(torch.randn(3, 4) for _ in range(3)),
            return_details=True,
        )
        weights = details["weights"]
        self.assertTrue(torch.equal(weights[0], weights[1]))
        self.assertTrue(torch.equal(weights[1], weights[2]))
        self.assertTrue(torch.equal(weights.sum(dim=1), torch.ones(3)))
        self.assertTrue(torch.equal(weights[0], torch.full((4,), 0.25)))
        fused.pow(2).sum().backward()
        self.assertIsNotNone(module.shared_logits.grad)

    def test_dynamic_shape_normalization_and_no_metadata_inputs(self):
        module = GranularityFusion(8, fusion_dim=4, hidden_dim=5,
                                   mode="dynamic")
        _fused, details = module(
            torch.randn(3, 8), tuple(torch.randn(3, 4) for _ in range(3)),
            return_details=True,
        )
        self.assertEqual(tuple(details["weights"].shape), (3, 4))
        self.assertTrue(torch.equal(
            details["weights"].sum(dim=1), torch.ones(3)
        ))
        parameters = tuple(inspect.signature(module.forward).parameters)
        for forbidden in ("pid", "camid", "label", "target"):
            self.assertNotIn(forbidden, parameters)

    def test_zero_initialized_dynamic_exactly_matches_static(self):
        static = GranularityFusion(8, fusion_dim=4, hidden_dim=5,
                                   mode="static")
        dynamic = GranularityFusion(8, fusion_dim=4, hidden_dim=5,
                                    mode="dynamic")
        dynamic.global_projection.load_state_dict(
            static.global_projection.state_dict()
        )
        global_feature = torch.randn(3, 8)
        local_features = tuple(torch.randn(3, 4) for _ in range(3))
        static_output, static_details = static(
            global_feature, local_features, return_details=True
        )
        dynamic_output, dynamic_details = dynamic(
            global_feature, local_features, return_details=True
        )
        self.assertTrue(torch.equal(
            static_details["weights"], dynamic_details["weights"]
        ))
        self.assertTrue(torch.equal(static_output, dynamic_output))
        self.assertTrue(torch.equal(
            dynamic.gating_mlp[2].weight,
            torch.zeros_like(dynamic.gating_mlp[2].weight),
        ))
        self.assertTrue(torch.equal(
            dynamic.gating_mlp[2].bias,
            torch.zeros_like(dynamic.gating_mlp[2].bias),
        ))

    def test_manually_configured_gate_varies_with_sample_content(self):
        module = GranularityFusion(8, fusion_dim=4, hidden_dim=5,
                                   mode="dynamic")
        with torch.no_grad():
            module.gating_mlp[0].weight.zero_()
            module.gating_mlp[0].bias.zero_()
            module.gating_mlp[0].weight[0, 0] = 1.0
            module.gating_mlp[2].weight.zero_()
            module.gating_mlp[2].bias.zero_()
            module.gating_mlp[2].weight[0, 0] = 1.0
            module.gating_mlp[2].weight[1, 0] = -1.0
        components = torch.zeros(2, 4, 4)
        components[1, 0, 0] = 2.0
        weights = module.weights_from_components(components)
        self.assertFalse(torch.equal(weights[0], weights[1]))

    def test_gradients_reach_all_new_branches_and_backbone_runs_once(self):
        model = build_synthetic_model("dynamic")
        model.train()
        scores, descriptor = model(torch.randn(4, 3, 20, 8))
        (scores.pow(2).mean() + descriptor.pow(2).mean()).backward()
        self.assertEqual(model.base.calls, 1)
        self.assertEqual(tuple(descriptor.shape), (4, 256))
        self.assertIsNotNone(
            model.granularity_fusion.global_projection.weight.grad
        )
        for projection in model.multi_granularity_head.projections.values():
            self.assertIsNotNone(projection.weight.grad)
        for parameter in model.granularity_fusion.gating_mlp.parameters():
            self.assertIsNotNone(parameter.grad)

    def test_static_and_dynamic_descriptor_dimensions_and_parameter_delta(self):
        static = build_synthetic_model("static")
        dynamic = build_synthetic_model("dynamic")
        self.assertEqual(static.descriptor_dim, 256)
        self.assertEqual(dynamic.descriptor_dim, 256)
        static_count = sum(
            parameter.numel()
            for parameter in static.granularity_fusion.parameters()
        )
        dynamic_count = sum(
            parameter.numel()
            for parameter in dynamic.granularity_fusion.parameters()
        )
        expected = fusion_parameter_counts(2048, 256, 256, 4)
        self.assertEqual(static_count, expected["static"])
        self.assertEqual(dynamic_count, expected["dynamic"])
        self.assertEqual(dynamic_count - static_count, 263424)

    def test_control_and_dynamic_configs_differ_only_by_mode_and_output(self):
        static = yaml.safe_load(STATIC_CONFIG.read_text(encoding="utf-8"))
        dynamic = yaml.safe_load(DYNAMIC_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            nested_differences(static, dynamic),
            ["MODEL.MULTI_GRANULARITY_FUSION_MODE", "OUTPUT_DIR"],
        )

    def test_formal_configs_preserve_parallel_b_independent_variables(self):
        for path, mode in ((STATIC_CONFIG, "static"),
                           (DYNAMIC_CONFIG, "dynamic")):
            source = yaml.safe_load(path.read_text(encoding="utf-8"))
            model = source["MODEL"]
            self.assertEqual(source["SEED"], 42)
            self.assertTrue(model["PART_ATTENTION"])
            self.assertEqual(model["PART_ATTENTION_PARTS"], 6)
            self.assertTrue(model["MULTI_GRANULARITY_LOCAL"])
            self.assertTrue(model["MULTI_GRANULARITY_FUSION"])
            self.assertEqual(model["MULTI_GRANULARITY_FUSION_MODE"], mode)
            self.assertFalse(model["PART_CORRESPONDENCE_CONSISTENCY"])
            self.assertFalse(model["CAMERA_CONDITIONAL_PART_ATTENTION"])
            self.assertTrue(model["CROSS_CAMERA_POSITIVE_ONLY"])
            self.assertEqual(model["CROSS_CAMERA_POSITIVE_LAMBDA"], 0.3)
            local_cfg = cfg.clone()
            local_cfg.merge_from_file(str(path))
            self.assertEqual(local_cfg.TEST.NECK_FEAT, "after")
            self.assertEqual(local_cfg.TEST.FEAT_NORM, "yes")
            self.assertEqual(local_cfg.TEST.RE_RANKING, "no")

    def test_record_identity_and_manifest_use_required_method_names(self):
        static = yaml.safe_load(STATIC_CONFIG.read_text(encoding="utf-8"))
        dynamic = yaml.safe_load(DYNAMIC_CONFIG.read_text(encoding="utf-8"))
        static_identity = experiment_identity(static)
        dynamic_identity = experiment_identity(dynamic)
        self.assertEqual(
            static_identity["method"],
            "C2-L03 + Multi-Granularity Static Fusion",
        )
        self.assertEqual(
            dynamic_identity["method"],
            "C2-L03 + Dynamic Granularity Gating",
        )
        self.assertFalse(static_identity["dynamic_granularity_gating"])
        self.assertTrue(dynamic_identity["dynamic_granularity_gating"])
        manifest = _model_manifest(dynamic)
        self.assertEqual(manifest["descriptor_dim"], 256)
        self.assertEqual(manifest["component_count"], 4)
        self.assertEqual(manifest["static_parameter_count"], 524548)
        self.assertEqual(manifest["dynamic_parameter_count"], 787972)

    def test_checkpoint_load_is_strict(self):
        source = GranularityFusion(8, 4, 5, mode="dynamic")
        target = GranularityFusion(8, 4, 5, mode="dynamic")
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.pt"
            bad = Path(directory) / "bad.pt"
            torch.save(source.state_dict(), str(good))
            load_checkpoint_strict(target, good)
            for name, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[name]))
            incomplete = dict(source.state_dict())
            incomplete.pop(next(iter(incomplete)))
            torch.save(incomplete, str(bad))
            with self.assertRaises(RuntimeError):
                load_checkpoint_strict(target, bad)

    def test_gate_analysis_outputs_samples_statistics_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            config_path = root / "config.yml"
            torch.save({"tensor": torch.ones(1)}, str(checkpoint))
            config_path.write_text("SEED: 42\n", encoding="utf-8")
            rows = []
            for index, weights in enumerate(
                    ((0.25, 0.25, 0.25, 0.25),
                     (0.10, 0.20, 0.30, 0.40))):
                row = {
                    "image_path": "image-{}.jpg".format(index),
                    "pid": index,
                    "camid": index + 1,
                    "gate_entropy": -sum(
                        value * torch.log(torch.tensor(value)).item()
                        for value in weights
                    ),
                    "max_granularity": GRANULARITY_LABELS[
                        max(range(4), key=lambda item: weights[item])
                    ],
                }
                for label, value in zip(GRANULARITY_LABELS, weights):
                    row["weight_{}".format(label)] = value
                rows.append(row)
            csv_path, summary_path = write_gate_analysis_outputs(
                rows, checkpoint, config_path, root,
                {"seed": 42, "branch": "branch", "commit": "abc"},
            )
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(written), 2)
            self.assertIn("weight_global", written[0])
            self.assertIn("gate_entropy", written[0])
            self.assertIn("max_granularity", written[0])
            self.assertEqual(
                summary["checkpoint_sha256"], sha256_file(checkpoint)
            )
            self.assertEqual(
                summary["per_sample_csv_sha256"], sha256_file(csv_path)
            )
            self.assertGreater(
                summary["sample_weight_variance"]["global"], 0.0
            )
            self.assertTrue(
                summary["random_initialization_is_not_adaptive_evidence"]
            )

    def test_record_schemas_and_historical_rows_are_backward_compatible(self):
        for field in (
                "multi_granularity_fusion", "fusion_mode",
                "dynamic_granularity_gating", "fusion_dimension",
                "gating_hidden_dimension", "component_count",
                "static_parameter_count", "dynamic_parameter_count",
                "gate_analysis_path", "gate_analysis_sha256"):
            self.assertIn(field, MAIN_FIELDS)
            self.assertIn(field, RUN_FIELDS)
            self.assertIn(field, FUSION_FIELDS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical.csv"
            path.write_text(
                "run_id,method\nlegacy-run,Historical Method\n",
                encoding="utf-8",
            )
            upsert_csv(
                path,
                ("run_id", "method", "fusion_mode"),
                {
                    "run_id": "new-run",
                    "method": "Dynamic",
                    "fusion_mode": "dynamic",
                },
            )
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            historical = next(row for row in rows
                              if row["run_id"] == "legacy-run")
            self.assertEqual(historical["method"], "Historical Method")
            self.assertEqual(historical["fusion_mode"], "")


if __name__ == "__main__":
    unittest.main()
