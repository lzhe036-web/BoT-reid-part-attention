# encoding: utf-8
"""Revision-aware signatures for the shared multi-granularity feature path."""

from __future__ import absolute_import

import ast
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml


STATIC_BASELINE_BRANCH = "exp/c2-l03-multi-granularity-local-feature"
STATIC_BASELINE_SHA = "9cd7dbcee07b255803c8c21f4d9c5ee67a30930e"
STATIC_CONFIG_PATH = (
    "configs/softmax_triplet_c2_l03_multi_granularity_part_autodl.yml"
)
DYNAMIC_CONFIG_PATH = (
    "configs/softmax_triplet_c2_l03_multi_granularity_dynamic_gating_autodl.yml"
)

_CONFIG_COMPONENTS = {
    "backbone_name": "MODEL.NAME",
    "part_attention_enabled": "MODEL.PART_ATTENTION",
    "part_attention_parts": "MODEL.PART_ATTENTION_PARTS",
    "multi_granularity_enabled": "MODEL.MULTI_GRANULARITY_PART",
    "scale_order": "MODEL.MULTI_GRANULARITY_PART_SCALES",
    "projection_dim": "MODEL.MULTI_GRANULARITY_PART_DIM",
    "aggregation": "MODEL.MULTI_GRANULARITY_PART_AGGREGATION",
    "fusion_shape_contract": "MODEL.MULTI_GRANULARITY_PART_FUSION",
    "neck": "MODEL.NECK",
    "neck_feat": "TEST.NECK_FEAT",
    "feature_norm": "TEST.FEAT_NORM",
    "re_ranking": "TEST.RE_RANKING",
    "cross_camera_pair_rule_enabled": "MODEL.CROSS_CAMERA_POSITIVE_ONLY",
    "cross_camera_pair_rule_mode": "MODEL.CROSS_CAMERA_POSITIVE_MODE",
}


class FeatureCompatibilityError(RuntimeError):
    pass


def _canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git(repo_root, args, binary=False):
    command = ["git", "-C", str(Path(repo_root).resolve())] + list(args)
    try:
        output = subprocess.check_output(command, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", b"") or b""
        raise FeatureCompatibilityError(
            "Git revision read failed for {}: {}".format(
                " ".join(args), stderr.decode("utf-8", errors="replace").strip()
            )
        ) from error
    return output if binary else output.decode("utf-8", errors="replace")


def require_commit(repo_root, revision):
    revision = str(revision).lower()
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise FeatureCompatibilityError(
            "Feature reference revision must be a full 40-character SHA"
        )
    resolved = _git(repo_root, ["rev-parse", "{}^{{commit}}".format(revision)]).strip()
    if resolved != revision:
        raise FeatureCompatibilityError(
            "Feature reference commit resolved to {}, expected {}".format(
                resolved, revision
            )
        )
    return resolved


def git_show_source(repo_root, revision, relative_path):
    """Read a source blob directly from Git without touching the worktree."""
    require_commit(repo_root, revision)
    normalized = Path(relative_path).as_posix().lstrip("/")
    if ".." in Path(normalized).parts:
        raise FeatureCompatibilityError("Repository-relative path escapes the repository")
    return _git(repo_root, ["show", "{}:{}".format(revision, normalized)])


def _find_class(tree, class_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise FeatureCompatibilityError("Missing class {}".format(class_name))


def _find_method(class_node, method_name):
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return node
    raise FeatureCompatibilityError(
        "Missing component {}.{}".format(class_node.name, method_name)
    )


def _contains_dynamic_gating(node):
    return "dynamic_gating" in ast.dump(node, include_attributes=False).lower()


class _SharedBaselineNormalizer(ast.NodeTransformer):
    """Remove only the declared gating experiment variable from shared methods."""

    def visit_arguments(self, node):
        node = self.generic_visit(node)
        retained = []
        defaults = list(node.defaults)
        default_offset = len(node.args) - len(defaults)
        retained_defaults = []
        for index, argument in enumerate(node.args):
            if "dynamic_gating" in argument.arg or "gating_" in argument.arg:
                continue
            retained.append(argument)
            if index >= default_offset:
                retained_defaults.append(defaults[index - default_offset])
        node.args = retained
        node.defaults = retained_defaults
        return node

    def visit_If(self, node):
        if _contains_dynamic_gating(node.test):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if _contains_dynamic_gating(node):
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if _contains_dynamic_gating(node):
            return None
        return self.generic_visit(node)


def _normalized_ast(node, shared_baseline=False):
    if shared_baseline:
        node = _SharedBaselineNormalizer().visit(node)
        if node is None:
            raise FeatureCompatibilityError("Shared AST normalization removed a component")
        ast.fix_missing_locations(node)
    return ast.dump(node, include_attributes=False)


def _source_component_hashes(source_map):
    baseline_tree = ast.parse(source_map["modeling/baseline.py"])
    part_attention = _find_class(baseline_tree, "PartAttentionHead")
    multi_head = _find_class(baseline_tree, "MultiGranularityPartHead")
    baseline = _find_class(baseline_tree, "Baseline")
    triplet_tree = ast.parse(source_map["layers/triplet_loss.py"])
    resnet_tree = ast.parse(source_map["modeling/backbones/resnet.py"])

    nodes = {
        "PartAttentionHead.__init__": _find_method(part_attention, "__init__"),
        "PartAttentionHead.forward": _find_method(part_attention, "forward"),
        "MultiGranularityPartHead.__init__": _find_method(multi_head, "__init__"),
        "horizontal_partition": _find_method(multi_head, "region_bounds"),
        "part_pooling": _find_method(multi_head, "pool_parts"),
        "local_descriptor_builder": _find_method(multi_head, "forward"),
        "Baseline.__init__shared": _find_method(baseline, "__init__"),
        "Baseline.forward_shared": _find_method(baseline, "forward"),
        "ResNet50.backbone_structure_forward": resnet_tree,
    }
    for name in ("CrossCameraPositiveLoss", "TripletLoss"):
        nodes["pair_distance.{}".format(name)] = _find_class(triplet_tree, name)
    for node in triplet_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "count_cross_camera_positives":
            nodes["same_pid_different_camera_pair_rule"] = node
            break
    if "same_pid_different_camera_pair_rule" not in nodes:
        raise FeatureCompatibilityError(
            "Missing count_cross_camera_positives pair-selection component"
        )

    hashes = {}
    for name, node in nodes.items():
        normalized = _normalized_ast(
            node, shared_baseline=name.startswith("Baseline.")
        )
        hashes[name] = _sha256_text(normalized)
    return hashes


def _nested(mapping, dotted_path):
    value = mapping
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise FeatureCompatibilityError(
                "Configuration is missing {}".format(dotted_path)
            )
        value = value[part]
    return value


def _load_yaml_text(text, label):
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise FeatureCompatibilityError("{} is not a YAML mapping".format(label))
    return value


def _config_components(configuration):
    return {
        name: _nested(configuration, dotted)
        for name, dotted in sorted(_CONFIG_COMPONENTS.items())
    }


_PARAMETER_SCHEMA_SCRIPT = r'''
import json
from config import cfg
from modeling import build_model
c = cfg.clone()
c.merge_from_file(CONFIG_PATH)
c.defrost()
c.MODEL.PRETRAIN_CHOICE = "none"
c.MODEL.PRETRAIN_PATH = ""
c.freeze()
m = build_model(c, num_classes=751)
schema = {}
for name, value in m.state_dict().items():
    if "multi_granularity_dynamic_gate" in name or name.startswith("classifier."):
        continue
    schema[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
print(json.dumps(schema, sort_keys=True, separators=(",", ":")))
'''


def _safe_extract_tar(payload, destination):
    root = Path(destination).resolve()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise FeatureCompatibilityError("Unsafe path in Git archive")
        archive.extractall(str(root))


def _revision_parameter_schema(repo_root, revision, config_path,
                               python_executable=None):
    require_commit(repo_root, revision)
    archive = _git(repo_root, ["archive", "--format=tar", revision], binary=True)
    with tempfile.TemporaryDirectory(prefix="bot-feature-signature-") as temp_dir:
        _safe_extract_tar(archive, temp_dir)
        environment = os.environ.copy()
        environment["CONFIG_PATH"] = str(Path(temp_dir) / config_path)
        script = "import os\nCONFIG_PATH=os.environ['CONFIG_PATH']\n" + _PARAMETER_SCHEMA_SCRIPT
        completed = subprocess.run(
            [python_executable or sys.executable, "-c", script],
            cwd=temp_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise FeatureCompatibilityError(
                "Isolated parameter-schema inspection failed: {}".format(
                    completed.stderr.decode("utf-8", errors="replace").strip()
                )
            )
        try:
            return json.loads(
                completed.stdout.decode("utf-8", errors="replace").strip()
            )
        except (TypeError, ValueError) as error:
            raise FeatureCompatibilityError(
                "Isolated parameter-schema inspection returned invalid JSON"
            ) from error


def revision_shared_signature(repo_root, revision, config_path,
                              source_overrides=None, config_override=None,
                              parameter_schema=None, python_executable=None):
    paths = (
        "modeling/baseline.py",
        "modeling/backbones/resnet.py",
        "layers/triplet_loss.py",
    )
    source_map = dict(source_overrides or {})
    for path in paths:
        if path not in source_map:
            source_map[path] = git_show_source(repo_root, revision, path)
    if config_override is None:
        config_text = git_show_source(repo_root, revision, config_path)
        configuration = _load_yaml_text(config_text, config_path)
    else:
        configuration = dict(config_override)
    if parameter_schema is None:
        parameter_schema = _revision_parameter_schema(
            repo_root, revision, config_path,
            python_executable=python_executable,
        )
    signature = {
        "signature_schema_version": 1,
        "component_ast_sha256": _source_component_hashes(source_map),
        "configuration": _config_components(configuration),
        "descriptor_contract": {
            "global_descriptor_dim": 2048,
            "local_descriptor_dims_in_scale_order": [256, 256, 256],
            "inference_descriptor_dim": 2816,
            "single_backbone_forward": True,
        },
        "shared_parameter_schema": parameter_schema,
    }
    return signature, _sha256_text(_canonical_json(signature))


def _fusion_signature(configuration, dynamic):
    payload = {
        "signature_schema_version": 1,
        "fusion": _nested(configuration, "MODEL.MULTI_GRANULARITY_PART_FUSION"),
        "scale_order": _nested(configuration, "MODEL.MULTI_GRANULARITY_PART_SCALES"),
        "descriptor_dim": 2816,
        "dynamic_gating": bool(dynamic),
    }
    if dynamic:
        payload["controller"] = {
            "input": _nested(configuration, "MODEL.MULTI_GRANULARITY_GATING_INPUT"),
            "linear": [2048, 3],
            "temperature": _nested(configuration, "MODEL.MULTI_GRANULARITY_GATING_TAU"),
            "normalization": _nested(
                configuration, "MODEL.MULTI_GRANULARITY_GATING_NORMALIZATION"
            ),
            "weight_scale": 3.0,
            "weight_initialization": "zeros",
            "bias_initialization": "zeros",
        }
    else:
        payload["controller"] = "not_applicable"
    return payload, _sha256_text(_canonical_json(payload))


def build_feature_compatibility_evidence(
        repo_root, reference_commit, current_commit,
        reference_config=STATIC_CONFIG_PATH,
        current_config=DYNAMIC_CONFIG_PATH, python_executable=None,
        reference_source_overrides=None, current_source_overrides=None,
        reference_config_override=None, current_config_override=None,
        reference_parameter_schema=None, current_parameter_schema=None):
    reference_commit = require_commit(repo_root, reference_commit)
    current_commit = require_commit(repo_root, current_commit)
    reference_signature, reference_sha = revision_shared_signature(
        repo_root, reference_commit, reference_config,
        source_overrides=reference_source_overrides,
        config_override=reference_config_override,
        parameter_schema=reference_parameter_schema,
        python_executable=python_executable,
    )
    current_signature, current_sha = revision_shared_signature(
        repo_root, current_commit, current_config,
        source_overrides=current_source_overrides,
        config_override=current_config_override,
        parameter_schema=current_parameter_schema,
        python_executable=python_executable,
    )
    reference_components = reference_signature["component_ast_sha256"]
    current_components = current_signature["component_ast_sha256"]
    mismatches = sorted(
        name for name in set(reference_components) | set(current_components)
        if reference_components.get(name) != current_components.get(name)
    )
    for name in ("configuration", "descriptor_contract", "shared_parameter_schema"):
        if reference_signature.get(name) != current_signature.get(name):
            mismatches.append(name)

    reference_cfg = (
        reference_config_override
        if reference_config_override is not None
        else _load_yaml_text(
            git_show_source(repo_root, reference_commit, reference_config),
            reference_config,
        )
    )
    current_cfg = (
        current_config_override
        if current_config_override is not None
        else _load_yaml_text(
            git_show_source(repo_root, current_commit, current_config),
            current_config,
        )
    )
    reference_fusion, reference_fusion_sha = _fusion_signature(
        reference_cfg, dynamic=False
    )
    current_fusion, current_fusion_sha = _fusion_signature(
        current_cfg, dynamic=True
    )
    return {
        "schema_version": 1,
        "feature_reference_branch": STATIC_BASELINE_BRANCH,
        "feature_reference_commit": reference_commit,
        "current_commit": current_commit,
        "source_read_method": "git show <revision>:<repo-relative-path>",
        "parameter_schema_read_method": "isolated git archive",
        "feature_reference_signature_sha256": reference_sha,
        "current_feature_signature_sha256": current_sha,
        "feature_compatibility_status": (
            "compatible" if not mismatches and reference_sha == current_sha
            else "incompatible"
        ),
        "mismatched_components": sorted(set(mismatches)),
        "components": {
            name: {
                "reference_sha256": reference_components.get(name),
                "current_sha256": current_components.get(name),
                "status": (
                    "compatible"
                    if reference_components.get(name) == current_components.get(name)
                    else "incompatible"
                ),
            }
            for name in sorted(set(reference_components) | set(current_components))
        },
        "shared_signature": current_signature,
        "fusion_gating_signature": {
            "reference": reference_fusion,
            "reference_sha256": reference_fusion_sha,
            "current": current_fusion,
            "current_sha256": current_fusion_sha,
            "status": (
                "expected_experiment_difference"
                if reference_fusion_sha != current_fusion_sha
                else "unexpectedly_identical"
            ),
        },
    }


def require_feature_compatibility(evidence):
    if evidence.get("feature_reference_commit") != STATIC_BASELINE_SHA:
        raise FeatureCompatibilityError(
            "Dynamic gating must bind to fixed Static baseline {}".format(
                STATIC_BASELINE_SHA
            )
        )
    if evidence.get("feature_compatibility_status") != "compatible":
        raise FeatureCompatibilityError(
            "Shared multi-granularity feature mismatch: {}".format(
                evidence.get("mismatched_components", [])
            )
        )
    fusion = evidence.get("fusion_gating_signature", {})
    if fusion.get("status") != "expected_experiment_difference":
        raise FeatureCompatibilityError(
            "Fusion/gating signature does not identify the declared experiment variable"
        )
    return evidence
