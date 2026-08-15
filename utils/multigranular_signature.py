# encoding: utf-8
"""Stable, revision-aware signatures for the shared feature pipeline."""

from __future__ import absolute_import

import ast
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import torch


NOT_RECORDED = "not_recorded"
FIXED_HARD_FEATURE_REFERENCE_COMMIT = (
    "6b46f2c3747124b97d59ed5cf987f33efb82282b"
)
FEATURE_REFERENCE_CONFIG = (
    "configs/softmax_triplet_c2l03_hard_shortest_path_alignment_autodl.yml"
)
FEATURE_SOURCE_COMPONENTS = {
    "baseline_forward": (
        "modeling/baseline.py", ("Baseline", "forward")
    ),
    "part_attention_init": (
        "modeling/baseline.py", ("PartAttentionHead", "__init__")
    ),
    "part_attention_forward": (
        "modeling/baseline.py", ("PartAttentionHead", "forward")
    ),
    "horizontal_part_bounds": (
        "layers/part_correspondence_consistency.py",
        ("horizontal_part_bounds",),
    ),
    "build_local_part_descriptors": (
        "layers/part_correspondence_consistency.py",
        ("build_local_part_descriptors",),
    ),
    "build_cross_camera_positive_pairs": (
        "layers/part_correspondence_consistency.py",
        ("build_cross_camera_positive_pairs",),
    ),
    "select_pair_local_features": (
        "layers/part_correspondence_consistency.py",
        ("select_pair_local_features",),
    ),
    "pairwise_local_distance_matrix": (
        "layers/part_correspondence_consistency.py",
        ("pairwise_local_distance_matrix",),
    ),
    "resnet50_backbone": ("modeling/backbones/resnet.py", None),
}


class FeatureCompatibilityError(RuntimeError):
    pass


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _nested(mapping, dotted_path, default=NOT_RECORDED):
    current = mapping
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _resolve_revision(repo_root, revision):
    if not revision or revision == NOT_RECORDED:
        raise FeatureCompatibilityError("Feature reference revision is missing")
    try:
        resolved = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "{}^{{commit}}".format(
                revision
            )],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace").strip().lower()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FeatureCompatibilityError(
            "Cannot resolve feature revision {!r}: {}".format(revision, error)
        )
    if len(resolved) != 40:
        raise FeatureCompatibilityError("Feature revision is not a full commit")
    return resolved


def git_show_source(repo_root, revision, repo_relative_path):
    """Read one source file directly from a Git commit object."""
    resolved = _resolve_revision(repo_root, revision)
    try:
        raw = subprocess.check_output(
            [
                "git", "-C", str(repo_root), "show",
                "{}:{}".format(resolved, repo_relative_path),
            ],
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FeatureCompatibilityError(
            "Cannot read {} from {}: {}".format(
                repo_relative_path, resolved, error
            )
        )
    return raw.decode("utf-8", errors="replace")


def _find_ast_component(tree, qualified_name):
    nodes = list(tree.body)
    for index, name in enumerate(qualified_name):
        matches = [
            node for node in nodes
            if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef))
            and node.name == name
        ]
        if len(matches) != 1:
            raise FeatureCompatibilityError(
                "AST component {} is missing or ambiguous".format(
                    ".".join(qualified_name[:index + 1])
                )
            )
        node = matches[0]
        nodes = list(getattr(node, "body", ()))
    return node


def ast_component_sha256(source, qualified_name=None):
    """Hash semantic Python AST while ignoring comments and formatting."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise FeatureCompatibilityError("Cannot parse feature source: {}".format(error))
    node = tree if qualified_name is None else _find_ast_component(
        tree, tuple(qualified_name)
    )
    return _sha256_text(ast.dump(node, include_attributes=False))


def revision_source_component_hashes(repo_root, revision,
                                     source_provider=None):
    provider = source_provider or git_show_source
    result = {}
    source_cache = {}
    for name, (path, qualified_name) in sorted(FEATURE_SOURCE_COMPONENTS.items()):
        cache_key = (str(revision), path)
        if cache_key not in source_cache:
            source_cache[cache_key] = provider(repo_root, revision, path)
        result[name] = ast_component_sha256(
            source_cache[cache_key], qualified_name
        )
    return result


def _current_feature_parameter_schema(configuration):
    """Build deterministic feature parameter names/shapes without weights."""
    from modeling.baseline import Baseline

    kwargs = {
        "part_attention": bool(_nested(
            configuration, "MODEL.PART_ATTENTION", False
        )),
        "part_attention_parts": int(_nested(
            configuration, "MODEL.PART_ATTENTION_PARTS", 6
        )),
        "part_correspondence_consistency": bool(_nested(
            configuration, "MODEL.PART_CORRESPONDENCE_CONSISTENCY", False
        )),
        "pcc_parts": int(_nested(configuration, "MODEL.PCC_PARTS", 6)),
        "pcc_mode": str(_nested(
            configuration, "MODEL.PCC_MODE", "fixed_index"
        )),
        "pcc_softmin_tau": _nested(
            configuration, "MODEL.PCC_SOFTMIN_TAU", 0.1
        ),
    }
    with torch.random.fork_rng(devices=[]):
        model = Baseline(
            1,
            int(_nested(configuration, "MODEL.LAST_STRIDE", 1)),
            "",
            str(_nested(configuration, "MODEL.NECK", "bnneck")),
            str(_nested(configuration, "TEST.NECK_FEAT", "after")),
            str(_nested(configuration, "MODEL.NAME")),
            "none",
            **kwargs
        )
    prefixes = ("base.", "part_attention_head.", "bottleneck.")
    return [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    ], int(model.in_planes)


_ARCHIVE_SCHEMA_SCRIPT = r'''
import inspect, json, sys, torch
archive, configuration_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, archive)
configuration = json.loads(configuration_json)
def nested(path, default=None):
    value = configuration
    for part in path.split('.'):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value
from modeling.baseline import Baseline
kwargs = {
    'part_attention': bool(nested('MODEL.PART_ATTENTION', False)),
    'part_attention_parts': int(nested('MODEL.PART_ATTENTION_PARTS', 6)),
    'part_correspondence_consistency': bool(nested('MODEL.PART_CORRESPONDENCE_CONSISTENCY', False)),
    'pcc_parts': int(nested('MODEL.PCC_PARTS', 6)),
    'pcc_mode': str(nested('MODEL.PCC_MODE', 'fixed_index')),
}
if 'pcc_softmin_tau' in inspect.signature(Baseline.__init__).parameters:
    kwargs['pcc_softmin_tau'] = nested('MODEL.PCC_SOFTMIN_TAU', 0.1)
with torch.random.fork_rng(devices=[]):
    model = Baseline(
        1, int(nested('MODEL.LAST_STRIDE', 1)), '',
        str(nested('MODEL.NECK', 'bnneck')),
        str(nested('TEST.NECK_FEAT', 'after')),
        str(nested('MODEL.NAME')), 'none', **kwargs
    )
prefixes = ('base.', 'part_attention_head.', 'bottleneck.')
parameters = [
    {'name': name, 'shape': list(parameter.shape)}
    for name, parameter in model.named_parameters()
    if name.startswith(prefixes)
]
print('FEATURE_PARAMETER_SCHEMA=' + json.dumps({
    'parameters': parameters, 'descriptor_dim': int(model.in_planes)
}, sort_keys=True, separators=(',', ':')))
'''


def revision_feature_parameter_schema(repo_root, revision, configuration):
    """Instantiate a revision directly from a temporary Git archive zip."""
    resolved = _resolve_revision(repo_root, revision)
    with tempfile.TemporaryDirectory(prefix="bot-feature-revision-") as directory:
        archive = Path(directory) / "revision.zip"
        try:
            subprocess.check_call(
                [
                    "git", "-C", str(repo_root), "archive", "--format=zip",
                    "--output", str(archive), resolved,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            output = subprocess.check_output(
                [
                    sys.executable, "-c", _ARCHIVE_SCHEMA_SCRIPT,
                    str(archive), _canonical_json(configuration),
                ],
                cwd=directory,
                stderr=subprocess.STDOUT,
            ).decode("utf-8", errors="replace")
        except (OSError, subprocess.CalledProcessError) as error:
            details = getattr(error, "output", b"")
            if isinstance(details, bytes):
                details = details.decode("utf-8", errors="replace")
            raise FeatureCompatibilityError(
                "Cannot derive revision parameter schema: {} {}".format(
                    error, details
                )
            )
    marker = "FEATURE_PARAMETER_SCHEMA="
    rows = [line for line in output.splitlines() if line.startswith(marker)]
    if len(rows) != 1:
        raise FeatureCompatibilityError(
            "Revision parameter schema output is missing"
        )
    return json.loads(rows[0][len(marker):])


def _definition_components(configuration, parameter_payload):
    descriptor_dim = int(parameter_payload["descriptor_dim"])
    parts = int(_nested(configuration, "MODEL.PCC_PARTS", 6))
    attention_parts = int(_nested(
        configuration, "MODEL.PART_ATTENTION_PARTS", parts
    ))
    definitions = {
        "feature_parameter_schema": parameter_payload,
        "feature_protocol": {
            "backbone": str(_nested(configuration, "MODEL.NAME")),
            "last_stride": int(_nested(
                configuration, "MODEL.LAST_STRIDE", 1
            )),
            "part_attention": bool(_nested(
                configuration, "MODEL.PART_ATTENTION", False
            )),
            "part_attention_parts": attention_parts,
            "pcc_parts": parts,
            "part_order": "top_to_bottom",
            "partition": "horizontal_integer_nonoverlap_full_height",
            "pair_rule": "same_pid_different_camera_i_less_than_j",
            "distance": "raw_euclidean_l2",
        },
        "global_descriptor_definition": {
            "source": "shared_backbone_feature_map",
            "pooling": "AdaptiveAvgPool2d(1)",
            "shape": ["B", descriptor_dim],
        },
        "local_descriptor_definition": {
            "source": "shared_backbone_feature_map",
            "pooling": "adaptive_avg_pool2d(1)",
            "normalization": "none",
            "shape": ["B", parts, descriptor_dim],
        },
        "inference_descriptor_definition": {
            "neck": str(_nested(configuration, "MODEL.NECK", "bnneck")),
            "neck_feature": str(_nested(
                configuration, "TEST.NECK_FEAT", "after"
            )),
            "shape": ["B", descriptor_dim],
        },
    }
    return {
        name: _sha256_text(_canonical_json(value))
        for name, value in sorted(definitions.items())
    }


def revision_feature_signature(repo_root, revision, configuration,
                               source_provider=None,
                               parameter_schema_provider=None):
    resolved = _resolve_revision(repo_root, revision)
    source_hashes = revision_source_component_hashes(
        repo_root, resolved, source_provider=source_provider
    )
    if parameter_schema_provider is None:
        parameter_payload = revision_feature_parameter_schema(
            repo_root, resolved, configuration
        )
    else:
        parameter_payload = parameter_schema_provider(
            repo_root, resolved, configuration
        )
    components = dict(source_hashes)
    components.update(_definition_components(configuration, parameter_payload))
    signature = _sha256_text(_canonical_json(components))
    return {
        "commit": resolved,
        "signature_sha256": signature,
        "component_sha256": components,
    }


def build_feature_compatibility_evidence(
        repo_root, reference_revision, current_revision,
        reference_configuration, current_configuration,
        source_provider=None, parameter_schema_provider=None):
    reference = revision_feature_signature(
        repo_root, reference_revision, reference_configuration,
        source_provider=source_provider,
        parameter_schema_provider=parameter_schema_provider,
    )
    current = revision_feature_signature(
        repo_root, current_revision, current_configuration,
        source_provider=source_provider,
        parameter_schema_provider=parameter_schema_provider,
    )
    component_names = sorted(
        set(reference["component_sha256"]) | set(current["component_sha256"])
    )
    components = {}
    mismatches = []
    for name in component_names:
        reference_hash = reference["component_sha256"].get(name, NOT_RECORDED)
        current_hash = current["component_sha256"].get(name, NOT_RECORDED)
        status = "compatible" if reference_hash == current_hash else "mismatch"
        if status != "compatible":
            mismatches.append(name)
        components[name] = {
            "reference_sha256": reference_hash,
            "current_sha256": current_hash,
            "status": status,
        }
    compatible = (
        not mismatches
        and reference["signature_sha256"] == current["signature_sha256"]
    )
    return {
        "schema_version": 1,
        "feature_reference_commit": reference["commit"],
        "feature_current_commit": current["commit"],
        "feature_reference_signature_sha256": reference[
            "signature_sha256"
        ],
        "current_feature_signature_sha256": current["signature_sha256"],
        "feature_compatibility_status": (
            "compatible" if compatible else "incompatible"
        ),
        "mismatched_components": mismatches,
        "components": components,
        "source_access": "git_show_revision_objects",
        "parameter_schema_access": "git_archive_zip_revision_import",
    }


def require_feature_compatibility(evidence):
    if evidence.get("feature_compatibility_status") != "compatible":
        raise FeatureCompatibilityError(
            "Shared feature mismatch: {}".format(
                evidence.get("mismatched_components", [])
            )
        )
    return evidence


def _source_sha256(callable_object):
    source = textwrap.dedent(inspect.getsource(callable_object))
    return ast_component_sha256(source)


def _feature_implementation_hashes():
    from layers.part_correspondence_consistency import (
        build_cross_camera_positive_pairs,
        build_local_part_descriptors,
        horizontal_part_bounds,
        pairwise_local_distance_matrix,
        select_pair_local_features,
    )
    from modeling.baseline import Baseline, PartAttentionHead

    callables = {
        "baseline_forward": Baseline.forward,
        "part_attention_init": PartAttentionHead.__init__,
        "part_attention_forward": PartAttentionHead.forward,
        "horizontal_part_bounds": horizontal_part_bounds,
        "build_local_part_descriptors": build_local_part_descriptors,
        "build_cross_camera_positive_pairs": build_cross_camera_positive_pairs,
        "select_pair_local_features": select_pair_local_features,
        "pairwise_local_distance_matrix": pairwise_local_distance_matrix,
    }
    return {
        name: _source_sha256(callable_object)
        for name, callable_object in sorted(callables.items())
    }


def multigranular_feature_payload(configuration):
    parameters, descriptor_dim = _current_feature_parameter_schema(configuration)
    parts = int(_nested(configuration, "MODEL.PCC_PARTS", 6))
    attention_parts = int(_nested(
        configuration, "MODEL.PART_ATTENTION_PARTS", parts
    ))
    return {
        "signature_schema_version": 2,
        "backbone": {
            "name": str(_nested(configuration, "MODEL.NAME")),
            "last_stride": int(_nested(
                configuration, "MODEL.LAST_STRIDE", 1
            )),
            "shared_forward": "Baseline.base(x) exactly once",
        },
        "input": {
            "size_train": list(_nested(
                configuration, "INPUT.SIZE_TRAIN", []
            )),
            "size_test": list(_nested(
                configuration, "INPUT.SIZE_TEST", []
            )),
        },
        "part_attention": {
            "enabled": bool(_nested(
                configuration, "MODEL.PART_ATTENTION", False
            )),
            "parts": attention_parts,
            "part_order": "top_to_bottom",
            "fusion": "global_average_feature_plus_attention_part_feature",
        },
        "global_descriptor": {
            "source": "shared_backbone_feature_map",
            "pooling": "torch.nn.AdaptiveAvgPool2d(output_size=1)",
            "shape": ["B", descriptor_dim],
        },
        "local_descriptor": {
            "builder": (
                "layers.part_correspondence_consistency."
                "build_local_part_descriptors"
            ),
            "source": "same_shared_backbone_feature_map",
            "partition": (
                "horizontal_part_bounds_integer_nonoverlap_full_height"
            ),
            "part_order": "top_to_bottom",
            "parts": parts,
            "pooling": (
                "torch.nn.functional.adaptive_avg_pool2d(output_size=1)"
            ),
            "normalization": "none",
            "shape": ["B", parts, descriptor_dim],
        },
        "pair_selection": {
            "builder": "build_cross_camera_positive_pairs",
            "rule": "same_pid_and_different_camera_and_i_less_than_j",
        },
        "local_distance": {
            "builder": "pairwise_local_distance_matrix",
            "metric": "raw_euclidean_l2",
            "shape": ["N_pairs", parts, parts],
        },
        "inference_descriptor": {
            "neck": str(_nested(configuration, "MODEL.NECK", "bnneck")),
            "neck_feature": str(_nested(
                configuration, "TEST.NECK_FEAT", "after"
            )),
            "shape": ["B", descriptor_dim],
        },
        "feature_implementation_sha256": _feature_implementation_hashes(),
        "feature_parameter_schema": parameters,
    }


def canonical_multigranular_feature_signature(configuration):
    payload = multigranular_feature_payload(configuration)
    canonical = _canonical_json(payload)
    return canonical, _sha256_text(canonical)
