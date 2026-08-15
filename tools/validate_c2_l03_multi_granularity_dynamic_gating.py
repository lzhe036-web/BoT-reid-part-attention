#!/usr/bin/env python
"""Static validator for the per-sample Dynamic Gating experiment."""

from __future__ import absolute_import

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modeling import build_model
from modeling.baseline import Baseline
from utils.experiment_recording import model_state_dict_schema, validate_dynamic_configuration
from utils.multigranularity_signatures import (
    DYNAMIC_CONFIG_PATH,
    STATIC_BASELINE_SHA,
    STATIC_CONFIG_PATH,
    build_feature_compatibility_evidence,
    require_feature_compatibility,
)


def _configuration(path):
    from config import cfg
    configuration = cfg.clone()
    configuration.merge_from_file(str(path))
    configuration.freeze()
    return configuration


def _shared_schema(model):
    return {
        key: value for key, value in model_state_dict_schema(dict(model.state_dict())).items()
        if "multi_granularity_dynamic_gate" not in key
        and not key.startswith("classifier.")
    }


def _soft_alignment_comparison_status():
    branch = "exp/c2l03-soft-min-alignment"
    config_path = "configs/softmax_triplet_c2l03_soft_min_alignment_autodl.yml"
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", branch],
            stderr=subprocess.PIPE,
        ).decode("utf-8", errors="replace").strip()
        source = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show", "{}:{}".format(commit, config_path)],
            stderr=subprocess.PIPE,
        ).decode("utf-8", errors="replace")
        configuration = yaml.safe_load(source)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {
            "status": "not_recorded",
            "reason": "Soft Alignment branch/config is unavailable",
        }
    model = configuration.get("MODEL", {})
    reasons = []
    if configuration.get("SEED") != 42:
        reasons.append("SEED is not 42")
    if model.get("MULTI_GRANULARITY_PART_SCALES") != [2, 4, 6]:
        reasons.append("Soft Alignment does not use K=[2,4,6]")
    if model.get("MULTI_GRANULARITY_PART") is not True:
        reasons.append("Soft Alignment does not enable the shared multi-granularity pipeline")
    if model.get("PART_ATTENTION") is not False:
        reasons.append("Soft Alignment uses a different Part Attention feature path")
    return {
        "branch": branch, "commit": commit,
        "status": "compatible" if not reasons else "incompatible",
        "reasons": reasons,
        "comparative_formal_allowed": not reasons,
    }


def run_validation(device="cpu"):
    static_path = REPO_ROOT / STATIC_CONFIG_PATH
    dynamic_path = REPO_ROOT / DYNAMIC_CONFIG_PATH
    static_source = yaml.safe_load(static_path.read_text(encoding="utf-8"))
    dynamic_source = yaml.safe_load(dynamic_path.read_text(encoding="utf-8"))
    differences = validate_dynamic_configuration(dynamic_source, static_source)
    static_cfg = _configuration(static_path)
    dynamic_cfg = _configuration(dynamic_path)
    for configuration in (static_cfg, dynamic_cfg):
        configuration.defrost()
        configuration.MODEL.PRETRAIN_CHOICE = "none"
        configuration.MODEL.PRETRAIN_PATH = ""
        configuration.MODEL.DEVICE = device
        configuration.freeze()
    static_model = build_model(static_cfg, num_classes=17).to(device)
    dynamic_model = build_model(dynamic_cfg, num_classes=17).to(device)
    dynamic_model.load_state_dict(static_model.state_dict(), strict=False)
    if _shared_schema(static_model) != _shared_schema(dynamic_model):
        raise RuntimeError("Static/Dynamic shared parameter schemas differ")
    values = torch.randn(3, 3, 256, 128, device=device)
    static_model.eval()
    dynamic_model.eval()
    with torch.no_grad():
        static_descriptor = static_model(values)
        dynamic_descriptor = dynamic_model(values)
    if tuple(static_descriptor.shape) != (3, 2816):
        raise RuntimeError("Static descriptor shape changed")
    if tuple(dynamic_descriptor.shape) != (3, 2816):
        raise RuntimeError("Dynamic descriptor shape changed")
    if not torch.allclose(static_descriptor, dynamic_descriptor, rtol=1e-6, atol=1e-7):
        raise RuntimeError("Zero-initialized Dynamic descriptor differs from Static")
    probabilities = dynamic_model._last_dynamic_gating["probabilities"]
    weights = dynamic_model._last_dynamic_gating["weights"]
    if not torch.allclose(probabilities.sum(1), torch.ones(3, device=probabilities.device)):
        raise RuntimeError("Gating probabilities do not sum to one")
    if not torch.allclose(weights.sum(1), torch.full((3,), 3.0, device=weights.device)):
        raise RuntimeError("Applied gating weights do not sum to three")
    dynamic_model.train()
    score, descriptor = dynamic_model(values)
    (score.square().mean() + descriptor.square().mean()).backward()
    gate = dynamic_model.multi_granularity_dynamic_gate.controller
    if gate.weight.grad is None or gate.bias.grad is None:
        raise RuntimeError("Gating controller did not receive gradients")
    if not torch.isfinite(gate.weight.grad).all() or not torch.isfinite(gate.bias.grad).all():
        raise RuntimeError("Gating controller gradients are non-finite")

    current_sources = {
        path: (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "modeling/baseline.py", "modeling/backbones/resnet.py",
            "layers/triplet_loss.py",
        )
    }
    current_schema = _shared_schema(dynamic_model)
    evidence = require_feature_compatibility(
        build_feature_compatibility_evidence(
            REPO_ROOT, STATIC_BASELINE_SHA, STATIC_BASELINE_SHA,
            current_source_overrides=current_sources,
            current_config_override=dynamic_source,
            current_parameter_schema=current_schema,
            python_executable=sys.executable,
        )
    )
    result = {
        "status": "passed", "device": device, "seed": 42,
        "static_descriptor_shape": list(static_descriptor.shape),
        "dynamic_descriptor_shape": list(dynamic_descriptor.shape),
        "formal_config_differences": differences,
        "shared_feature_signature_sha256": evidence["current_feature_signature_sha256"],
        "reference_feature_signature_sha256": evidence["feature_reference_signature_sha256"],
        "gating_signature_sha256": evidence["fusion_gating_signature"]["current_sha256"],
        "feature_compatibility_status": evidence["feature_compatibility_status"],
        "validation_scope": "worktree validation; not an experiment run",
        "soft_alignment_cross_comparison": _soft_alignment_comparison_status(),
    }
    return result


def main():
    results = [run_validation("cpu")]
    if torch.cuda.is_available():
        results.append(run_validation("cuda"))
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
