#!/usr/bin/env python
# encoding: utf-8
"""Fail-closed protocol and epoch-boundary validation for warmup20."""

from __future__ import absolute_import

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from layers import effective_pcc_lambda, make_loss
from layers.part_correspondence_consistency import soft_min_alignment_loss
from layers.triplet_loss import CrossCameraPositiveLoss, TripletLoss
from tools.run_experiment import _build_config_comparison
from utils.experiment_recording import atomic_write_json


BASELINE_BRANCH = (
    "origin/exp/c2l03-soft-min-alignment-lambda-sweep-tau0p2"
)
BASELINE_SHA = "67b7bbf528a0a6279a3f9ab86aed43ad91b1ef63"
BASELINE_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2l03_soft_min_alignment_tau0p2_"
    "lambda0p05_autodl.yml"
)
WARMUP_CONFIG = REPO_ROOT / "configs" / (
    "softmax_triplet_c2l03_soft_min_alignment_tau0p2_"
    "lambda0p05_warmup20_autodl.yml"
)
EXPECTED_DIFFERENCES = (
    "MODEL.PCC_WARMUP_EPOCHS",
    "OUTPUT_DIR",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the C2-L03 local-alignment warm-up protocol"
    )
    parser.add_argument("--config-file", default=str(WARMUP_CONFIG))
    parser.add_argument("--baseline-config", default=str(BASELINE_CONFIG))
    parser.add_argument("--output")
    return parser.parse_args(argv)


def _load(path):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(path))
    local_cfg.freeze()
    return local_cfg


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _git(*arguments):
    return subprocess.check_output(
        ["git"] + list(arguments), cwd=str(REPO_ROOT), text=True
    ).strip()


def _source_mapping(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _loss_at_epoch(local_cfg, epoch, tensors):
    score_source, feature_source, local_source, pids, camids = tensors
    score = score_source.detach().clone().requires_grad_(True)
    feature = feature_source.detach().clone().requires_grad_(True)
    local = local_source.detach().clone().requires_grad_(True)
    loss_fn = make_loss(local_cfg, 2)
    loss_fn.set_epoch(epoch)
    output = loss_fn(score, feature, pids, camids, local)
    base_total = (
        F.cross_entropy(score, pids)
        + TripletLoss(local_cfg.SOLVER.MARGIN)(feature, pids)[0]
        + float(local_cfg.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA)
        * CrossCameraPositiveLoss("mean")(feature, pids, camids)
    )
    alignment = soft_min_alignment_loss(
        local, pids, camids, float(local_cfg.MODEL.PCC_SOFTMIN_TAU)
    )[0]
    expected_effective = effective_pcc_lambda(
        local_cfg.MODEL.PCC_LAMBDA,
        local_cfg.MODEL.PCC_WARMUP_EPOCHS,
        epoch,
    )
    expected_total = base_total
    if expected_effective > 0.0:
        expected_total = expected_total + expected_effective * alignment
    _require(
        torch.allclose(output["loss_total"], expected_total, atol=1e-7),
        "total loss differs at epoch {}".format(epoch),
    )
    _require(
        float(output["pcc_effective_lambda"]) == expected_effective,
        "effective PCC lambda differs at epoch {}".format(epoch),
    )
    _require(
        float(output["pcc_configured_lambda"]) == 0.05,
        "configured PCC lambda changed",
    )
    _require(
        float(output["alignment_temperature"]) == 0.2,
        "alignment temperature changed",
    )
    output["loss_total"].backward()
    if expected_effective == 0.0:
        _require(
            local.grad is None or torch.count_nonzero(local.grad).item() == 0,
            "local alignment produced gradient while gated",
        )
    else:
        _require(
            local.grad is not None
            and torch.isfinite(local.grad).all()
            and torch.count_nonzero(local.grad).item() > 0,
            "local alignment gradient is missing after activation",
        )
    _require(
        score.grad is not None and torch.count_nonzero(score.grad).item() > 0,
        "ID loss gradient was disabled",
    )
    _require(
        feature.grad is not None
        and torch.count_nonzero(feature.grad).item() > 0,
        "triplet/cross-camera loss gradient was disabled",
    )
    return {
        "epoch": epoch,
        "configured_lambda": 0.05,
        "effective_lambda": expected_effective,
        "alignment_temperature": 0.2,
        "local_gradient_present": local.grad is not None,
        "other_loss_gradients_present": True,
    }


def main(argv=None):
    args = parse_args(argv)
    candidate_path = Path(args.config_file).resolve()
    baseline_path = Path(args.baseline_config).resolve()
    _require(candidate_path.is_file(), "candidate config is missing")
    _require(baseline_path.is_file(), "baseline config is missing")
    baseline_tip = _git("rev-parse", BASELINE_BRANCH)
    _require(baseline_tip == BASELINE_SHA, "remote-tracking baseline tip differs")
    merge_base = _git("merge-base", BASELINE_BRANCH, "HEAD")
    _require(merge_base == BASELINE_SHA, "HEAD is not based on baseline SHA")
    committed_baseline = _git(
        "show", "{}:{}".format(
            BASELINE_SHA,
            baseline_path.relative_to(REPO_ROOT).as_posix(),
        )
    )
    _require(
        baseline_path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        == committed_baseline.replace("\r\n", "\n").strip(),
        "working baseline config differs from the fixed remote commit",
    )

    baseline_cfg = _load(baseline_path)
    candidate_cfg = _load(candidate_path)
    comparison = _build_config_comparison(
        baseline_path, baseline_cfg, candidate_path, candidate_cfg,
        EXPECTED_DIFFERENCES,
    )
    source = _source_mapping(candidate_path)
    _require(source["SEED"] == 42, "seed changed")
    _require(source["DATASETS"]["NAMES"] == "market1501", "dataset changed")
    _require(source["MODEL"]["PCC_PARTS"] == 6, "K changed")
    _require(source["MODEL"]["PCC_LAMBDA"] == 0.05, "lambda_p changed")
    _require(source["MODEL"]["PCC_SOFTMIN_TAU"] == 0.2, "tau_a changed")
    _require(source["MODEL"]["PCC_WARMUP_EPOCHS"] == 20, "warm-up changed")
    _require(
        source["MODEL"]["CROSS_CAMERA_POSITIVE_LAMBDA"] == 0.3,
        "cross-camera positive lambda changed",
    )
    _require(source["SOLVER"]["MAX_EPOCHS"] == 120, "epoch count changed")
    _require(
        source["SOLVER"]["WARMUP_ITERS"]
        == _source_mapping(baseline_path)["SOLVER"]["WARMUP_ITERS"],
        "learning-rate warm-up changed",
    )

    torch.manual_seed(20260824)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = (
        torch.randn(4, 2, device=device),
        torch.randn(4, 16, device=device),
        torch.randn(4, 6, 16, device=device),
        torch.tensor([0, 0, 1, 1], device=device),
        torch.tensor([0, 1, 0, 1], device=device),
    )
    candidate_for_loss = candidate_cfg.clone()
    candidate_for_loss.defrost()
    candidate_for_loss.MODEL.IF_LABELSMOOTH = "off"
    candidate_for_loss.freeze()
    boundaries = [
        _loss_at_epoch(candidate_for_loss, epoch, tensors)
        for epoch in (1, 20, 21, 120)
    ]

    baseline_for_loss = baseline_cfg.clone()
    baseline_for_loss.defrost()
    baseline_for_loss.MODEL.IF_LABELSMOOTH = "off"
    baseline_for_loss.freeze()
    default_behavior = _loss_at_epoch(baseline_for_loss, 1, tensors)
    _require(
        int(baseline_for_loss.MODEL.PCC_WARMUP_EPOCHS) == 0
        and default_behavior["effective_lambda"] == 0.05,
        "default warm-up=0 did not preserve immediate activation",
    )

    report = {
        "status": "pass",
        "baseline_branch": BASELINE_BRANCH,
        "baseline_sha": BASELINE_SHA,
        "head": _git("rev-parse", "HEAD"),
        "merge_base": merge_base,
        "device": str(device),
        "config_comparison": comparison,
        "boundary_evidence": boundaries,
        "default_warmup_zero_evidence": default_behavior,
        "other_training_variables_unchanged": True,
    }
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
