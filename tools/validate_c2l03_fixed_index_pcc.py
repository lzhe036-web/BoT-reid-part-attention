#!/usr/bin/env python
# encoding: utf-8
"""Synthetic shape and gradient validation for Fixed-Index PCC."""

from __future__ import absolute_import

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.part_correspondence_consistency import (
    build_cross_camera_positive_pairs,
    fixed_index_distances,
    fixed_index_pcc_loss,
    pairwise_local_distance_matrix,
    select_pair_local_features,
)
from layers.triplet_loss import (
    CrossCameraPositiveLoss,
    CrossEntropyLabelSmooth,
    TripletLoss,
)
from modeling.baseline import Baseline


def main():
    torch.manual_seed(20260807)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Baseline(
        2, 1, '', 'bnneck', 'after', 'resnet50', 'none',
        part_attention=True, part_attention_parts=6,
        part_correspondence_consistency=True, pcc_parts=6,
    ).to(device)
    images = torch.randn(4, 3, 256, 128, device=device)
    pids = torch.tensor([0, 0, 1, 1], device=device)
    camids = torch.tensor([0, 1, 0, 1], device=device)
    feature_shapes = []
    backbone_calls = []

    def capture_backbone(_module, _inputs, output):
        backbone_calls.append(1)
        feature_shapes.append(list(output.shape))

    hook = model.base.register_forward_hook(capture_backbone)
    model.train()
    cls_score, global_feat, local_features = model(images)
    hook.remove()
    if len(backbone_calls) != 1:
        raise AssertionError('the training forward must call the backbone exactly once')

    pairs = build_cross_camera_positive_pairs(pids, camids)
    local_a, local_b = select_pair_local_features(local_features, pairs)
    distance_matrix = pairwise_local_distance_matrix(local_a, local_b)
    diagonal = distance_matrix.diagonal(dim1=1, dim2=2)
    loss_pcc, pair_count, _ = fixed_index_pcc_loss(local_features, pids, camids)
    loss_id = CrossEntropyLabelSmooth(
        num_classes=2, use_gpu=device.type == 'cuda'
    )(cls_score, pids)
    loss_triplet = TripletLoss(0.3)(global_feat, pids)[0]
    loss_cross_camera = CrossCameraPositiveLoss('mean')(
        global_feat, pids, camids
    )
    loss_total = (
        loss_id + loss_triplet + 0.3 * loss_cross_camera + 0.1 * loss_pcc
    )
    loss_total.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise AssertionError('model gradients contain NaN or Inf')

    model.eval()
    with torch.no_grad():
        inference = model(images)
    if inference.shape != (4, 2048):
        raise AssertionError('PCC changed the inference descriptor dimension')
    if not torch.allclose(
            fixed_index_distances(distance_matrix), diagonal.mean(dim=1)):
        raise AssertionError('fixed-index loss did not use the matrix diagonal')

    report = {
        'device': str(device),
        'backbone_forward_calls_training': len(backbone_calls),
        'backbone_feature_map': feature_shapes[0],
        'pcc_local_features': list(local_features.shape),
        'valid_pair_count': pair_count,
        'pair_local_features_a': list(local_a.shape),
        'pair_local_features_b': list(local_b.shape),
        'local_distance_matrix': list(distance_matrix.shape),
        'diagonal_distances': list(diagonal.shape),
        'loss_pcc_shape': list(loss_pcc.shape),
        'loss_total_shape': list(loss_total.shape),
        'loss_id': float(loss_id.detach().item()),
        'loss_triplet': float(loss_triplet.detach().item()),
        'loss_cross_camera_positive': float(
            loss_cross_camera.detach().item()
        ),
        'loss_pcc': float(loss_pcc.detach().item()),
        'loss_total': float(loss_total.detach().item()),
        'loss_pcc_finite': bool(torch.isfinite(loss_pcc).item()),
        'loss_total_finite': bool(torch.isfinite(loss_total).item()),
        'finite_gradients': finite_gradients,
        'inference_feature': list(inference.shape),
        'inference_descriptor_dim': int(inference.size(1)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
