# encoding: utf-8
"""Fixed-index cross-camera part correspondence consistency primitives."""

from __future__ import absolute_import

import torch
import torch.nn.functional as F


def horizontal_part_bounds(height, num_parts):
    """Return non-overlapping integer bounds that cover the full height."""
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    if isinstance(num_parts, bool) or not isinstance(num_parts, int) or num_parts <= 0:
        raise ValueError("num_parts must be a positive integer")
    if height < num_parts:
        raise ValueError(
            "feature-map height {} cannot form {} non-empty parts".format(
                height, num_parts
            )
        )
    bounds = [
        (part_index * height // num_parts,
         (part_index + 1) * height // num_parts)
        for part_index in range(num_parts)
    ]
    if bounds[0][0] != 0 or bounds[-1][1] != height:
        raise RuntimeError("horizontal part bounds do not cover the full height")
    if any(start >= end for start, end in bounds):
        raise RuntimeError("horizontal part bounds contain an empty part")
    if any(bounds[index][1] != bounds[index + 1][0]
           for index in range(len(bounds) - 1)):
        raise RuntimeError("horizontal part bounds contain a gap or overlap")
    return bounds


def build_local_part_descriptors(feature_map, num_parts=6):
    """Pool horizontal regions into descriptors shaped ``[B, K, C]``."""
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape [B, C, H, W]")
    bounds = horizontal_part_bounds(int(feature_map.size(2)), int(num_parts))
    descriptors = []
    for start, end in bounds:
        region = feature_map[:, :, start:end, :]
        descriptors.append(F.adaptive_avg_pool2d(region, 1).flatten(1))
    return torch.stack(descriptors, dim=1)


def build_cross_camera_positive_pairs(pids, camids):
    """Build unique unordered ``i < j`` same-PID, different-camera pairs."""
    if pids is None or camids is None:
        raise ValueError("pids and camids are required for PCC")
    pids = pids.reshape(-1)
    camids = camids.reshape(-1)
    if pids.size(0) != camids.size(0):
        raise ValueError("pids and camids must have the same batch length")
    same_pid = pids[:, None].eq(pids[None, :])
    different_camera = camids[:, None].ne(camids[None, :])
    unique_upper_triangle = torch.ones_like(same_pid, dtype=torch.bool).triu(1)
    return (same_pid & different_camera & unique_upper_triangle).nonzero(
        as_tuple=False
    )


def select_pair_local_features(local_features, pair_indices):
    """Select the two local-descriptor tensors for each pair."""
    if local_features.dim() != 3:
        raise ValueError("local_features must have shape [B, K, C]")
    if pair_indices.dim() != 2 or pair_indices.size(1) != 2:
        raise ValueError("pair_indices must have shape [N_pairs, 2]")
    return local_features[pair_indices[:, 0]], local_features[pair_indices[:, 1]]


def pairwise_local_distance_matrix(local_a, local_b):
    """Return all Euclidean part distances with shape ``[N, K, K]``."""
    if local_a.dim() != 3 or local_b.dim() != 3:
        raise ValueError("local_a and local_b must have shape [N, K, C]")
    if local_a.size(0) != local_b.size(0):
        raise ValueError("local pair batches must have equal length")
    if local_a.size(2) != local_b.size(2):
        raise ValueError("local descriptor dimensions must match")
    differences = local_a.unsqueeze(2) - local_b.unsqueeze(1)
    return torch.linalg.vector_norm(differences, ord=2, dim=-1)


def fixed_index_distances(distance_matrix):
    """Average only same-index (main-diagonal) part distances per pair."""
    if distance_matrix.dim() != 3:
        raise ValueError("distance_matrix must have shape [N, K, K]")
    if distance_matrix.size(1) != distance_matrix.size(2):
        raise ValueError("distance_matrix must be square over its part axes")
    return distance_matrix.diagonal(dim1=1, dim2=2).mean(dim=1)


def fixed_index_pcc_loss(local_features, pids, camids):
    """Compute PCC over unique same-PID, different-camera pairs."""
    pair_indices = build_cross_camera_positive_pairs(pids, camids)
    pair_count = int(pair_indices.size(0))
    if pair_count == 0:
        zero = local_features.sum() * 0.0
        return zero, pair_count, zero
    local_a, local_b = select_pair_local_features(local_features, pair_indices)
    distance_matrix = pairwise_local_distance_matrix(local_a, local_b)
    per_pair_distances = fixed_index_distances(distance_matrix)
    loss = per_pair_distances.mean()
    return loss, pair_count, loss
