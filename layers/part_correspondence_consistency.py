# encoding: utf-8
"""Cross-camera local-part alignment primitives."""

from __future__ import absolute_import

import math

import torch
import torch.nn.functional as F


SUPPORTED_ALIGNMENT_MODES = ("fixed_index", "hard_shortest_path", "soft_min")


def validate_softmin_tau(tau):
    """Return a finite positive Soft-Min temperature or fail closed."""
    if isinstance(tau, bool):
        raise ValueError("PCC_SOFTMIN_TAU must be a finite positive number")
    try:
        value = float(tau)
    except (TypeError, ValueError):
        raise ValueError("PCC_SOFTMIN_TAU must be a finite positive number")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("PCC_SOFTMIN_TAU must be finite and greater than zero")
    return value


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


def _validate_square_distance_matrix(distance_matrix):
    if distance_matrix.dim() != 3:
        raise ValueError("distance_matrix must have shape [N, K, K]")
    if distance_matrix.size(1) != distance_matrix.size(2):
        raise ValueError("distance_matrix must be square over its part axes")
    if distance_matrix.size(1) <= 0:
        raise ValueError("distance_matrix must contain at least one part")


def _hard_shortest_path_dynamic_programming(distance_matrix):
    """Return raw hard costs and the exact up-first predecessor decisions.

    The accumulated costs are built without in-place tensor writes, preserving
    autograd.  At ties, ``up <= left`` deterministically selects ``up``.
    """
    _validate_square_distance_matrix(distance_matrix)
    num_parts = int(distance_matrix.size(1))
    accumulated = [[None for _ in range(num_parts)] for _ in range(num_parts)]
    predecessor_is_up = [
        [None for _ in range(num_parts)] for _ in range(num_parts)
    ]
    accumulated[0][0] = distance_matrix[:, 0, 0]
    for row in range(1, num_parts):
        accumulated[row][0] = (
            distance_matrix[:, row, 0] + accumulated[row - 1][0]
        )
    for column in range(1, num_parts):
        accumulated[0][column] = (
            distance_matrix[:, 0, column] + accumulated[0][column - 1]
        )
    for row in range(1, num_parts):
        for column in range(1, num_parts):
            up = accumulated[row - 1][column]
            left = accumulated[row][column - 1]
            choose_up = up <= left
            predecessor_is_up[row][column] = choose_up.detach()
            accumulated[row][column] = distance_matrix[:, row, column] + (
                torch.where(choose_up, up, left)
            )
    return accumulated[-1][-1], predecessor_is_up


def _path_absolute_offsets(distance_matrix, predecessor_is_up):
    """Backtrack observation-only path offsets using DP's saved decisions."""
    num_pairs = int(distance_matrix.size(0))
    num_parts = int(distance_matrix.size(1))
    path_length = 2 * num_parts - 1
    false_choices = torch.zeros(
        num_pairs, dtype=torch.bool, device=distance_matrix.device
    )
    detached_choices = torch.stack([
        torch.stack([
            predecessor_is_up[row][column]
            if predecessor_is_up[row][column] is not None else false_choices
            for column in range(num_parts)
        ], dim=1)
        for row in range(num_parts)
    ], dim=1).cpu()
    offsets = []
    for pair_index in range(num_pairs):
        row = num_parts - 1
        column = num_parts - 1
        absolute_offset_sum = 0
        while True:
            absolute_offset_sum += abs(row - column)
            if row == 0 and column == 0:
                break
            if row == 0:
                column -= 1
            elif column == 0:
                row -= 1
            elif bool(detached_choices[pair_index, row, column]):
                row -= 1
            else:
                column -= 1
        offsets.append(float(absolute_offset_sum) / float(path_length))
    return distance_matrix.new_tensor(offsets)


def hard_shortest_path_costs(distance_matrix):
    """Return unnormalized monotonic path costs shaped ``[N]``.

    Paths start at ``(0, 0)``, end at ``(K-1, K-1)``, and permit only down
    and right moves.  Diagonal moves are never considered.
    """
    costs, _predecessor_is_up = _hard_shortest_path_dynamic_programming(
        distance_matrix
    )
    return costs


def hard_shortest_path_cost(distance_matrix):
    """Return the raw path cost for one ``[K, K]`` distance matrix."""
    if distance_matrix.dim() != 2:
        raise ValueError("distance_matrix must have shape [K, K]")
    return hard_shortest_path_costs(distance_matrix.unsqueeze(0))[0]


def hard_shortest_path_costs_and_offsets(distance_matrix):
    """Return raw costs and observation-only mean absolute path offsets."""
    costs, predecessor_is_up = _hard_shortest_path_dynamic_programming(
        distance_matrix
    )
    offsets = _path_absolute_offsets(distance_matrix, predecessor_is_up)
    return costs, offsets


def softmin_two_predecessors(up, left, tau):
    """Numerically stable differentiable minimum of two predecessor costs."""
    tau_value = validate_softmin_tau(tau)
    tau_tensor = up.new_tensor(tau_value)
    scaled = torch.stack((-up / tau_tensor, -left / tau_tensor), dim=0)
    return -tau_tensor * torch.logsumexp(scaled, dim=0)


def soft_min_path_costs(distance_matrix, tau):
    """Return Soft-Min right/down path costs shaped ``[N]``."""
    _validate_square_distance_matrix(distance_matrix)
    tau_value = validate_softmin_tau(tau)
    num_parts = int(distance_matrix.size(1))
    accumulated = [[None for _ in range(num_parts)] for _ in range(num_parts)]
    accumulated[0][0] = distance_matrix[:, 0, 0]
    for row in range(1, num_parts):
        accumulated[row][0] = (
            distance_matrix[:, row, 0] + accumulated[row - 1][0]
        )
    for column in range(1, num_parts):
        accumulated[0][column] = (
            distance_matrix[:, 0, column] + accumulated[0][column - 1]
        )
    for row in range(1, num_parts):
        for column in range(1, num_parts):
            accumulated[row][column] = distance_matrix[:, row, column] + (
                softmin_two_predecessors(
                    accumulated[row - 1][column],
                    accumulated[row][column - 1],
                    tau_value,
                )
            )
    return accumulated[-1][-1]


def soft_min_path_cost(distance_matrix, tau):
    """Return the Soft-Min cost for one ``[K, K]`` distance matrix."""
    if distance_matrix.dim() != 2:
        raise ValueError("distance_matrix must have shape [K, K]")
    return soft_min_path_costs(distance_matrix.unsqueeze(0), tau)[0]


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


def hard_shortest_path_alignment_loss(local_features, pids, camids):
    """Compute normalized hard alignment over valid cross-camera pairs."""
    pair_indices = build_cross_camera_positive_pairs(pids, camids)
    pair_count = int(pair_indices.size(0))
    if pair_count == 0:
        zero = local_features.sum() * 0.0
        return zero, pair_count, zero, zero
    local_a, local_b = select_pair_local_features(local_features, pair_indices)
    distance_matrix = pairwise_local_distance_matrix(local_a, local_b)
    raw_costs, path_offsets = hard_shortest_path_costs_and_offsets(
        distance_matrix
    )
    path_length = 2 * int(distance_matrix.size(1)) - 1
    hard_alignment_loss = (raw_costs / float(path_length)).mean()
    return (
        hard_alignment_loss,
        pair_count,
        raw_costs.detach().mean(),
        path_offsets.mean(),
    )


def soft_min_alignment_loss(local_features, pids, camids, tau):
    """Compute normalized Soft-Min alignment over valid camera pairs."""
    tau_value = validate_softmin_tau(tau)
    pair_indices = build_cross_camera_positive_pairs(pids, camids)
    pair_count = int(pair_indices.size(0))
    if pair_count == 0:
        zero = local_features.sum() * 0.0
        return zero, pair_count, zero
    local_a, local_b = select_pair_local_features(local_features, pair_indices)
    distance_matrix = pairwise_local_distance_matrix(local_a, local_b)
    raw_costs = soft_min_path_costs(distance_matrix, tau_value)
    path_length = 2 * int(distance_matrix.size(1)) - 1
    soft_alignment_loss = (raw_costs / float(path_length)).mean()
    return soft_alignment_loss, pair_count, raw_costs.detach().mean()


def part_alignment_loss(local_features, pids, camids, mode,
                        softmin_tau=None):
    """Dispatch a supported alignment mode and expose compatible statistics."""
    zero = local_features.sum() * 0.0
    if mode == "fixed_index":
        loss, pair_count, mean_distance = fixed_index_pcc_loss(
            local_features, pids, camids
        )
        return {
            "loss_pcc": loss,
            "valid_pcc_pair_count": pair_count,
            "mean_fixed_index_part_distance": mean_distance,
            "hard_alignment_loss": zero,
            "valid_alignment_pair_count": 0,
            "mean_hard_path_cost": zero,
            "mean_path_absolute_offset": zero,
            "soft_alignment_loss": zero,
            "mean_soft_path_cost": zero,
        }
    if mode == "hard_shortest_path":
        loss, pair_count, mean_cost, mean_offset = (
            hard_shortest_path_alignment_loss(local_features, pids, camids)
        )
        return {
            "loss_pcc": loss,
            "valid_pcc_pair_count": pair_count,
            "mean_fixed_index_part_distance": zero,
            "hard_alignment_loss": loss,
            "valid_alignment_pair_count": pair_count,
            "mean_hard_path_cost": mean_cost,
            "mean_path_absolute_offset": mean_offset,
            "soft_alignment_loss": zero,
            "mean_soft_path_cost": zero,
        }
    if mode == "soft_min":
        loss, pair_count, mean_cost = soft_min_alignment_loss(
            local_features, pids, camids, softmin_tau
        )
        return {
            "loss_pcc": loss,
            "valid_pcc_pair_count": pair_count,
            "mean_fixed_index_part_distance": zero,
            "hard_alignment_loss": zero,
            "valid_alignment_pair_count": pair_count,
            "mean_hard_path_cost": zero,
            "mean_path_absolute_offset": zero,
            "soft_alignment_loss": loss,
            "mean_soft_path_cost": mean_cost,
        }
    raise ValueError(
        "Unsupported PCC_MODE {!r}; expected one of {}".format(
            mode, ", ".join(SUPPORTED_ALIGNMENT_MODES)
        )
    )
