# encoding: utf-8
"""Attention-free multi-granularity local features from one feature map."""

from __future__ import absolute_import

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


def horizontal_part_bounds(height, scale):
    """Return non-overlapping bounds that cover every feature-map row."""
    height = int(height)
    scale = int(scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if height < scale:
        raise ValueError(
            "feature-map height {} is smaller than scale {}".format(
                height, scale
            )
        )
    bounds = [
        (height * part_index // scale,
         height * (part_index + 1) // scale)
        for part_index in range(scale)
    ]
    if bounds[0][0] != 0 or bounds[-1][1] != height:
        raise RuntimeError("horizontal partition does not cover full height")
    if any(start >= end for start, end in bounds):
        raise RuntimeError("horizontal partition contains an empty part")
    if any(bounds[index][1] != bounds[index + 1][0]
           for index in range(len(bounds) - 1)):
        raise RuntimeError("horizontal partition contains a gap or overlap")
    return bounds


class MultiGranularityLocalFeature(nn.Module):
    """Pool K horizontal parts and arithmetic-mean them within each scale.

    Each scale owns one shared linear projection.  There are deliberately no
    attention, alignment, gating, or learnable part-weight components.
    """

    def __init__(self, in_channels, scales=(2, 4, 6), projection_dim=256,
                 aggregation="mean"):
        super(MultiGranularityLocalFeature, self).__init__()
        self.in_channels = int(in_channels)
        self.scales = tuple(int(scale) for scale in scales)
        self.projection_dim = int(projection_dim)
        self.aggregation = str(aggregation).lower()
        if not self.scales or len(set(self.scales)) != len(self.scales):
            raise ValueError("scales must be a non-empty set of unique values")
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("all scales must be positive")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if self.aggregation != "mean":
            raise ValueError(
                "only arithmetic mean aggregation is supported, got {}"
                .format(aggregation)
            )
        self.projections = nn.ModuleDict()
        for scale in self.scales:
            projection = nn.Linear(
                self.in_channels, self.projection_dim, bias=True
            )
            nn.init.kaiming_normal_(projection.weight, a=0, mode="fan_out")
            nn.init.constant_(projection.bias, 0.0)
            self.projections[str(scale)] = projection

    @property
    def output_dim(self):
        return len(self.scales) * self.projection_dim

    def forward(self, feature_map, return_details=False):
        if feature_map.dim() != 4:
            raise ValueError("feature_map must have shape [N, C, H, W]")
        if int(feature_map.size(1)) != self.in_channels:
            raise ValueError(
                "expected {} channels, got {}".format(
                    self.in_channels, feature_map.size(1)
                )
            )
        aggregated_features = []
        details = OrderedDict()
        height = int(feature_map.size(2))
        for scale in self.scales:
            bounds = horizontal_part_bounds(height, scale)
            part_features = []
            for start, end in bounds:
                region = feature_map[:, :, start:end, :]
                pooled = F.adaptive_avg_pool2d(region, 1).flatten(1)
                projected = self.projections[str(scale)](pooled)
                part_features.append(projected)
            stacked = torch.stack(part_features, dim=1)
            aggregated = stacked.mean(dim=1)
            aggregated_features.append(aggregated)
            if return_details:
                details[scale] = {
                    "bounds": bounds,
                    "part_features": tuple(part_features),
                    "aggregated": aggregated,
                }
        if return_details:
            return tuple(aggregated_features), details
        return tuple(aggregated_features)
