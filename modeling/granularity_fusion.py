# encoding: utf-8
"""Static and per-sample dynamic fusion of ReID granularity components."""

from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F


GRANULARITY_LABELS = ("global", "k2", "k4", "k6")


def fusion_parameter_counts(global_dim=2048, fusion_dim=256,
                            hidden_dim=256, component_count=4):
    """Return the trainable parameter counts for both fusion alternatives."""
    global_dim = int(global_dim)
    fusion_dim = int(fusion_dim)
    hidden_dim = int(hidden_dim)
    component_count = int(component_count)
    global_projection = global_dim * fusion_dim + fusion_dim
    static = global_projection + component_count
    gating_input_dim = component_count * fusion_dim
    dynamic_mlp = (
        gating_input_dim * hidden_dim + hidden_dim
        + hidden_dim * component_count + component_count
    )
    return {"static": static, "dynamic": global_projection + dynamic_mlp}


class GranularityFusion(nn.Module):
    """Fuse Global/K2/K4/K6 into one descriptor of ``fusion_dim``.

    Static mode learns one sample-independent set of four logits. Dynamic mode
    predicts four logits from the sample's four feature components. Neither
    path accepts identity, camera, or label metadata.
    """

    def __init__(self, global_dim=2048, fusion_dim=256, hidden_dim=256,
                 mode="static", component_count=4):
        super(GranularityFusion, self).__init__()
        self.global_dim = int(global_dim)
        self.fusion_dim = int(fusion_dim)
        self.hidden_dim = int(hidden_dim)
        self.mode = str(mode).lower()
        self.component_count = int(component_count)
        if self.fusion_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("fusion and gating hidden dimensions must be positive")
        if self.component_count != len(GRANULARITY_LABELS):
            raise ValueError("Global/K2/K4/K6 fusion requires four components")
        if self.mode not in ("static", "dynamic"):
            raise ValueError("fusion mode must be 'static' or 'dynamic'")

        self.global_projection = nn.Linear(
            self.global_dim, self.fusion_dim, bias=True
        )
        nn.init.kaiming_normal_(
            self.global_projection.weight, a=0, mode="fan_out"
        )
        nn.init.constant_(self.global_projection.bias, 0.0)

        if self.mode == "static":
            self.shared_logits = nn.Parameter(torch.zeros(self.component_count))
            self.gating_mlp = None
        else:
            self.register_parameter("shared_logits", None)
            self.gating_mlp = nn.Sequential(
                nn.Linear(
                    self.component_count * self.fusion_dim,
                    self.hidden_dim,
                ),
                nn.ReLU(inplace=False),
                nn.Linear(self.hidden_dim, self.component_count),
            )
            nn.init.kaiming_normal_(
                self.gating_mlp[0].weight, a=0, mode="fan_out"
            )
            nn.init.constant_(self.gating_mlp[0].bias, 0.0)
            # Uniform initial weights make Dynamic exactly match zero-logit
            # Static fusion when both receive the same four components.
            nn.init.constant_(self.gating_mlp[2].weight, 0.0)
            nn.init.constant_(self.gating_mlp[2].bias, 0.0)

    def build_components(self, global_feature, local_features):
        if global_feature.dim() != 2:
            raise ValueError("global_feature must have shape [B, global_dim]")
        if int(global_feature.size(1)) != self.global_dim:
            raise ValueError(
                "expected global dimension {}, got {}".format(
                    self.global_dim, global_feature.size(1)
                )
            )
        local_features = tuple(local_features)
        if len(local_features) != self.component_count - 1:
            raise ValueError("expected K2/K4/K6 local feature components")
        projected_global = self.global_projection(global_feature)
        components = (projected_global,) + local_features
        batch_size = int(global_feature.size(0))
        for label, component in zip(GRANULARITY_LABELS, components):
            if tuple(component.shape) != (batch_size, self.fusion_dim):
                raise ValueError(
                    "{} component must have shape [{}, {}], got {}".format(
                        label, batch_size, self.fusion_dim,
                        tuple(component.shape),
                    )
                )
        return torch.stack(components, dim=1)

    def weights_from_components(self, components):
        if components.dim() != 3:
            raise ValueError(
                "components must have shape [B, {}, {}]".format(
                    self.component_count, self.fusion_dim
                )
            )
        expected = (
            int(components.size(0)), self.component_count, self.fusion_dim
        )
        if tuple(components.shape) != expected:
            raise ValueError(
                "components must have shape [B, {}, {}]".format(
                    self.component_count, self.fusion_dim
                )
            )
        if self.mode == "static":
            weights = F.softmax(self.shared_logits, dim=0)
            return weights.unsqueeze(0).expand(components.size(0), -1)
        logits = self.gating_mlp(components.reshape(components.size(0), -1))
        return F.softmax(logits, dim=1)

    def forward(self, global_feature, local_features, return_details=False):
        components = self.build_components(global_feature, local_features)
        weights = self.weights_from_components(components)
        fused = torch.sum(weights.unsqueeze(-1) * components, dim=1)
        if return_details:
            return fused, {
                "components": components,
                "weights": weights,
                "labels": GRANULARITY_LABELS,
                "mode": self.mode,
            }
        return fused
