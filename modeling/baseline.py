# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class PartAttentionHead(nn.Module):
    def __init__(self, in_planes, num_parts=6):
        super(PartAttentionHead, self).__init__()
        self.num_parts = num_parts
        self.attention = nn.Linear(in_planes, 1)
        nn.init.normal_(self.attention.weight, std=0.001)
        nn.init.constant_(self.attention.bias, 0.0)

    def forward(self, x):
        part_feats = []
        height = x.size(2)
        for part_idx in range(self.num_parts):
            start = height * part_idx // self.num_parts
            end = height * (part_idx + 1) // self.num_parts
            part = x[:, :, start:end, :]
            part_feat = F.adaptive_avg_pool2d(part, 1).view(x.size(0), -1)
            part_feats.append(part_feat)

        part_feats = torch.stack(part_feats, dim=1)
        attention_scores = self.attention(part_feats).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        part_feat = torch.sum(part_feats * attention_weights, dim=1)
        return part_feat


class MultiGranularityPartHead(nn.Module):
    """Project and aggregate horizontal parts at several granularities.

    Every scale owns one projection shared by all of its parts.  Different
    scales never share projection parameters.
    """

    def __init__(self, in_planes, scales, projection_dim=256,
                 aggregation='mean'):
        super(MultiGranularityPartHead, self).__init__()
        self.scales = self._validate_scales(scales)
        if (not isinstance(projection_dim, int)
                or isinstance(projection_dim, bool)
                or projection_dim < 2):
            raise ValueError(
                'MULTI_GRANULARITY_PART_DIM must be an integer greater than or '
                'equal to 2, got {}'.format(
                    projection_dim
                )
            )
        aggregation = str(aggregation).lower()
        if aggregation != 'mean':
            raise ValueError(
                "MULTI_GRANULARITY_PART_AGGREGATION must be 'mean', got {!r}".format(
                    aggregation
                )
            )

        self.in_planes = in_planes
        self.projection_dim = projection_dim
        self.aggregation = aggregation
        self.projections = nn.ModuleDict({
            str(scale): nn.Sequential(
                nn.Linear(in_planes, projection_dim, bias=False),
                nn.LayerNorm(projection_dim),
                nn.ReLU(inplace=False),
            )
            for scale in self.scales
        })
        self.projections.apply(weights_init_kaiming)

    @staticmethod
    def _validate_scales(scales):
        try:
            validated = tuple(scales)
        except TypeError:
            raise ValueError(
                'MULTI_GRANULARITY_PART_SCALES must be a sequence of positive integers'
            )
        if not validated:
            raise ValueError('MULTI_GRANULARITY_PART_SCALES must not be empty')
        if any(
                not isinstance(scale, int)
                or isinstance(scale, bool)
                or scale <= 0
                for scale in validated):
            raise ValueError(
                'MULTI_GRANULARITY_PART_SCALES must contain only positive integers, '
                'got {}'.format(list(validated))
            )
        if len(set(validated)) != len(validated):
            raise ValueError(
                'MULTI_GRANULARITY_PART_SCALES must not contain duplicates, got {}'.format(
                    list(validated)
                )
            )
        if tuple(sorted(validated)) != validated:
            raise ValueError(
                'MULTI_GRANULARITY_PART_SCALES must be sorted in ascending order, got {}'.format(
                    list(validated)
                )
            )
        return validated

    @staticmethod
    def region_bounds(height, scale):
        if height < scale:
            raise ValueError(
                'Backbone feature-map height {} is smaller than part scale {}; '
                'empty horizontal regions are not allowed'.format(height, scale)
            )
        return tuple(
            (height * part_idx // scale,
             height * (part_idx + 1) // scale)
            for part_idx in range(scale)
        )

    def pool_parts(self, feature_map, scale):
        """Return the GAP vector of every horizontal region as [B, scale, C]."""
        if scale not in self.scales:
            raise ValueError(
                'Scale {} is not configured; expected one of {}'.format(
                    scale, list(self.scales)
                )
            )
        height = feature_map.size(2)
        part_features = []
        for start, end in self.region_bounds(height, scale):
            region = feature_map[:, :, start:end, :]
            part_features.append(
                F.adaptive_avg_pool2d(region, 1).flatten(1)
            )
        return torch.stack(part_features, dim=1)

    def forward(self, feature_map):
        if feature_map.dim() != 4:
            raise ValueError(
                'MultiGranularityPartHead expects [B,C,H,W], got shape {}'.format(
                    tuple(feature_map.shape)
                )
            )
        if feature_map.size(1) != self.in_planes:
            raise ValueError(
                'Expected {} backbone channels, got {}'.format(
                    self.in_planes, feature_map.size(1)
                )
            )
        height = feature_map.size(2)
        if height < max(self.scales):
            raise ValueError(
                'Backbone feature-map height {} is smaller than maximum part scale {}; '
                'empty horizontal regions are not allowed'.format(
                    height, max(self.scales)
                )
            )

        scale_features = []
        for scale in self.scales:
            pooled_parts = self.pool_parts(feature_map, scale)
            batch_size, num_parts, channels = pooled_parts.shape
            projected_parts = self.projections[str(scale)](
                pooled_parts.reshape(batch_size * num_parts, channels)
            ).view(batch_size, num_parts, self.projection_dim)
            scale_features.append(projected_parts.mean(dim=1))
        return tuple(scale_features)


class MultiGranularityDynamicGate(nn.Module):
    """Per-sample scaled-softmax controller for existing scale descriptors.

    ``global`` (G1) feeds the global descriptor ``g`` to the controller.  The
    ``concat_global_local`` variant (G2-global-local) feeds
    ``[g, z2, z4, z6]`` before the same controller.  ``concat_local``
    (G2-local-only) feeds only ``[z2, z4, z6]``.  The ``concat_z2_z6``
    ablation feeds ``[z2, z6]`` and produces exactly two normalized weights.
    Its z4 descriptor is still computed by the part head, but deliberately
    excluded from both the controller and the gated fusion.  No mode adds a
    fourth global-feature gate.
    """

    VALID_GATING_INPUTS = (
        'global', 'concat_global_local', 'concat_local', 'concat_z2_z6'
    )

    def __init__(self, in_planes, num_scales, temperature=1.0,
                 gating_input='global', normalization='scaled_softmax',
                 local_feature_dim=None):
        super(MultiGranularityDynamicGate, self).__init__()
        if gating_input not in self.VALID_GATING_INPUTS:
            raise ValueError(
                "MULTI_GRANULARITY_GATING_INPUT must be one of {}, got {!r}"
                .format(self.VALID_GATING_INPUTS, gating_input)
            )
        if normalization != 'scaled_softmax':
            raise ValueError(
                "MULTI_GRANULARITY_GATING_NORMALIZATION must be "
                "'scaled_softmax', got {!r}".format(normalization)
            )
        if (isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not math.isfinite(float(temperature))
                or float(temperature) <= 0.0):
            raise ValueError(
                'MULTI_GRANULARITY_GATING_TAU must be finite and greater than '
                'zero, got {!r}'.format(temperature)
            )
        if (not isinstance(num_scales, int) or isinstance(num_scales, bool)
                or num_scales <= 0):
            raise ValueError('num_scales must be a positive integer')

        self.in_planes = int(in_planes)
        self.num_scales = int(num_scales)
        self.active_scale_indices = (
            (0, 2) if gating_input == 'concat_z2_z6'
            else tuple(range(self.num_scales))
        )
        if gating_input == 'concat_z2_z6' and self.num_scales < 3:
            raise ValueError(
                'concat_z2_z6 gating requires z2, z4, and z6 features'
            )
        self.gate_count = len(self.active_scale_indices)
        self.temperature = float(temperature)
        self.gating_input = gating_input
        self.normalization = normalization
        self.local_feature_dim = (
            None if local_feature_dim is None else int(local_feature_dim)
        )
        if self.gating_input in (
                'concat_global_local', 'concat_local', 'concat_z2_z6'):
            if self.local_feature_dim is None or self.local_feature_dim <= 0:
                raise ValueError(
                    '{} gating requires a positive '
                    'local_feature_dim'.format(self.gating_input)
                )
            local_input_dim = self.gate_count * self.local_feature_dim
            self.controller_input_dim = (
                self.in_planes + local_input_dim
                if self.gating_input == 'concat_global_local'
                else local_input_dim
            )
        else:
            self.controller_input_dim = self.in_planes
        self.controller = nn.Linear(self.controller_input_dim, self.gate_count)
        nn.init.constant_(self.controller.weight, 0.0)
        nn.init.constant_(self.controller.bias, 0.0)

    def controller_input(self, global_feat, scale_features=None):
        """Return the configured controller input without changing descriptors."""
        if global_feat.dim() != 2 or global_feat.size(1) != self.in_planes:
            raise ValueError(
                'Dynamic gate expects global_feat [B,{}], got {}'.format(
                    self.in_planes, tuple(global_feat.shape)
                )
            )
        if self.gating_input == 'global':
            return global_feat
        if scale_features is None or len(scale_features) != self.num_scales:
            received = 0 if scale_features is None else len(scale_features)
            raise ValueError(
                '{} gating expects {} local scale features, got {}'.format(
                    self.gating_input, self.num_scales, received
                )
            )
        checked = []
        for index, feature in enumerate(scale_features):
            expected = (global_feat.size(0), self.local_feature_dim)
            if feature.dim() != 2 or tuple(feature.shape) != expected:
                raise ValueError(
                    'Local scale feature {} must have shape {}, got {}'.format(
                        index, expected, tuple(feature.shape)
                    )
                )
            checked.append(feature)
        if self.gating_input == 'concat_local':
            return torch.cat(tuple(checked), dim=1)
        if self.gating_input == 'concat_z2_z6':
            return torch.cat(
                tuple(checked[index] for index in self.active_scale_indices),
                dim=1,
            )
        return torch.cat((global_feat,) + tuple(checked), dim=1)

    def forward(self, global_feat, scale_features=None):
        controller_input = self.controller_input(global_feat, scale_features)
        logits = self.controller(controller_input)
        probabilities = F.softmax(logits / self.temperature, dim=1)
        weights = (
            probabilities if self.gating_input == 'concat_z2_z6'
            else float(self.gate_count) * probabilities
        )
        return logits, probabilities, weights


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice,
                 part_attention=False, part_attention_parts=6,
                 multi_granularity_part=False,
                 multi_granularity_part_scales=(2, 4, 6),
                  multi_granularity_part_dim=256,
                  multi_granularity_part_aggregation='mean',
                  multi_granularity_part_fusion='concat',
                  multi_granularity_dynamic_gating=False,
                  multi_granularity_gating_input='global',
                  multi_granularity_gating_tau=1.0,
                  multi_granularity_gating_normalization='scaled_softmax'):
        super(Baseline, self).__init__()
        if part_attention and multi_granularity_part:
            raise ValueError(
                'MODEL.PART_ATTENTION and MODEL.MULTI_GRANULARITY_PART cannot '
                'both be enabled; disable the legacy fixed-K part attention'
            )
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.part_attention = part_attention
        self.multi_granularity_part = multi_granularity_part
        if type(multi_granularity_dynamic_gating) is not bool:
            raise ValueError(
                'MULTI_GRANULARITY_DYNAMIC_GATING must be a boolean, got {!r}'
                .format(multi_granularity_dynamic_gating)
            )
        self.multi_granularity_dynamic_gating = multi_granularity_dynamic_gating
        self._last_dynamic_gating = None

        if self.multi_granularity_dynamic_gating and not self.multi_granularity_part:
            raise ValueError(
                'MODEL.MULTI_GRANULARITY_DYNAMIC_GATING=True requires '
                'MODEL.MULTI_GRANULARITY_PART=True'
            )

        if self.part_attention:
            self.part_attention_head = PartAttentionHead(self.in_planes, part_attention_parts)

        self.feature_dim = self.in_planes
        if self.multi_granularity_part:
            fusion = str(multi_granularity_part_fusion).lower()
            if fusion != 'concat':
                raise ValueError(
                    "MULTI_GRANULARITY_PART_FUSION must be 'concat', got {!r}".format(
                        fusion
                    )
                )
            self.multi_granularity_part_fusion = fusion
            self.multi_granularity_part_head = MultiGranularityPartHead(
                in_planes=self.in_planes,
                scales=multi_granularity_part_scales,
                projection_dim=multi_granularity_part_dim,
                aggregation=multi_granularity_part_aggregation,
            )
            self.feature_dim += (
                len(self.multi_granularity_part_head.scales)
                * self.multi_granularity_part_head.projection_dim
            )
            if self.multi_granularity_dynamic_gating:
                self.multi_granularity_dynamic_gate = MultiGranularityDynamicGate(
                    in_planes=self.in_planes,
                    num_scales=len(self.multi_granularity_part_head.scales),
                    temperature=multi_granularity_gating_tau,
                    gating_input=str(multi_granularity_gating_input).lower(),
                    normalization=str(
                        multi_granularity_gating_normalization
                    ).lower(),
                    local_feature_dim=self.multi_granularity_part_head.projection_dim,
                )
                excluded_scale_count = (
                    len(self.multi_granularity_part_head.scales)
                    - len(self.multi_granularity_dynamic_gate.active_scale_indices)
                )
                self.feature_dim -= (
                    excluded_scale_count
                    * self.multi_granularity_part_head.projection_dim
                )

        if self.neck == 'no':
            self.classifier = nn.Linear(self.feature_dim, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.feature_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        feature_map = self.base(x)
        global_feat = self.gap(feature_map)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.part_attention:
            part_feat = self.part_attention_head(feature_map)
            global_feat = global_feat + part_feat
        if self.multi_granularity_part:
            scale_features = self.multi_granularity_part_head(feature_map)
            if self.multi_granularity_dynamic_gating:
                gate_logits, probabilities, weights = (
                    self.multi_granularity_dynamic_gate(global_feat, scale_features)
                )
                active_indices = self.multi_granularity_dynamic_gate.active_scale_indices
                scale_features = tuple(
                    scale_features[scale_index] * weights[:, gate_index:gate_index + 1]
                    for gate_index, scale_index in enumerate(active_indices)
                )
                self._last_dynamic_gating = {
                    'logits': gate_logits.detach(),
                    'probabilities': probabilities.detach(),
                    'weights': weights.detach(),
                    'scales': tuple(
                        self.multi_granularity_part_head.scales[index]
                        for index in active_indices
                    ),
                }
            else:
                self._last_dynamic_gating = None
            fused_pre_bn = torch.cat((global_feat,) + scale_features, dim=1)
        else:
            fused_pre_bn = global_feat

        if self.neck == 'no':
            feat = fused_pre_bn
        elif self.neck == 'bnneck':
            feat = self.bottleneck(fused_pre_bn)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, fused_pre_bn  # pre-BN feature for metric losses
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return fused_pre_bn

    def dynamic_gating_values(self, global_feat, scale_features=None):
        """Expose differentiable controller values for validation/evidence."""
        if not self.multi_granularity_dynamic_gating:
            raise RuntimeError('Dynamic multi-granularity gating is disabled')
        return self.multi_granularity_dynamic_gate(global_feat, scale_features)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
