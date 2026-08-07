# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import cfg
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .multi_granularity_local import MultiGranularityLocalFeature


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
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
    def __init__(self, in_planes, num_parts=6,
                 camera_conditional=False, num_cameras=None):
        super(PartAttentionHead, self).__init__()
        self.num_parts = int(num_parts)
        self.camera_conditional = bool(camera_conditional)
        self.num_cameras = int(num_cameras) if num_cameras is not None else None
        if self.num_parts <= 0:
            raise ValueError("num_parts must be positive")
        if self.camera_conditional:
            if self.num_cameras is None or self.num_cameras <= 0:
                raise ValueError(
                    "num_cameras must be positive when camera-conditional "
                    "part attention is enabled"
                )
            self.camera_embedding = nn.Embedding(
                self.num_cameras, self.num_parts
            )
            nn.init.zeros_(self.camera_embedding.weight)
        self.attention = nn.Linear(in_planes, 1)
        nn.init.normal_(self.attention.weight, std=0.001)
        nn.init.constant_(self.attention.bias, 0.0)

    def _validate_camids(self, camids, batch_size, device):
        if camids is None:
            raise ValueError(
                "camids are required when camera-conditional part attention "
                "is enabled"
            )
        if not torch.is_tensor(camids):
            raise TypeError("camids must be a torch.Tensor")
        if camids.dtype != torch.long:
            raise TypeError("camids must have dtype torch.long")
        if camids.dim() != 1:
            raise ValueError("camids must have shape [batch]")
        if int(camids.numel()) != int(batch_size):
            raise ValueError(
                "camids batch size {} does not match image batch size {}"
                .format(camids.numel(), batch_size)
            )
        if camids.device != device:
            raise ValueError(
                "camids must be on the same device as the model input"
            )
        if camids.numel() > 0:
            minimum = int(camids.min().item())
            maximum = int(camids.max().item())
            if minimum < 0 or maximum >= self.num_cameras:
                raise ValueError(
                    "camids must be in [0, {}), got min={} max={}"
                    .format(self.num_cameras, minimum, maximum)
                )
        return camids

    def forward(self, x, camids=None, return_attention=False):
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
        if self.camera_conditional:
            camids = self._validate_camids(
                camids, part_feats.size(0), part_feats.device
            )
            attention_scores = attention_scores + self.camera_embedding(camids)
        attention_weights = F.softmax(attention_scores, dim=1)
        part_feat = torch.sum(
            part_feats * attention_weights.unsqueeze(-1), dim=1
        )
        if return_attention:
            return part_feat, attention_weights
        return part_feat


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat,
                 model_name, pretrain_choice, part_attention=None,
                 part_attention_parts=None,
                 camera_conditional_part_attention=None, num_cameras=None,
                 multi_granularity_local=None,
                 multi_granularity_scales=None, multi_granularity_dim=None,
                 multi_granularity_aggregation=None):
        super(Baseline, self).__init__()
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
        if part_attention is None:
            part_attention = cfg.MODEL.PART_ATTENTION
        if part_attention_parts is None:
            part_attention_parts = cfg.MODEL.PART_ATTENTION_PARTS
        self.part_attention = part_attention
        if camera_conditional_part_attention is None:
            camera_conditional_part_attention = (
                cfg.MODEL.CAMERA_CONDITIONAL_PART_ATTENTION
            )
        self.camera_conditional_part_attention = bool(
            camera_conditional_part_attention
        )
        self.num_cameras = int(num_cameras) if num_cameras is not None else None
        if self.camera_conditional_part_attention and not self.part_attention:
            raise ValueError(
                "camera-conditional part attention requires PART_ATTENTION=True"
            )

        if self.part_attention:
            self.part_attention_head = PartAttentionHead(
                self.in_planes,
                part_attention_parts,
                camera_conditional=self.camera_conditional_part_attention,
                num_cameras=self.num_cameras,
            )

        if multi_granularity_local is None:
            multi_granularity_local = cfg.MODEL.MULTI_GRANULARITY_LOCAL
        if multi_granularity_scales is None:
            multi_granularity_scales = cfg.MODEL.MULTI_GRANULARITY_SCALES
        if multi_granularity_dim is None:
            multi_granularity_dim = cfg.MODEL.MULTI_GRANULARITY_DIM
        if multi_granularity_aggregation is None:
            multi_granularity_aggregation = (
                cfg.MODEL.MULTI_GRANULARITY_AGGREGATION
            )
        self.multi_granularity_local = bool(multi_granularity_local)
        self.descriptor_dim = self.in_planes
        if self.multi_granularity_local:
            self.multi_granularity_head = MultiGranularityLocalFeature(
                self.in_planes,
                scales=multi_granularity_scales,
                projection_dim=multi_granularity_dim,
                aggregation=multi_granularity_aggregation,
            )
            self.descriptor_dim += self.multi_granularity_head.output_dim

        if self.neck == 'no':
            self.classifier = nn.Linear(self.descriptor_dim, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.descriptor_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.descriptor_dim, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def _forward_impl(self, x, camids=None, return_shape_trace=False):
        feature_map = self.base(x)
        global_feat = self.gap(feature_map)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        shape_trace = None
        if return_shape_trace:
            shape_trace = {
                'backbone_feature_map': tuple(feature_map.shape),
                'global_feature': tuple(global_feat.shape),
                'baseline_existing_attention': bool(self.part_attention),
                'camera_conditional_part_attention': bool(
                    self.camera_conditional_part_attention
                ),
                'camera_count': (
                    self.num_cameras
                    if self.camera_conditional_part_attention else 0
                ),
                'new_module_attention': False,
            }
        if self.part_attention:
            part_feat = self.part_attention_head(feature_map, camids=camids)
            global_feat = global_feat + part_feat
        if return_shape_trace:
            shape_trace['global_feature_after_part_attention'] = tuple(
                global_feat.shape
            )

        if self.multi_granularity_local:
            local_features, local_details = self.multi_granularity_head(
                feature_map, return_details=True
            )
            if return_shape_trace:
                for scale, details in local_details.items():
                    shape_trace['k{}_part_features'.format(scale)] = [
                        tuple(part.shape) for part in details['part_features']
                    ]
                    shape_trace['k{}_part_bounds'.format(scale)] = list(
                        details['bounds']
                    )
                    shape_trace['k{}_aggregated'.format(scale)] = tuple(
                        details['aggregated'].shape
                    )
            global_feat = torch.cat((global_feat,) + local_features, dim=1)
        if return_shape_trace:
            shape_trace['concat_feature'] = tuple(global_feat.shape)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        if return_shape_trace:
            shape_trace['bnneck_feature'] = tuple(feat.shape)
            shape_trace['classifier_input'] = tuple(feat.shape)
            shape_trace['inference_feature'] = tuple(
                feat.shape if self.neck_feat == 'after' else global_feat.shape
            )

        if self.training:
            cls_score = self.classifier(feat)
            result = (cls_score, global_feat)  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                result = feat
            else:
                # print("Test with feature before BN")
                result = global_feat
        if return_shape_trace:
            return result, shape_trace
        return result

    def forward(self, x, camids=None):
        return self._forward_impl(
            x, camids=camids, return_shape_trace=False
        )

    def forward_with_shape_trace(self, x, camids=None):
        """Synthetic-validation helper; the formal training path never calls it."""
        return self._forward_impl(x, camids=camids, return_shape_trace=True)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
