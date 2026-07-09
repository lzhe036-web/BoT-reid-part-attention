# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import (
    CameraAwareTripletLoss,
    CrossEntropyLabelSmooth,
    HierarchicalCameraAwareTripletLoss,
    TripletLoss,
    count_cross_camera_positives,
)
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    camera_triplet = None
    if cfg.MODEL.CAMERA_AWARE_TRIPLET:
        if cfg.MODEL.HIERARCHICAL_CAMERA_AWARE_TRIPLET:
            camera_triplet = HierarchicalCameraAwareTripletLoss(
                margin=cfg.MODEL.CAMERA_AWARE_TRIPLET_MARGIN,
                negative_temperature=cfg.MODEL.HIERARCHICAL_CAMERA_AWARE_NEGATIVE_TEMPERATURE,
                easy_weight=cfg.MODEL.HIERARCHICAL_CAMERA_AWARE_EASY_WEIGHT,
                boundary_weight=cfg.MODEL.HIERARCHICAL_CAMERA_AWARE_BOUNDARY_WEIGHT,
                hard_weight=cfg.MODEL.HIERARCHICAL_CAMERA_AWARE_HARD_WEIGHT
            )
        else:
            camera_triplet = CameraAwareTripletLoss(
                margin=cfg.MODEL.CAMERA_AWARE_TRIPLET_MARGIN,
                mode=cfg.MODEL.CAMERA_AWARE_TRIPLET_MODE
            )

    def id_loss(score, target):
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            return xent(score, target)
        return F.cross_entropy(score, target)

    def with_camera_loss(total_loss, id_loss_value, triplet_loss_value, feat, target, camids):
        if not cfg.MODEL.CAMERA_AWARE_TRIPLET:
            return total_loss
        if camids is None:
            camera_loss = feat.sum() * 0.0
            cross_camera_count = 0
        else:
            camera_loss = camera_triplet(feat, target, camids)
            cross_camera_count = count_cross_camera_positives(target, camids)
        total_loss = total_loss + cfg.MODEL.CAMERA_AWARE_TRIPLET_LAMBDA * camera_loss
        return {
            'loss_total': total_loss,
            'loss_id': id_loss_value,
            'loss_triplet': triplet_loss_value,
            'loss_camera_triplet': camera_loss,
            'cross_camera_positive_count': cross_camera_count,
        }

    if sampler == 'softmax':
        def loss_func(score, feat, target, camids=None):
            id_loss_value = id_loss(score, target)
            triplet_loss_value = feat.sum() * 0.0
            return with_camera_loss(id_loss_value, id_loss_value, triplet_loss_value, feat, target, camids)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target, camids=None):
            id_loss_value = score.sum() * 0.0
            triplet_loss_value = triplet(feat, target)[0]
            return with_camera_loss(triplet_loss_value, id_loss_value, triplet_loss_value, feat, target, camids)
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, camids=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                id_loss_value = id_loss(score, target)
                triplet_loss_value = triplet(feat, target)[0]
                total_loss = id_loss_value + triplet_loss_value
                return with_camera_loss(total_loss, id_loss_value, triplet_loss_value, feat, target, camids)
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target, camids=None):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion
