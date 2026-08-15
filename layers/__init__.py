# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import (
    CameraAwareTripletLoss,
    CrossEntropyLabelSmooth,
    CrossCameraPositiveLoss,
    TripletLoss,
    count_cross_camera_positives,
)
from .center_loss import CenterLoss
from .part_correspondence_consistency import (
    SUPPORTED_ALIGNMENT_MODES,
    part_alignment_loss,
    validate_softmin_tau,
)


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
    cross_camera_positive = None
    pcc_enabled = cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY
    if pcc_enabled and cfg.MODEL.PCC_PARTS != 6:
        raise ValueError("Part alignment requires PCC_PARTS=6")
    if pcc_enabled and cfg.MODEL.PCC_MODE not in SUPPORTED_ALIGNMENT_MODES:
        raise ValueError(
            "Unsupported PCC_MODE {!r}; expected one of {}".format(
                cfg.MODEL.PCC_MODE, ", ".join(SUPPORTED_ALIGNMENT_MODES)
            )
        )
    if pcc_enabled and cfg.MODEL.PCC_MODE == 'soft_min':
        validate_softmin_tau(cfg.MODEL.PCC_SOFTMIN_TAU)
    if cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY:
        if cfg.MODEL.CAMERA_AWARE_TRIPLET:
            print("CROSS_CAMERA_POSITIVE_ONLY is enabled; skip CAMERA_AWARE_TRIPLET auxiliary loss.")
        cross_camera_positive = CrossCameraPositiveLoss(
            mode=cfg.MODEL.CROSS_CAMERA_POSITIVE_MODE
        )
    if cfg.MODEL.CAMERA_AWARE_TRIPLET:
        camera_triplet = CameraAwareTripletLoss(
            margin=cfg.MODEL.CAMERA_AWARE_TRIPLET_MARGIN,
            mode=cfg.MODEL.CAMERA_AWARE_TRIPLET_MODE
        )

    def id_loss(score, target):
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            return xent(score, target)
        return F.cross_entropy(score, target)

    def with_camera_loss(total_loss, id_loss_value, triplet_loss_value, feat,
                         target, camids, pcc_local_features=None):
        has_camera_auxiliary = (
            cfg.MODEL.CAMERA_AWARE_TRIPLET
            or cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY
        )
        if not has_camera_auxiliary and not pcc_enabled:
            return total_loss

        zero = feat.sum() * 0.0
        auxiliary_loss = zero
        cross_camera_count = 0
        if has_camera_auxiliary:
            if camids is None:
                auxiliary_loss = zero
            elif cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY:
                auxiliary_loss = cross_camera_positive(feat, target, camids)
                cross_camera_count = count_cross_camera_positives(target, camids)
            else:
                auxiliary_loss = camera_triplet(feat, target, camids)
                cross_camera_count = count_cross_camera_positives(target, camids)
            if cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY:
                total_loss = (
                    total_loss
                    + cfg.MODEL.CROSS_CAMERA_POSITIVE_LAMBDA * auxiliary_loss
                )
            else:
                total_loss = (
                    total_loss
                    + cfg.MODEL.CAMERA_AWARE_TRIPLET_LAMBDA * auxiliary_loss
                )

        loss_pcc = zero
        valid_pcc_pair_count = 0
        mean_fixed_index_part_distance = zero
        hard_alignment_loss = zero
        valid_alignment_pair_count = 0
        mean_hard_path_cost = zero
        mean_path_absolute_offset = zero
        soft_alignment_loss = zero
        mean_soft_path_cost = zero
        if pcc_enabled:
            if pcc_local_features is None:
                raise RuntimeError(
                    "PCC is enabled but the model returned no local descriptors"
                )
            if pcc_local_features.dim() != 3:
                raise RuntimeError("PCC local descriptors must have shape [B, K, C]")
            if pcc_local_features.size(1) != cfg.MODEL.PCC_PARTS:
                raise RuntimeError(
                    "PCC local descriptor count {} does not match PCC_PARTS {}"
                    .format(pcc_local_features.size(1), cfg.MODEL.PCC_PARTS)
                )
            if camids is None:
                raise RuntimeError("PCC is enabled but camera IDs are unavailable")
            alignment = part_alignment_loss(
                pcc_local_features,
                target,
                camids,
                cfg.MODEL.PCC_MODE,
                softmin_tau=cfg.MODEL.PCC_SOFTMIN_TAU,
            )
            loss_pcc = alignment['loss_pcc']
            valid_pcc_pair_count = alignment['valid_pcc_pair_count']
            mean_fixed_index_part_distance = alignment[
                'mean_fixed_index_part_distance'
            ]
            hard_alignment_loss = alignment['hard_alignment_loss']
            valid_alignment_pair_count = alignment[
                'valid_alignment_pair_count'
            ]
            mean_hard_path_cost = alignment['mean_hard_path_cost']
            mean_path_absolute_offset = alignment[
                'mean_path_absolute_offset'
            ]
            soft_alignment_loss = alignment['soft_alignment_loss']
            mean_soft_path_cost = alignment['mean_soft_path_cost']
            total_loss = total_loss + cfg.MODEL.PCC_LAMBDA * loss_pcc

        return {
            'loss_total': total_loss,
            'loss_id': id_loss_value,
            'loss_triplet': triplet_loss_value,
            'loss_camera_triplet': auxiliary_loss if cfg.MODEL.CAMERA_AWARE_TRIPLET and not cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY else zero,
            'loss_cross_camera_positive': auxiliary_loss if cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY else zero,
            'cross_camera_positive_count': cross_camera_count,
            'loss_pcc': loss_pcc,
            'valid_pcc_pair_count': valid_pcc_pair_count,
            'mean_fixed_index_part_distance': mean_fixed_index_part_distance,
            'hard_alignment_loss': hard_alignment_loss,
            'valid_alignment_pair_count': valid_alignment_pair_count,
            'mean_hard_path_cost': mean_hard_path_cost,
            'mean_path_absolute_offset': mean_path_absolute_offset,
            'soft_alignment_loss': soft_alignment_loss,
            'mean_soft_path_cost': mean_soft_path_cost,
            'alignment_temperature': (
                cfg.MODEL.PCC_SOFTMIN_TAU
                if cfg.MODEL.PCC_MODE == 'soft_min' else None
            ),
        }

    if sampler == 'softmax':
        def loss_func(score, feat, target, camids=None, pcc_local_features=None):
            id_loss_value = id_loss(score, target)
            triplet_loss_value = feat.sum() * 0.0
            return with_camera_loss(
                id_loss_value, id_loss_value, triplet_loss_value, feat,
                target, camids, pcc_local_features
            )
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target, camids=None, pcc_local_features=None):
            id_loss_value = score.sum() * 0.0
            triplet_loss_value = triplet(feat, target)[0]
            return with_camera_loss(
                triplet_loss_value, id_loss_value, triplet_loss_value, feat,
                target, camids, pcc_local_features
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, camids=None, pcc_local_features=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                id_loss_value = id_loss(score, target)
                triplet_loss_value = triplet(feat, target)[0]
                total_loss = id_loss_value + triplet_loss_value
                return with_camera_loss(
                    total_loss, id_loss_value, triplet_loss_value, feat,
                    target, camids, pcc_local_features
                )
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY:
        raise ValueError("Part alignment does not support center-loss training")
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
