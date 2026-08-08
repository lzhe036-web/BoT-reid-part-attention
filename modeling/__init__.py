# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline


def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    model = Baseline(
        num_classes,
        cfg.MODEL.LAST_STRIDE,
        cfg.MODEL.PRETRAIN_PATH,
        cfg.MODEL.NECK,
        cfg.TEST.NECK_FEAT,
        cfg.MODEL.NAME,
        cfg.MODEL.PRETRAIN_CHOICE,
        part_attention=cfg.MODEL.PART_ATTENTION,
        part_attention_parts=cfg.MODEL.PART_ATTENTION_PARTS,
        part_correspondence_consistency=cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY,
        pcc_parts=cfg.MODEL.PCC_PARTS,
        multi_granularity_local=cfg.MODEL.MULTI_GRANULARITY_LOCAL,
        multi_granularity_scales=cfg.MODEL.MULTI_GRANULARITY_SCALES,
        multi_granularity_dim=cfg.MODEL.MULTI_GRANULARITY_DIM,
        multi_granularity_aggregation=(
            cfg.MODEL.MULTI_GRANULARITY_AGGREGATION
        ),
        multi_granularity_fusion=cfg.MODEL.MULTI_GRANULARITY_FUSION,
        multi_granularity_fusion_mode=(
            cfg.MODEL.MULTI_GRANULARITY_FUSION_MODE
        ),
        multi_granularity_fusion_dim=(
            cfg.MODEL.MULTI_GRANULARITY_FUSION_DIM
        ),
        dynamic_gating_hidden_dim=cfg.MODEL.DYNAMIC_GATING_HIDDEN_DIM,
    )
    return model
