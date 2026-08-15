# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline


def build_model(cfg, num_classes):
    if (cfg.MODEL.MULTI_GRANULARITY_PART
            and str(cfg.MODEL.IF_WITH_CENTER).lower() == 'yes'):
        raise ValueError(
            "MODEL.MULTI_GRANULARITY_PART=True is incompatible with "
            "MODEL.IF_WITH_CENTER='yes': center loss still assumes a 2048-dimensional "
            "descriptor. Set IF_WITH_CENTER='no' for this experiment."
        )
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
        multi_granularity_part=cfg.MODEL.MULTI_GRANULARITY_PART,
        multi_granularity_part_scales=cfg.MODEL.MULTI_GRANULARITY_PART_SCALES,
        multi_granularity_part_dim=cfg.MODEL.MULTI_GRANULARITY_PART_DIM,
        multi_granularity_part_aggregation=cfg.MODEL.MULTI_GRANULARITY_PART_AGGREGATION,
        multi_granularity_part_fusion=cfg.MODEL.MULTI_GRANULARITY_PART_FUSION,
        multi_granularity_dynamic_gating=cfg.MODEL.MULTI_GRANULARITY_DYNAMIC_GATING,
        multi_granularity_gating_input=cfg.MODEL.MULTI_GRANULARITY_GATING_INPUT,
        multi_granularity_gating_tau=cfg.MODEL.MULTI_GRANULARITY_GATING_TAU,
        multi_granularity_gating_normalization=cfg.MODEL.MULTI_GRANULARITY_GATING_NORMALIZATION,
    )
    return model
