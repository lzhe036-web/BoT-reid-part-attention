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
    )
    return model
