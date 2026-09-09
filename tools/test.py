# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    alpha_variant = (cfg.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL
                     and cfg.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA in (0.1, 0.5)
                     and cfg.MODEL.MULTI_GRANULARITY_GATING_TAU == 0.5)
    if alpha_variant:
        from utils.g2_e_checkpoint_identity import validate
        validate(cfg.TEST.WEIGHT, cfg)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    if alpha_variant:
        from tools.analyze_dynamic_gating import _state_dict
        checkpoint = torch.load(cfg.TEST.WEIGHT, map_location="cpu")
        model.load_state_dict(_state_dict(checkpoint), strict=True)
    else:
        model.load_param(cfg.TEST.WEIGHT)

    inference(cfg, model, val_loader, num_query)


if __name__ == '__main__':
    main()
