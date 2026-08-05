# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from pathlib import Path

# Set before importing torch so supported CUDA runtimes can choose a
# deterministic cuBLAS workspace configuration.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import yaml

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_train_with_center
from modeling import build_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR

from utils.logger import setup_logger
from utils.reproducibility import (
    ensure_python_hash_seed,
    read_explicit_config_seed,
    seed_everything,
    validate_seed,
    validate_seed_evidence_chain,
    write_reproducibility_record,
)


def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes)

    if cfg.MODEL.IF_WITH_CENTER == 'no':
        print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        optimizer = make_optimizer(cfg, model)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        loss_func = make_loss(cfg, num_classes)     # modified by gu

        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        arguments = {}

        do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch     # add for using self trained model
        )
    elif cfg.MODEL.IF_WITH_CENTER == 'yes':
        print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        loss_func, center_criterion = make_loss_with_center(cfg, num_classes)  # modified by gu
        optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        arguments = {}

        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
            print('Path to the checkpoint of center_param:', path_to_center_param)
            path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
            print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            center_criterion.load_state_dict(torch.load(path_to_center_param))
            optimizer_center.load_state_dict(torch.load(path_to_optimizer_center))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        do_train_with_center(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch     # add for using self trained model
        )
    else:
        print("Unsupported value for cfg.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(cfg.MODEL.IF_WITH_CENTER))


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    strict_expected_seed = os.environ.get("BOT_EXPECTED_TRAINING_SEED")
    if strict_expected_seed is not None:
        strict_expected_seed = validate_seed(int(strict_expected_seed))
    if not args.config_file:
        raise ValueError(
            "--config_file with an explicit top-level SEED is required for training"
        )
    # Source validation is unconditional. BOT_EXPECTED_TRAINING_SEED adds a
    # formal identity constraint; it never enables or disables this check.
    source_config_seed = read_explicit_config_seed(args.config_file)
    formal_source = None
    if strict_expected_seed is not None:
        from utils.experiment_recording import (
            FORMAL_CONFIG_RELATIVE_PATH,
            PreflightError,
            validate_formal_protocol,
        )
        repo_root = Path(__file__).resolve().parents[1]
        expected_config = (repo_root / FORMAL_CONFIG_RELATIVE_PATH).resolve()
        if Path(args.config_file).resolve() != expected_config:
            raise PreflightError(
                "Formal training may only use {}".format(expected_config)
            )
        with open(args.config_file, "r", encoding="utf-8") as handle:
            formal_source = yaml.safe_load(handle)
        validate_formal_protocol(formal_source, "source")

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if strict_expected_seed is not None:
        validate_formal_protocol(cfg, "resolved")

    # Fail before constructing a dataset or model if reproducibility cannot be
    # guaranteed and documented for this run.
    training_seed = validate_seed(cfg.SEED)
    validate_seed_evidence_chain(
        source_config_seed,
        training_seed,
        training_seed,
        expected_seed=strict_expected_seed,
    )
    ensure_python_hash_seed(training_seed)
    seed_state = seed_everything(training_seed)
    validate_seed_evidence_chain(
        source_config_seed,
        training_seed,
        seed_state["seed"],
        expected_seed=strict_expected_seed,
    )
    output_dir = cfg.OUTPUT_DIR
    if not output_dir:
        raise ValueError("OUTPUT_DIR must be set so the training seed can be recorded")
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    metadata_path, metadata = write_reproducibility_record(
        output_dir=output_dir,
        cfg=cfg,
        seed_state=seed_state,
        config_file=args.config_file,
        source_config_seed=source_config_seed,
        cli_overrides=args.opts,
        command=sys.argv,
    )
    validate_seed_evidence_chain(
        source_config_seed,
        training_seed,
        seed_state["seed"],
        metadata_seed=metadata["seed"],
        expected_seed=strict_expected_seed,
    )
    logger.info(
        "Reproducibility fixed explicitly: training_seed={}, Python=True, NumPy=True, "
        "PyTorch=True, CUDA_all={}, cudnn.deterministic=True, cudnn.benchmark=False"
        .format(training_seed, seed_state["torch_cuda_all_seeded"])
    )
    logger.info("Reproducibility metadata saved to {}".format(metadata_path))
    train(cfg)


if __name__ == '__main__':
    main()
