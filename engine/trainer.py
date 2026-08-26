# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP


def _unpack_train_batch(batch):
    if len(batch) == 3:
        img, target, camids = batch
    else:
        img, target = batch
        camids = None
    return img, target, camids


def _loss_output_to_dict(loss_output):
    if isinstance(loss_output, dict):
        loss = loss_output['loss_total']
        return loss, loss_output

    return loss_output, {
        'loss_total': loss_output,
        'loss_id': loss_output,
        'loss_triplet': loss_output.new_tensor(0.0),
        'loss_camera_triplet': loss_output.new_tensor(0.0),
        'loss_cross_camera_positive': loss_output.new_tensor(0.0),
        'cross_camera_positive_count': 0,
        'loss_pcc': loss_output.new_tensor(0.0),
        'valid_pcc_pair_count': 0,
        'mean_fixed_index_part_distance': loss_output.new_tensor(0.0),
        'hard_alignment_loss': loss_output.new_tensor(0.0),
        'valid_alignment_pair_count': 0,
        'mean_hard_path_cost': loss_output.new_tensor(0.0),
        'mean_path_absolute_offset': loss_output.new_tensor(0.0),
        'soft_alignment_loss': loss_output.new_tensor(0.0),
        'mean_soft_path_cost': loss_output.new_tensor(0.0),
        'windowed_soft_alignment_loss': loss_output.new_tensor(0.0),
        'mean_windowed_soft_path_cost': loss_output.new_tensor(0.0),
        'alignment_window': None,
        'alignment_temperature': None,
    }


def _unpack_train_model_output(model_output):
    if not isinstance(model_output, (tuple, list)):
        raise RuntimeError("Training model output must be a tuple or list")
    if len(model_output) == 2:
        score, feat = model_output
        return score, feat, None
    if len(model_output) == 3:
        score, feat, pcc_local_features = model_output
        return score, feat, pcc_local_features
    raise RuntimeError(
        "Training model output must contain 2 values, or 3 when PCC is enabled"
    )


def _item(value):
    if torch.is_tensor(value):
        return value.item()
    return value


def _engine_epoch_length(engine):
    """Return Ignite's authoritative epoch length for logging/evidence only."""
    epoch_length = engine.state.epoch_length
    if epoch_length is None or int(epoch_length) <= 0:
        raise RuntimeError(
            "Ignite engine.state.epoch_length must be a positive integer"
        )
    return int(epoch_length)


def _attach_epoch_evidence_logging(trainer, logger):
    """Attach observation-only epoch counters and authoritative evidence."""
    epoch_log_state = {"iteration": 0}

    @trainer.on(Events.EPOCH_STARTED)
    def reset_epoch_log_counter(_engine):
        epoch_log_state["iteration"] = 0

    @trainer.on(Events.ITERATION_COMPLETED)
    def increment_epoch_log_counter(_engine):
        epoch_log_state["iteration"] += 1

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_evidence(engine):
        logger.info(
            "EPOCH_EVIDENCE epoch={} global_iteration={} epoch_length={}"
            .format(
                int(engine.state.epoch),
                int(engine.state.iteration),
                _engine_epoch_length(engine),
            )
        )

    return epoch_log_state


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target, camids = _unpack_train_batch(batch)
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        camids = camids.to(device) if camids is not None and torch.cuda.device_count() >= 1 else camids
        score, feat, pcc_local_features = _unpack_train_model_output(model(img))
        if pcc_local_features is None:
            loss_output = loss_fn(score, feat, target, camids)
        else:
            loss_output = loss_fn(
                score, feat, target, camids, pcc_local_features
            )
        loss, loss_dict = _loss_output_to_dict(loss_output)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return {
            'loss_total': _item(loss_dict['loss_total']),
            'loss_id': _item(loss_dict['loss_id']),
            'loss_triplet': _item(loss_dict['loss_triplet']),
            'loss_camera_triplet': _item(loss_dict['loss_camera_triplet']),
            'loss_cross_camera_positive': _item(loss_dict.get('loss_cross_camera_positive', loss_dict['loss_total'].new_tensor(0.0) if torch.is_tensor(loss_dict['loss_total']) else 0.0)),
            'cross_camera_positive_count': loss_dict['cross_camera_positive_count'],
            'loss_pcc': _item(loss_dict.get('loss_pcc', 0.0)),
            'valid_pcc_pair_count': loss_dict.get('valid_pcc_pair_count', 0),
            'mean_fixed_index_part_distance': _item(
                loss_dict.get('mean_fixed_index_part_distance', 0.0)
            ),
            'hard_alignment_loss': _item(
                loss_dict.get('hard_alignment_loss', 0.0)
            ),
            'valid_alignment_pair_count': loss_dict.get(
                'valid_alignment_pair_count', 0
            ),
            'mean_hard_path_cost': _item(
                loss_dict.get('mean_hard_path_cost', 0.0)
            ),
            'mean_path_absolute_offset': _item(
                loss_dict.get('mean_path_absolute_offset', 0.0)
            ),
            'soft_alignment_loss': _item(
                loss_dict.get('soft_alignment_loss', 0.0)
            ),
            'mean_soft_path_cost': _item(
                loss_dict.get('mean_soft_path_cost', 0.0)
            ),
            'windowed_soft_alignment_loss': _item(
                loss_dict.get('windowed_soft_alignment_loss', 0.0)
            ),
            'mean_windowed_soft_path_cost': _item(
                loss_dict.get('mean_windowed_soft_path_cost', 0.0)
            ),
            'alignment_window': loss_dict.get('alignment_window'),
            'alignment_temperature': loss_dict.get(
                'alignment_temperature'
            ),
            'acc': acc.item(),
        }

    return Engine(_update)


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target, camids = _unpack_train_batch(batch)
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        camids = camids.to(device) if camids is not None and torch.cuda.device_count() >= 1 else camids
        score, feat, pcc_local_features = _unpack_train_model_output(model(img))
        if pcc_local_features is None:
            loss_output = loss_fn(score, feat, target, camids)
        else:
            loss_output = loss_fn(
                score, feat, target, camids, pcc_local_features
            )
        loss, loss_dict = _loss_output_to_dict(loss_output)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return {
            'loss_total': _item(loss_dict['loss_total']),
            'loss_id': _item(loss_dict['loss_id']),
            'loss_triplet': _item(loss_dict['loss_triplet']),
            'loss_camera_triplet': _item(loss_dict['loss_camera_triplet']),
            'loss_cross_camera_positive': _item(loss_dict.get('loss_cross_camera_positive', loss_dict['loss_total'].new_tensor(0.0) if torch.is_tensor(loss_dict['loss_total']) else 0.0)),
            'cross_camera_positive_count': loss_dict['cross_camera_positive_count'],
            'loss_pcc': _item(loss_dict.get('loss_pcc', 0.0)),
            'valid_pcc_pair_count': loss_dict.get('valid_pcc_pair_count', 0),
            'mean_fixed_index_part_distance': _item(
                loss_dict.get('mean_fixed_index_part_distance', 0.0)
            ),
            'acc': acc.item(),
        }

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(
        output_dir,
        cfg.MODEL.NAME,
        n_saved=10,
        require_empty=False
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=checkpoint_period),
        checkpointer,
        {'model': model, 'optimizer': optimizer}
    )
    epoch_log_state = _attach_epoch_evidence_logging(trainer, logger)
    timer = Timer(average=True)

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x['loss_total']).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x['loss_id']).attach(trainer, 'avg_loss_id')
    RunningAverage(output_transform=lambda x: x['loss_triplet']).attach(trainer, 'avg_loss_triplet')
    RunningAverage(output_transform=lambda x: x['loss_camera_triplet']).attach(trainer, 'avg_loss_camera_triplet')
    RunningAverage(output_transform=lambda x: x['loss_cross_camera_positive']).attach(trainer, 'avg_loss_cross_camera_positive')
    RunningAverage(output_transform=lambda x: x['cross_camera_positive_count']).attach(trainer, 'avg_cross_camera_positive_count')
    if cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY:
        RunningAverage(output_transform=lambda x: x['loss_pcc']).attach(trainer, 'avg_loss_pcc')
        RunningAverage(output_transform=lambda x: x['valid_pcc_pair_count']).attach(trainer, 'avg_valid_pcc_pair_count')
        if cfg.MODEL.PCC_MODE == 'fixed_index':
            RunningAverage(output_transform=lambda x: x['mean_fixed_index_part_distance']).attach(trainer, 'avg_mean_fixed_index_part_distance')
    RunningAverage(output_transform=lambda x: x['acc']).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()
        if cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY:
            engine.state.pcc_epoch_pair_count = 0
            if cfg.MODEL.PCC_MODE == 'fixed_index':
                engine.state.pcc_epoch_distance_sum = 0.0
            elif cfg.MODEL.PCC_MODE == 'hard_shortest_path':
                engine.state.hard_epoch_loss_sum = 0.0
                engine.state.hard_epoch_cost_sum = 0.0
                engine.state.hard_epoch_offset_sum = 0.0
            elif cfg.MODEL.PCC_MODE == 'soft_min':
                engine.state.soft_epoch_loss_sum = 0.0
                engine.state.soft_epoch_cost_sum = 0.0
            elif cfg.MODEL.PCC_MODE == 'windowed_soft_min':
                engine.state.windowed_soft_epoch_loss_sum = 0.0
                engine.state.windowed_soft_epoch_cost_sum = 0.0

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iteration_in_epoch = epoch_log_state["iteration"]

        if cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY:
            pair_count = int(engine.state.output['valid_pcc_pair_count'])
            engine.state.pcc_epoch_pair_count += pair_count
            if cfg.MODEL.PCC_MODE == 'fixed_index':
                engine.state.pcc_epoch_distance_sum += (
                    pair_count
                    * float(engine.state.output['mean_fixed_index_part_distance'])
                )
            elif cfg.MODEL.PCC_MODE == 'hard_shortest_path':
                engine.state.hard_epoch_loss_sum += (
                    pair_count
                    * float(engine.state.output['hard_alignment_loss'])
                )
                engine.state.hard_epoch_cost_sum += (
                    pair_count
                    * float(engine.state.output['mean_hard_path_cost'])
                )
                engine.state.hard_epoch_offset_sum += (
                    pair_count
                    * float(engine.state.output['mean_path_absolute_offset'])
                )
            elif cfg.MODEL.PCC_MODE == 'soft_min':
                engine.state.soft_epoch_loss_sum += (
                    pair_count
                    * float(engine.state.output['soft_alignment_loss'])
                )
                engine.state.soft_epoch_cost_sum += (
                    pair_count
                    * float(engine.state.output['mean_soft_path_cost'])
                )
            elif cfg.MODEL.PCC_MODE == 'windowed_soft_min':
                engine.state.windowed_soft_epoch_loss_sum += (
                    pair_count * float(
                        engine.state.output['windowed_soft_alignment_loss']
                    )
                )
                engine.state.windowed_soft_epoch_cost_sum += (
                    pair_count * float(
                        engine.state.output['mean_windowed_soft_path_cost']
                    )
                )

        if iteration_in_epoch % log_period == 0:
            if (cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY
                    and cfg.MODEL.PCC_MODE == 'fixed_index'):
                logger.info("Epoch[{}] Iteration[{}/{}] loss_total: {:.3f}, loss_id: {:.3f}, "
                            "loss_triplet: {:.3f}, loss_camera_triplet: {:.3f}, "
                            "loss_cross_camera_positive: {:.3f}, loss_pcc: {:.3f}, "
                            "cross_camera_positive_count: {:.1f}, valid_pcc_pair_count: {:.1f}, "
                            "mean_fixed_index_part_distance: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(engine.state.epoch, iteration_in_epoch,
                                    _engine_epoch_length(engine),
                                    engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss_id'],
                                    engine.state.metrics['avg_loss_triplet'],
                                    engine.state.metrics['avg_loss_camera_triplet'],
                                    engine.state.metrics['avg_loss_cross_camera_positive'],
                                    engine.state.metrics['avg_loss_pcc'],
                                    engine.state.metrics['avg_cross_camera_positive_count'],
                                    engine.state.metrics['avg_valid_pcc_pair_count'],
                                    engine.state.metrics['avg_mean_fixed_index_part_distance'],
                                    engine.state.metrics['avg_acc'],
                                    scheduler.get_lr()[0]))
            elif (cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY
                    and cfg.MODEL.PCC_MODE == 'hard_shortest_path'):
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] loss_total: {:.3f}, loss_id: {:.3f}, "
                    "loss_triplet: {:.3f}, loss_camera_triplet: {:.3f}, "
                    "loss_cross_camera_positive: {:.3f}, loss_pcc: {:.3f}, "
                    "cross_camera_positive_count: {:.1f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(
                        engine.state.epoch, iteration_in_epoch,
                        _engine_epoch_length(engine),
                        engine.state.metrics['avg_loss'],
                        engine.state.metrics['avg_loss_id'],
                        engine.state.metrics['avg_loss_triplet'],
                        engine.state.metrics['avg_loss_camera_triplet'],
                        engine.state.metrics['avg_loss_cross_camera_positive'],
                        engine.state.metrics['avg_loss_pcc'],
                        engine.state.metrics['avg_cross_camera_positive_count'],
                        engine.state.metrics['avg_acc'],
                        scheduler.get_lr()[0],
                    )
                )
                logger.info(
                    "Hard Alignment Batch - Epoch: {} Iteration: {} "
                    "hard_alignment_loss: {:.6f} "
                    "valid_alignment_pair_count: {} "
                    "mean_hard_path_cost: {:.6f} "
                    "mean_path_absolute_offset: {:.6f}"
                    .format(
                        engine.state.epoch,
                        iteration_in_epoch,
                        float(engine.state.output['hard_alignment_loss']),
                        int(engine.state.output['valid_alignment_pair_count']),
                        float(engine.state.output['mean_hard_path_cost']),
                        float(engine.state.output['mean_path_absolute_offset']),
                    )
                )
            elif (cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY
                    and cfg.MODEL.PCC_MODE == 'soft_min'):
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] loss_total: {:.3f}, loss_id: {:.3f}, "
                    "loss_triplet: {:.3f}, loss_camera_triplet: {:.3f}, "
                    "loss_cross_camera_positive: {:.3f}, loss_pcc: {:.3f}, "
                    "cross_camera_positive_count: {:.1f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(
                        engine.state.epoch, iteration_in_epoch,
                        _engine_epoch_length(engine),
                        engine.state.metrics['avg_loss'],
                        engine.state.metrics['avg_loss_id'],
                        engine.state.metrics['avg_loss_triplet'],
                        engine.state.metrics['avg_loss_camera_triplet'],
                        engine.state.metrics['avg_loss_cross_camera_positive'],
                        engine.state.metrics['avg_loss_pcc'],
                        engine.state.metrics['avg_cross_camera_positive_count'],
                        engine.state.metrics['avg_acc'],
                        scheduler.get_lr()[0],
                    )
                )
                logger.info(
                    "Soft Alignment Batch - Epoch: {} Iteration: {} "
                    "soft_alignment_loss: {:.6f} "
                    "valid_alignment_pair_count: {} "
                    "mean_soft_path_cost: {:.6f} "
                    "alignment_temperature: {:.12g}"
                    .format(
                        engine.state.epoch,
                        iteration_in_epoch,
                        float(engine.state.output['soft_alignment_loss']),
                        int(engine.state.output['valid_alignment_pair_count']),
                        float(engine.state.output['mean_soft_path_cost']),
                        float(engine.state.output['alignment_temperature']),
                    )
                )
            elif (cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY
                    and cfg.MODEL.PCC_MODE == 'windowed_soft_min'):
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] loss_total: {:.3f}, loss_id: {:.3f}, "
                    "loss_triplet: {:.3f}, loss_camera_triplet: {:.3f}, "
                    "loss_cross_camera_positive: {:.3f}, loss_pcc: {:.3f}, "
                    "cross_camera_positive_count: {:.1f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(
                        engine.state.epoch, iteration_in_epoch,
                        _engine_epoch_length(engine),
                        engine.state.metrics['avg_loss'],
                        engine.state.metrics['avg_loss_id'],
                        engine.state.metrics['avg_loss_triplet'],
                        engine.state.metrics['avg_loss_camera_triplet'],
                        engine.state.metrics['avg_loss_cross_camera_positive'],
                        engine.state.metrics['avg_loss_pcc'],
                        engine.state.metrics['avg_cross_camera_positive_count'],
                        engine.state.metrics['avg_acc'],
                        scheduler.get_lr()[0],
                    )
                )
                logger.info(
                    "Windowed Soft Alignment Batch - Epoch: {} Iteration: {} "
                    "window: {} alignment_temperature: {:.12g} "
                    "windowed_soft_alignment_loss: {:.6f} "
                    "valid_alignment_pair_count: {} "
                    "mean_windowed_soft_path_cost: {:.6f}"
                    .format(
                        engine.state.epoch,
                        iteration_in_epoch,
                        int(engine.state.output['alignment_window']),
                        float(engine.state.output['alignment_temperature']),
                        float(engine.state.output['windowed_soft_alignment_loss']),
                        int(engine.state.output['valid_alignment_pair_count']),
                        float(engine.state.output['mean_windowed_soft_path_cost']),
                    )
                )
            else:
                logger.info("Epoch[{}] Iteration[{}/{}] loss_total: {:.3f}, loss_id: {:.3f}, "
                            "loss_triplet: {:.3f}, loss_camera_triplet: {:.3f}, "
                            "loss_cross_camera_positive: {:.3f}, "
                            "cross_camera_positive_count: {:.1f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(engine.state.epoch, iteration_in_epoch,
                                    _engine_epoch_length(engine),
                                    engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss_id'],
                                    engine.state.metrics['avg_loss_triplet'],
                                    engine.state.metrics['avg_loss_camera_triplet'],
                                    engine.state.metrics['avg_loss_cross_camera_positive'],
                                    engine.state.metrics['avg_cross_camera_positive_count'],
                                    engine.state.metrics['avg_acc'],
                                    scheduler.get_lr()[0]))
            if (cfg.MODEL.CAMERA_AWARE_TRIPLET or cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY) and engine.state.output['cross_camera_positive_count'] == 0:
                logger.info("No cross-camera positive anchors in current batch; cross-camera auxiliary loss is skipped.")

    if cfg.MODEL.PART_CORRESPONDENCE_CONSISTENCY:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_pcc_epoch_summary(engine):
            pair_count = int(engine.state.pcc_epoch_pair_count)
            if cfg.MODEL.PCC_MODE == 'fixed_index':
                mean_distance = (
                    engine.state.pcc_epoch_distance_sum / float(pair_count)
                    if pair_count else 0.0
                )
                logger.info(
                    "PCC Epoch Summary - Epoch: {} valid_pcc_pair_count: {} "
                    "mean_fixed_index_part_distance: {:.6f}"
                    .format(engine.state.epoch, pair_count, mean_distance)
                )
            elif cfg.MODEL.PCC_MODE == 'hard_shortest_path':
                denominator = float(pair_count) if pair_count else 1.0
                logger.info(
                    "Hard Alignment Epoch Summary - Epoch: {} "
                    "hard_alignment_loss: {:.6f} "
                    "valid_alignment_pair_count: {} "
                    "mean_hard_path_cost: {:.6f} "
                    "mean_path_absolute_offset: {:.6f}"
                    .format(
                        engine.state.epoch,
                        engine.state.hard_epoch_loss_sum / denominator,
                        pair_count,
                        engine.state.hard_epoch_cost_sum / denominator,
                        engine.state.hard_epoch_offset_sum / denominator,
                    )
                )
            elif cfg.MODEL.PCC_MODE == 'soft_min':
                denominator = float(pair_count) if pair_count else 1.0
                logger.info(
                    "Soft Alignment Epoch Summary - Epoch: {} "
                    "soft_alignment_loss: {:.6f} "
                    "valid_alignment_pair_count: {} "
                    "mean_soft_path_cost: {:.6f} "
                    "alignment_temperature: {:.12g}"
                    .format(
                        engine.state.epoch,
                        engine.state.soft_epoch_loss_sum / denominator,
                        pair_count,
                        engine.state.soft_epoch_cost_sum / denominator,
                        float(cfg.MODEL.PCC_SOFTMIN_TAU),
                    )
                )
            elif cfg.MODEL.PCC_MODE == 'windowed_soft_min':
                denominator = float(pair_count) if pair_count else 1.0
                logger.info(
                    "Windowed Soft Alignment Epoch Summary - Epoch: {} "
                    "window: {} alignment_temperature: {:.12g} "
                    "windowed_soft_alignment_loss: {:.6f} "
                    "valid_alignment_pair_count: {} "
                    "mean_windowed_soft_path_cost: {:.6f}"
                    .format(
                        engine.state.epoch,
                        int(cfg.MODEL.PCC_SOFTMIN_WINDOW),
                        float(cfg.MODEL.PCC_SOFTMIN_TAU),
                        engine.state.windowed_soft_epoch_loss_sum / denominator,
                        pair_count,
                        engine.state.windowed_soft_epoch_cost_sum / denominator,
                    )
                )

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(
        output_dir,
        cfg.MODEL.NAME,
        n_saved=10,
        require_empty=False
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=checkpoint_period),
        checkpointer,
        {
            'model': model,
            'optimizer': optimizer,
            'center_param': center_criterion,
            'optimizer_center': optimizer_center
        }
    )
    epoch_log_state = _attach_epoch_evidence_logging(trainer, logger)
    timer = Timer(average=True)

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x['loss_total']).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x['loss_id']).attach(trainer, 'avg_loss_id')
    RunningAverage(output_transform=lambda x: x['loss_triplet']).attach(trainer, 'avg_loss_triplet')
    RunningAverage(output_transform=lambda x: x['loss_camera_triplet']).attach(trainer, 'avg_loss_camera_triplet')
    RunningAverage(output_transform=lambda x: x['loss_cross_camera_positive']).attach(trainer, 'avg_loss_cross_camera_positive')
    RunningAverage(output_transform=lambda x: x['cross_camera_positive_count']).attach(trainer, 'avg_cross_camera_positive_count')
    RunningAverage(output_transform=lambda x: x['acc']).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iteration_in_epoch = epoch_log_state["iteration"]

        if iteration_in_epoch % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] loss_total: {:.3f}, loss_id: {:.3f}, "
                        "loss_triplet: {:.3f}, loss_camera_triplet: {:.3f}, "
                        "loss_cross_camera_positive: {:.3f}, "
                        "cross_camera_positive_count: {:.1f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iteration_in_epoch,
                                _engine_epoch_length(engine),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss_id'],
                                engine.state.metrics['avg_loss_triplet'],
                                engine.state.metrics['avg_loss_camera_triplet'],
                                engine.state.metrics['avg_loss_cross_camera_positive'],
                                engine.state.metrics['avg_cross_camera_positive_count'],
                                engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
            if (cfg.MODEL.CAMERA_AWARE_TRIPLET or cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY) and engine.state.output['cross_camera_positive_count'] == 0:
                logger.info("No cross-camera positive anchors in current batch; cross-camera auxiliary loss is skipped.")

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)
