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
from utils.experiment_recording import append_validation_record, utc_now
from utils.dynamic_gating_evidence import (
    GatingEpochAccumulator,
    append_gating_epoch_record,
)

global ITER
ITER = 0


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
    }


def _item(value):
    if torch.is_tensor(value):
        return value.item()
    return value


def _model_gating_evidence(model):
    module = model.module if isinstance(model, nn.DataParallel) else model
    evidence = getattr(module, '_last_dynamic_gating', None)
    if not evidence:
        return None
    return {
        key: value.detach().to('cpu') if torch.is_tensor(value) else value
        for key, value in evidence.items()
    }


def _engine_epoch_length(engine):
    epoch_length = engine.state.epoch_length
    if epoch_length is None or int(epoch_length) <= 0:
        raise RuntimeError(
            'Ignite engine.state.epoch_length must be a positive integer'
        )
    return int(epoch_length)


def _attach_epoch_evidence_logging(trainer, logger):
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_evidence(engine):
        logger.info(
            'EPOCH_EVIDENCE epoch={} global_iteration={} epoch_length={}'
            .format(
                int(engine.state.epoch), int(engine.state.iteration),
                _engine_epoch_length(engine),
            )
        )


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
        score, feat = model(img)
        loss_output = loss_fn(score, feat, target, camids)
        loss, loss_dict = _loss_output_to_dict(loss_output)
        loss.backward()
        optimizer.step()
        gating_evidence = _model_gating_evidence(model)
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return {
            'loss_total': _item(loss_dict['loss_total']),
            'loss_id': _item(loss_dict['loss_id']),
            'loss_triplet': _item(loss_dict['loss_triplet']),
            'loss_camera_triplet': _item(loss_dict['loss_camera_triplet']),
            'loss_cross_camera_positive': _item(loss_dict.get('loss_cross_camera_positive', loss_dict['loss_total'].new_tensor(0.0) if torch.is_tensor(loss_dict['loss_total']) else 0.0)),
            'cross_camera_positive_count': loss_dict['cross_camera_positive_count'],
            'gating_probabilities': (
                gating_evidence['probabilities']
                if gating_evidence is not None else None
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
        score, feat = model(img)
        loss_output = loss_fn(score, feat, target, camids)
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
    _attach_epoch_evidence_logging(trainer, logger)
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
        if cfg.MODEL.MULTI_GRANULARITY_DYNAMIC_GATING:
            engine.state.dynamic_gating_accumulator = GatingEpochAccumulator(
                cfg.MODEL.MULTI_GRANULARITY_GATING_TAU
            )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if cfg.MODEL.MULTI_GRANULARITY_DYNAMIC_GATING:
            probabilities = engine.state.output.get('gating_probabilities')
            if probabilities is None:
                raise RuntimeError(
                    'Dynamic gating is enabled but the model emitted no gate evidence'
                )
            engine.state.dynamic_gating_accumulator.update(probabilities)

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] loss_total: {:.3f}, loss_id: {:.3f}, "
                        "loss_triplet: {:.3f}, loss_camera_triplet: {:.3f}, "
                        "loss_cross_camera_positive: {:.3f}, "
                        "cross_camera_positive_count: {:.1f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss_id'],
                                engine.state.metrics['avg_loss_triplet'],
                                engine.state.metrics['avg_loss_camera_triplet'],
                                engine.state.metrics['avg_loss_cross_camera_positive'],
                                engine.state.metrics['avg_cross_camera_positive_count'],
                                engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
            if (cfg.MODEL.CAMERA_AWARE_TRIPLET or cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY) and engine.state.output['cross_camera_positive_count'] == 0:
                logger.info("No cross-camera positive anchors in current batch; cross-camera auxiliary loss is skipped.")
        if len(train_loader) == ITER:
            ITER = 0

    if cfg.MODEL.MULTI_GRANULARITY_DYNAMIC_GATING:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_dynamic_gating_epoch_summary(engine):
            statistics = engine.state.dynamic_gating_accumulator.summary()
            append_gating_epoch_record(
                output_dir,
                engine.state.epoch,
                engine.state.iteration,
                _engine_epoch_length(engine),
                statistics,
            )
            logger.info(
                'DYNAMIC_GATING_EPOCH {}'.format(
                    ' '.join(
                        '{}={:.12g}'.format(key, value)
                        if isinstance(value, float)
                        else '{}={}'.format(key, value)
                        for key, value in statistics.items()
                    )
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
            append_validation_record(output_dir, {
                'epoch': int(engine.state.epoch),
                'global_iteration': int(engine.state.iteration),
                'timestamp_utc': utc_now(),
                'rank1_percent': float(cmc[0]) * 100.0,
                'rank5_percent': float(cmc[4]) * 100.0,
                'rank10_percent': float(cmc[9]) * 100.0,
                'map_percent': float(mAP) * 100.0,
                're_ranking': str(cfg.TEST.RE_RANKING),
                'neck_feat': str(cfg.TEST.NECK_FEAT),
                'feat_norm': str(cfg.TEST.FEAT_NORM),
            })

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
    _attach_epoch_evidence_logging(trainer, logger)
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
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] loss_total: {:.3f}, loss_id: {:.3f}, "
                        "loss_triplet: {:.3f}, loss_camera_triplet: {:.3f}, "
                        "loss_cross_camera_positive: {:.3f}, "
                        "cross_camera_positive_count: {:.1f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss_id'],
                                engine.state.metrics['avg_loss_triplet'],
                                engine.state.metrics['avg_loss_camera_triplet'],
                                engine.state.metrics['avg_loss_cross_camera_positive'],
                                engine.state.metrics['avg_cross_camera_positive_count'],
                                engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
            if (cfg.MODEL.CAMERA_AWARE_TRIPLET or cfg.MODEL.CROSS_CAMERA_POSITIVE_ONLY) and engine.state.output['cross_camera_positive_count'] == 0:
                logger.info("No cross-camera positive anchors in current batch; cross-camera auxiliary loss is skipped.")
        if len(train_loader) == ITER:
            ITER = 0

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
            append_validation_record(output_dir, {
                'epoch': int(engine.state.epoch),
                'global_iteration': int(engine.state.iteration),
                'timestamp_utc': utc_now(),
                'rank1_percent': float(cmc[0]) * 100.0,
                'rank5_percent': float(cmc[4]) * 100.0,
                'rank10_percent': float(cmc[9]) * 100.0,
                'map_percent': float(mAP) * 100.0,
                're_ranking': str(cfg.TEST.RE_RANKING),
                'neck_feat': str(cfg.TEST.NECK_FEAT),
                'feat_norm': str(cfg.TEST.FEAT_NORM),
            })

    trainer.run(train_loader, max_epochs=epochs)
