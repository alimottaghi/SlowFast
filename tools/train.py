#!/usr/bin/env python3
"""Train a classification model."""
import math
import copy
import numpy as np
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.models import build_model
from slowfast.utils.meters import TrainMeter, ValMeter, EpochTimer
from slowfast.datasets import loader
from slowfast.datasets import utils as data_utils

logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, scaler, train_meter, cur_epoch, cfg, writer=None):
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    mb_size = cfg.TRAIN.BATCH_SIZE * max(cfg.NUM_GPUS, 1)
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        inputs, labels, meta = transfer_to_device(inputs, labels, meta, cfg)

        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs)
            loss = loss_fun(preds, labels)
        perform_backward_pass(optimizer, scaler, loss, model, cfg)
        loss, top1_err, top5_err = calculate_errors(loss, preds, labels, cfg)
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        write_to_tensorboard(writer, loss, lr, data_size, cur_epoch, cur_iter, cfg, inputs, preds, labels, top1_err, top5_err, tag="Train")

        train_meter.iter_toc()
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.update_predictions(preds, labels)
        torch.cuda.synchronize()
        train_meter.iter_tic()

    clear_memory(inputs, labels, preds, loss)
    log_epoch_stats(train_meter, cur_epoch, writer, cfg, tag="Train")
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    model.eval()
    val_meter.iter_tic()
    mb_size = cfg.TRAIN.BATCH_SIZE * max(cfg.NUM_GPUS, 1)

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        inputs, labels, meta = transfer_to_device(inputs, labels, meta, cfg)
        val_meter.data_toc()

        preds = model(inputs)

        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                preds, labels = du.all_gather([preds, labels])
        else:
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()
            val_meter.update_stats(top1_err, top5_err, mb_size)
            write_to_tensorboard(writer, None, None, len(val_loader), cur_epoch, cur_iter, cfg, inputs, preds, labels, top1_err, top5_err, tag="Val")

        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    log_epoch_stats(val_meter, cur_epoch, writer, cfg, tag="Val")
    val_meter.reset()


def transfer_to_device(inputs, labels, meta, cfg):
    if cfg.NUM_GPUS:
        inputs = [input.cuda(non_blocking=True) for input in inputs] if isinstance(inputs, (list,)) else inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        for key, val in meta.items():
            meta[key] = [v.cuda(non_blocking=True) for v in val] if isinstance(val, (list,)) else val.cuda(non_blocking=True)
    return inputs, labels, meta


def perform_backward_pass(optimizer, scaler, loss, model, cfg):
    misc.check_nan_losses(loss)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if cfg.SOLVER.CLIP_GRAD_VAL:
        torch.nn.utils.clip_grad_value_(model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL)
    elif cfg.SOLVER.CLIP_GRAD_L2NORM:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM)
    scaler.step(optimizer)
    scaler.update()


def calculate_errors(loss, preds, labels, cfg):
    top1_err, top5_err = None, None
    if cfg.DATA.MULTI_LABEL:
        if cfg.NUM_GPUS > 1:
            [loss] = du.all_reduce([loss])
        loss = loss.item()
    else:
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
    return loss, top1_err, top5_err


def write_to_tensorboard(writer, loss, lr, data_size, cur_epoch, cur_iter, cfg, inputs, preds, labels, top1_err, top5_err, tag="Train"):
    if writer is not None:
        if cur_iter % cfg.LOG_PERIOD == 0:
            writer.add_scalars(
                {
                    f"{tag}/loss": loss,
                    f"{tag}/lr": lr,
                    f"{tag}/Top1_err": top1_err,
                    f"{tag}/Top5_err": top5_err,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

        if cfg.TENSORBOARD.SAMPLE_VIS.ENABLE and (cur_iter) % cfg.TENSORBOARD.SAMPLE_VIS.LOG_PERIOD == 0:
            writer.add_video_pred(
                inputs[0],
                torch.argmax(preds, dim=1),
                labels,
                tag=f"Sample/{tag}",
                global_step=data_size * cur_epoch + cur_iter,
            )


def clear_memory(*args):
    for arg in args:
        del arg
    torch.cuda.empty_cache()


def log_epoch_stats(meter, cur_epoch, writer, cfg, tag="Train"):
    meter.log_epoch_stats(cur_epoch)
    if writer is not None and cfg.TENSORBOARD.EPOCH_LOG.ENABLE:
        writer.writer.add_scalars(
            "Error/Top1_err",
            {tag: meter.num_top1_mis / meter.num_samples}, global_step=cur_epoch
        )
        writer.writer.add_scalars(
            "Error/Top5_err",
            {tag: meter.num_top5_mis / meter.num_samples}, global_step=cur_epoch
        )
        all_preds = [pred.clone().detach() for pred in meter.all_preds]
        all_labels = [label.clone().detach() for label in meter.all_labels]
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(
            preds=all_preds,
            labels=all_labels,
            global_step=cur_epoch,
            tag=f"Confusion/{tag}"
        )


def log_epoch_time(cur_epoch, start_epoch, epoch_timer, loader_len):
    logger.info(f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
                f"from {start_epoch} to {cur_epoch} take "
                f"{epoch_timer.avg_epoch_time():.2f}s in average and "
                f"{epoch_timer.median_epoch_time():.2f}s in median.")
    logger.info(f"For epoch {cur_epoch}, each iteraction takes "
                f"{epoch_timer.last_epoch_time()/loader_len:.2f}s in average. "
                f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
                f"{epoch_timer.avg_epoch_time()/loader_len:.2f}s in average.")


def calculate_and_update_precise_bn(loader, model, num_iters, cfg):
    def _gen_loader():
        for inputs, _, _ in loader:
            inputs, _, _ = transfer_to_device(inputs, None, None, cfg)
            yield inputs

    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    du.init_distributed_training(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    writer = tb.TensorboardWriter(cfg) if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS) else None
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    optimizer = optim.construct_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None)

    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(cfg, "train", is_precise_bn=True) if cfg.BN.USE_PRECISE_STATS else None

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    logger.info("Start epoch: {}".format(start_epoch + 1))
    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        loader.shuffle_dataset(train_loader, cur_epoch)
        epoch_timer.epoch_tic()
        train_epoch(train_loader, model, optimizer, scaler, train_meter, cur_epoch, cfg, writer)
        epoch_timer.epoch_toc()
        log_epoch_time(cur_epoch, start_epoch, epoch_timer, len(train_loader))

        is_checkp_epoch = cu.is_checkpoint_epoch(cfg, cur_epoch, None)
        is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch, None)

        if (is_checkp_epoch or is_eval_epoch) and cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(precise_bn_loader, model, min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)), cfg)
        _ = misc.aggregate_sub_bn_stats(model)

        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, scaler if cfg.TRAIN.MIXED_PRECISION else None)

        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()

    raise SystemExit('Training Ends')
