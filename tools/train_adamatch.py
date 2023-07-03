#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

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
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import EpochTimer, TrainMeter, ValMeter, AdaMeter, AdaEmbedMeter

logger = logging.get_logger(__name__)


def train_epoch(
    train_loaders, 
    model, 
    optimizer, 
    scaler, 
    train_meter, 
    cur_epoch, 
    cfg, 
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loaders (list of loader): source and target video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    source_loader = train_loaders[0]
    target_unl_loader = train_loaders[1]
    if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
        target_lab_loader = train_loaders[2]
    
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(source_loader)
    target_unl_iter = iter(target_unl_loader)
    target_unl_size = len(target_unl_loader)
    if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
        target_lab_iter = iter(target_lab_loader)
        target_lab_size = len(target_lab_loader)

    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    loss_fun_none = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="none")

    for cur_iter, (inputs_source, labels_source, _, _) in enumerate(source_loader):
        # Load the data.
        if cur_iter%target_unl_size==0:
            target_unl_iter = iter(target_unl_loader)
        inputs_target_unl, labels_target_unl, _, _ = next(target_unl_iter)
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            if cur_iter%target_lab_size==0:
                target_lab_iter = iter(target_lab_loader)
            inputs_target_lab, labels_target_lab, _, _ = next(target_lab_iter)
        
        # Transfer the data to the current GPU device.
        for i in range(len(inputs_source)):
            inputs_source[i] = inputs_source[i].cuda(non_blocking=True)
            inputs_target_unl[i] = inputs_target_unl[i].cuda(non_blocking=True)
        labels_source = labels_source.cuda()
        labels_target_unl = labels_target_unl.cuda()
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            for i in range(len(inputs_source)):
                inputs_target_lab[i] = inputs_target_lab[i].cuda(non_blocking=True)
            labels_target_lab = labels_target_lab.cuda()

        # Update the learning rate and mu.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        mu = (0.5 + math.cos(math.pi * (cfg.SOLVER.MAX_EPOCH - cur_epoch - float(cur_iter) / data_size) / cfg.SOLVER.MAX_EPOCH) / 2)

        # Generate alignment for predictions
        if cfg.ADAMATCH.ALIGNMENT.ENABLE and len(train_meter.all_lab_preds) > 10:
            p_source = F.softmax(torch.cat(train_meter.all_lab_preds, dim=0), dim=1).mean(dim=0).detach()
            p_target = F.softmax(torch.cat(train_meter.all_unl_preds, dim=0), dim=1).mean(dim=0).detach()
            alignment = (1e-6 + p_source) / (1e-6 + p_target)
        else:
            alignment = torch.ones(cfg.MODEL.NUM_CLASSES).cuda()

        # Generate the sampling rate for pseudo labels.
        if cfg.ADAMATCH.SAMPLING.ENABLE and len(train_meter.all_pseudo_labels) > 10:
            all_pseudo_labels = train_meter.all_pseudo_labels.copy()
            all_pseudo_labels.extend([torch.tensor(i).cuda() for i in range(cfg.MODEL.NUM_CLASSES)])
            all_pseudo_labels = torch.stack(all_pseudo_labels, dim=0).detach()
            freq = torch.bincount(all_pseudo_labels)
            p_sample = torch.minimum((torch.min(freq) + 1) / (freq + 1), torch.ones_like(freq))
        else:
            p_sample = torch.ones(cfg.MODEL.NUM_CLASSES).cuda()

        # Generate threshold.
        if cfg.ADAMATCH.THRESHOLDING.ENABLE and len(train_meter.all_lab_preds) > 10:
            if cfg.ADAMATCH.THRESHOLDING.TYPE=='uniform':
                c_tau = cfg.ADAMATCH.THRESHOLDING.TAU * torch.ones(cfg.MODEL.NUM_CLASSES).cuda()
            elif cfg.ADAMATCH.THRESHOLDING.TYPE=='relative':
                max_preds = F.softmax(torch.cat(train_meter.all_lab_preds, dim=0), dim=1).max(dim=1)[0].detach()
                c_tau = cfg.ADAMATCH.THRESHOLDING.TAU * max_preds.mean(dim=0) * torch.ones(cfg.MODEL.NUM_CLASSES).cuda()
                c_tau = torch.minimum(c_tau, 0.99 * torch.ones_like(c_tau))
            elif cfg.ADAMATCH.THRESHOLDING.TYPE=='adaptive':
                all_preds = F.softmax(torch.cat(train_meter.all_lab_preds, dim=0), dim=1).detach()
                max_preds = all_preds.max(dim=1)[0]
                class_preds = torch.argmax(all_preds, dim=1)
                c_tau = torch.stack([cfg.ADAMATCH.THRESHOLDING.TAU * max_preds[class_preds==i].mean(dim=0) for i in range(cfg.MODEL.NUM_CLASSES)])
                c_tau = torch.minimum(c_tau, 0.99 * torch.ones_like(c_tau))
                c_tau = torch.maximum(c_tau, 0.8 * torch.ones_like(c_tau))
            elif cfg.ADAMATCH.THRESHOLDING.TYPE=='elastic':
                all_preds = F.softmax(torch.cat(train_meter.all_lab_preds, dim=0), dim=1).detach()
                max_preds = all_preds.max(dim=1)[0]
                class_preds = torch.argmax(all_preds, dim=1)
                c_tau = torch.stack([max_preds[class_preds==i].mean(dim=0) for i in range(cfg.MODEL.NUM_CLASSES)])
                c_tau = cfg.ADAMATCH.THRESHOLDING.TAU * c_tau / torch.max(c_tau)
        else:
            c_tau = torch.ones(cfg.MODEL.NUM_CLASSES).cuda()

        train_meter.data_toc()
        source_weak = inputs_source[0]
        source_strong = inputs_source[1]
        target_unl_weak = inputs_target_unl[0]
        target_unl_strong = inputs_target_unl[1]
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            target_lab_weak = inputs_target_lab[0]
            target_lab_strong = inputs_target_lab[1]
        
        if not cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            inputs_weak = torch.cat((source_weak, target_unl_weak), dim=0)
            inputs_strong = torch.cat((source_strong, target_unl_strong), dim=0)
        else:
            inputs_weak = torch.cat((source_weak, target_lab_weak, target_unl_weak), dim=0)
            inputs_strong = torch.cat((source_strong, target_lab_strong, target_unl_strong), dim=0)

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # Forward pass.
            optimizer.zero_grad()

            # Compute the predictions.
            logists_strong = model([inputs_strong])
            with torch.no_grad():
                logists_weak = model([inputs_weak])
            
            if not cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
                logits_sls, logits_tus = torch.split(logists_strong, [len(source_strong),len(target_unl_strong)])
                logits_slw, logits_tuw = torch.split(logists_weak, [len(source_weak),len(target_unl_weak)])
            else:
                logits_sls, logits_tls, logits_tus = torch.split(logists_strong, [len(source_strong),len(target_lab_strong),len(target_unl_strong)])
                logits_slw, logits_tlw, logits_tuw = torch.split(logists_weak, [len(source_weak),len(target_lab_weak),len(target_unl_weak)])

            preds_tuw = F.softmax(logits_tuw, dim=1)
            preds_tuw_refined = F.normalize(alignment * preds_tuw, p=1, dim=1)
            pseudo_labels = torch.argmax(preds_tuw_refined, dim=1)
            pred_mask = preds_tuw_refined.max(dim=1)[0] >= c_tau[pseudo_labels]
            sampling_mask = torch.bernoulli(p_sample[pseudo_labels]) > 0
            mask = pred_mask * sampling_mask

            # Compute the loss.
            if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
                logits_labeled = torch.cat((logits_sls, logits_tls), dim=0)
                labels_labeled = torch.cat((labels_source, labels_target_lab), dim=0)
            else:
                logits_labeled = logits_sls
                labels_labeled = labels_source
            loss_s = loss_fun(logits_labeled, labels_labeled)
            loss_t = loss_fun_none(logits_tus, pseudo_labels)
            loss_t = (mask * loss_t).mean(dim=0)
            loss = loss_s + cfg.ADAMATCH.MU * mu * loss_t
            loss.backward()
            optimizer.step()

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(logits_sls, labels_source, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / logits_sls.size(0)) * 100.0 for x in num_topks_correct
        ]
        num_pred_mask = torch.sum(pred_mask)
        num_sampling_mask = torch.sum(sampling_mask)
        num_pseudo = torch.sum(mask)
        num_correct = torch.sum(pseudo_labels[mask]==labels_target_unl[mask])
        mean_tau = torch.mean(c_tau)
        
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, loss_s, loss_t, top1_err, top5_err = du.all_reduce(
                [loss, loss_s, loss_t, top1_err, top5_err]
            )
            num_pred_mask, num_sampling_mask, num_pseudo, num_correct = du.all_reduce(
                [num_pred_mask, num_sampling_mask, num_pseudo, num_correct], 
                average=False,
            )

        # Copy the stats from GPU to CPU (sync point).
        loss, loss_s, loss_t, top1_err, top5_err = (
            loss.item(),
            loss_s.item(),
            loss_t.item(),
            top1_err.item(),
            top5_err.item()
        )
        num_pred_mask, num_sampling_mask, num_pseudo, num_correct, mean_tau = (
            num_pred_mask.item(), 
            num_sampling_mask.item(),
            num_pseudo.item(), 
            num_correct.item(),
            mean_tau.item()
        )

        batch_size = inputs_source[0].size(0)*max(cfg.NUM_GPUS, 1)
        pred_mask_mis = (1 - num_pred_mask / (batch_size * cfg.ADAPTATION.BETA)) * 100.0
        sampling_mask_mis = (1 - num_sampling_mask / (batch_size * cfg.ADAPTATION.BETA)) * 100.0
        pseudo_mis = (1 - num_pseudo / (batch_size * cfg.ADAPTATION.BETA)) * 100.0
        pseudo_err = (1 - num_correct / num_pseudo) * 100.0 if num_pseudo > 0 else None

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss_s,
            loss_t,
            lr,
            batch_size,
        )
        # write to tensorboard format if available.
        if writer is not None:
            dict2write = {
                "Train/loss": loss,
                "Train/loss_s": loss_s,
                "Train/loss_t": loss_t,
                "Train/lr": lr,
                "Train/Top1_err": top1_err,
                "Train/Top5_err": top5_err,
                "Ada/mu": mu, 
                "Ada/pred_mask_mis": pred_mask_mis,
                "Ada/sampling_mask_mis": sampling_mask_mis,
                "Ada/pseudo_mis": pseudo_mis,
                "Ada/tau": mean_tau,
            }
            if pseudo_err is not None:
                dict2write["Ada/pseudo_err"] = pseudo_err
            writer.add_scalars(dict2write, global_step=data_size * cur_epoch + cur_iter)

            if cfg.TENSORBOARD.DIST_VIS.ENABLE and (data_size * cur_epoch + cur_iter)%cfg.TENSORBOARD.DIST_VIS.LOG_PERIOD==10:
                writer.add_confusion_matrix(
                    torch.argmax(torch.cat(train_meter.all_lab_preds, dim=0), dim=1), 
                    torch.cat(train_meter.all_lab_labels, dim=0), 
                    tag="Confusion/Labeled", 
                    global_step=data_size * cur_epoch + cur_iter
                )
                writer.add_confusion_matrix(
                    torch.argmax(torch.cat(train_meter.all_unl_preds, dim=0), dim=1), 
                    torch.cat(train_meter.all_unl_labels, dim=0), 
                    tag="Confusion/Unlabeled", 
                    global_step=data_size * cur_epoch + cur_iter
                )
                writer.add_confusion_matrix(
                    torch.stack(train_meter.all_true_labels, dim=0), 
                    torch.stack(train_meter.all_pseudo_labels, dim=0), 
                    tag="Confusion/PseudoLabel", 
                    global_step=data_size * cur_epoch + cur_iter
                )
                writer.add_distribution(
                    F.softmax(torch.cat(train_meter.all_lab_preds, dim=0), dim=1), 
                    tag="Distribution/Labeled", 
                    global_step=data_size * cur_epoch + cur_iter
                )
                writer.add_distribution(
                    F.softmax(torch.cat(train_meter.all_unl_preds, dim=0), dim=1), 
                    tag="Distribution/Unlabeled", 
                    global_step=data_size * cur_epoch + cur_iter
                )
                writer.add_distribution(
                    torch.cat(train_meter.all_unl_preds_refined, dim=0), 
                    tag="Distribution/Refined", 
                    global_step=data_size * cur_epoch + cur_iter
                )
                writer.add_distribution(
                    torch.stack(train_meter.all_pseudo_labels, dim=0), 
                    tag="Distribution/PseudoLabel", 
                    global_step=data_size * cur_epoch + cur_iter
                )
                writer.add_distribution(
                    c_tau, 
                    tag="Distribution/Tau", 
                    global_step=data_size * cur_epoch + cur_iter
                )
                writer.add_distribution(
                    p_sample, 
                    tag="Distribution/Sampling", 
                    global_step=data_size * cur_epoch + cur_iter
                )

            if cfg.TENSORBOARD.SAMPLE_VIS.ENABLE and (data_size * cur_epoch + cur_iter)%cfg.TENSORBOARD.SAMPLE_VIS.LOG_PERIOD==0:
                writer.add_video_pred(
                    source_strong, 
                    torch.argmax(logits_sls, dim=1), 
                    labels_source,
                    tag="Sample/Source",
                    global_step = data_size * cur_epoch + cur_iter,
                )
                writer.add_video_pred(
                    target_unl_strong, 
                    pseudo_labels, 
                    labels_target_unl,
                    mask,
                    tag="Sample/Target",
                    global_step = data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.update_predictions(
            logits_sls, logits_slw, labels_source, 
            logits_tus, logits_tuw, labels_target_unl, preds_tuw_refined,
        )
        train_meter.update_pseudos(labels_target_unl, pseudo_labels, mask)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()
        del inputs_source, inputs_target_unl, labels_source, labels_target_unl
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            del inputs_target_lab, labels_target_lab

        # in case of fragmented memory
        torch.cuda.empty_cache()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.TENSORBOARD.EPOCH_LOG.ENABLE:
            writer.writer.add_scalars(
                "Error/Top1_err",
                {"Train": train_meter.num_top1_mis / train_meter.num_samples}, global_step=cur_epoch
            )
            writer.writer.add_scalars(
                "Error/Top5_err",
                {"Train": train_meter.num_top5_mis / train_meter.num_samples}, global_step=cur_epoch
            )
        if cfg.TENSORBOARD.CONFUSION_MATRIX.ENABLE:
            all_preds = [pred.clone().detach() for pred in train_meter.all_lab_preds]
            all_labels = [label.clone().detach() for label in train_meter.all_lab_labels]
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, 
                labels=all_labels, 
                global_step=cur_epoch, 
                tag="Confusion/Train"
            )
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
    val_loader, model, val_meter, cur_epoch, cfg, writer=None
    ):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        preds = model(inputs)

        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                preds, labels = du.all_gather([preds, labels])
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

            # Combine the errors across the GPUs.
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])

            # Copy the errors from GPU to CPU (sync point).
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

                if cfg.TENSORBOARD.SAMPLE_VIS.ENABLE and (len(val_loader) * cur_epoch + cur_iter)%cfg.TENSORBOARD.SAMPLE_VIS.LOG_PERIOD==0:
                    writer.add_video_pred(
                        inputs[0], 
                        torch.argmax(preds, dim=1), 
                        labels,
                        tag="Sample/Val",
                        global_step = len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.TENSORBOARD.EPOCH_LOG.ENABLE:
            writer.writer.add_scalars(
                "Error/Top1_err",
                {"Val": val_meter.num_top1_mis / val_meter.num_samples}, global_step=cur_epoch
            )
            writer.writer.add_scalars(
                "Error/Top5_err",
                {"Val": val_meter.num_top5_mis / val_meter.num_samples}, global_step=cur_epoch
            )
        if cfg.TENSORBOARD.CONFUSION_MATRIX.ENABLE:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, 
                labels=all_labels, 
                global_step=cur_epoch, 
                tag="Confusion/Val"
            )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer,
        scaler if cfg.TRAIN.MIXED_PRECISION else None)

    # Create the video train and val loaders.
    if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
        source_cfg = copy.deepcopy(cfg)
        source_cfg.DATA.IMDB_FILES.TRAIN = cfg.ADAPTATION.SOURCE
        source_cfg.DATA.IMDB_FILES.VAL = cfg.ADAPTATION.TARGET
        source_loader = loader.construct_loader(source_cfg, "train")
        val_loader = loader.construct_loader(source_cfg, "val")
        target_lab_cfg = copy.deepcopy(cfg)
        target_lab_cfg.DATA.IMDB_FILES.TRAIN = cfg.ADAPTATION.TARGET
        target_lab_cfg.DATA.IMDB_FILES.VAL = cfg.ADAPTATION.SOURCE
        target_lab_cfg.TRAIN.BATCH_SIZE = int(cfg.ADAPTATION.ALPHA * source_cfg.TRAIN.BATCH_SIZE)
        target_lab_loader = loader.construct_loader(target_lab_cfg, "lab")
        target_unl_cfg = copy.deepcopy(cfg)
        target_unl_cfg.DATA.IMDB_FILES.TRAIN = cfg.ADAPTATION.TARGET
        target_unl_cfg.DATA.IMDB_FILES.VAL = cfg.ADAPTATION.SOURCE
        target_unl_cfg.TRAIN.BATCH_SIZE = int(cfg.ADAPTATION.BETA * source_cfg.TRAIN.BATCH_SIZE)
        target_unl_loader = loader.construct_loader(target_unl_cfg, "unl")
        bn_cfg = copy.deepcopy(cfg)
        bn_cfg.DATA.IMDB_FILES.TRAIN = cfg.ADAPTATION.SOURCE + cfg.ADAPTATION.TARGET 
        bn_cfg.ADAMATCH.ENABLE = False
        precise_bn_loader = (
            loader.construct_loader(bn_cfg, "train", is_precise_bn=True)
            if cfg.BN.USE_PRECISE_STATS
            else None
        )
        train_loaders = [source_loader, target_unl_loader, target_lab_loader]
    else:
        source_cfg = copy.deepcopy(cfg)
        source_cfg.DATA.IMDB_FILES.TRAIN = cfg.ADAPTATION.SOURCE
        source_cfg.DATA.IMDB_FILES.VAL = cfg.ADAPTATION.TARGET
        source_loader = loader.construct_loader(source_cfg, "train")
        val_loader = loader.construct_loader(source_cfg, "val")
        target_unl_cfg = copy.deepcopy(cfg)
        target_unl_cfg.DATA.IMDB_FILES.TRAIN = cfg.ADAPTATION.TARGET
        target_unl_cfg.DATA.IMDB_FILES.VAL = cfg.ADAPTATION.SOURCE
        target_unl_cfg.TRAIN.BATCH_SIZE = int(cfg.ADAPTATION.BETA * source_cfg.TRAIN.BATCH_SIZE)
        target_unl_loader = loader.construct_loader(target_unl_cfg, "train")
        bn_cfg = copy.deepcopy(cfg)
        bn_cfg.DATA.IMDB_FILES.TRAIN = cfg.ADAPTATION.SOURCE + cfg.ADAPTATION.TARGET 
        bn_cfg.ADAMATCH.ENABLE = False
        precise_bn_loader = (
            loader.construct_loader(bn_cfg, "train", is_precise_bn=True)
            if cfg.BN.USE_PRECISE_STATS
            else None
        )
        train_loaders = [source_loader, target_unl_loader]
    
    # Create meters.
    train_meter = AdaEmbedMeter(len(train_loaders[0]), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Shuffle the dataset.
        for train_loader in train_loaders:
            loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loaders, 
            model, 
            optimizer, 
            scaler, 
            train_meter, 
            cur_epoch, 
            cfg, 
            writer,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loaders[0]):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loaders[0]):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, 
            cur_epoch, 
            None
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR, 
                model, 
                optimizer, 
                cur_epoch, 
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(
                val_loader, 
                model, 
                val_meter, 
                cur_epoch, 
                cfg, 
                writer,
            )

    if writer is not None:
        writer.close()
    raise SystemExit('Training Ends')
