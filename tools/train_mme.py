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
from slowfast.utils.meters import EpochTimer, TrainMeter, ValMeter, AdaMeter

logger = logging.get_logger(__name__)


class GradientReverse(torch.autograd.Function):
    scale = torch.tensor(1.0, requires_grad=False)
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()
    
def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)

def adentropy(classifier, feat, lamda, eta=1.0):
    logits = classifier(feat, reverse=True, eta=eta)
    preds = F.softmax(logits, dim=1)
    loss_adent = lamda * torch.mean(
        torch.sum(preds * (torch.log(preds + 1e-5)), 1))
    return loss_adent


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor, self).__init__()

        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.inc = inc
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out


def train_epoch(
    train_loaders, 
    model, 
    optimizers, 
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

    optimizer_f, optimizer_c = optimizers[0], optimizers[1]
    
    # Enable train mode.
    model.train()

    train_meter.iter_tic()
    data_size = len(source_loader)
    target_unl_iter = iter(target_unl_loader)
    target_unl_size = len(target_unl_loader)
    if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
        target_lab_iter = iter(target_lab_loader)
        target_lab_size = len(target_lab_loader)

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

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer_f, lr)
        optim.set_lr(optimizer_c, lr)

        train_meter.data_toc()
        source_weak = inputs_source[0]
        source_strong = inputs_source[1]
        target_unl_weak = inputs_target_unl[0]
        target_unl_strong = inputs_target_unl[1]
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            target_lab_weak = inputs_target_lab[0]
            target_lab_strong = inputs_target_lab[1]
        
        if not cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            lab_inputs = [source_strong]
            lab_labels = labels_source
            unl_inputs = [target_unl_weak]
            unl_labels = labels_target_unl
        else:
            lab_inputs = [torch.cat((source_strong, target_lab_strong), dim=0)]
            lab_labels = torch.cat((labels_source, labels_target_lab), dim=0)
            unl_inputs = [target_unl_weak]
            unl_labels = labels_target_unl

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # Step A train all networks to minimize loss on source domain
            optimizer_f.zero_grad()
            optimizer_c.zero_grad()

            lab_preds, lab_feats = model(lab_inputs)

            criterion = nn.CrossEntropyLoss()
            loss_s = criterion(lab_preds, lab_labels)
            loss_s.backward()
            optimizer_f.step()
            optimizer_c.step()

            # Step B train classifier to maximize discrepancy
            optimizer_f.zero_grad()
            optimizer_c.zero_grad()

            unl_preds, unl_feats = model(unl_inputs, reverse=True)
            new_preds = F.softmax(unl_preds, dim=1)
            loss_h = cfg.ADAEMBED.LAMBDA * torch.mean(
                torch.sum(new_preds * (torch.log(new_preds + 1e-5)), 1))
            loss_h.backward()
            optimizer_f.step()
            optimizer_c.step()

            prototypes = model.module.head.weight

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(lab_preds, lab_labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / lab_preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss_s, loss_h, top1_err, top5_err = du.all_reduce(
                [loss_s, loss_h, top1_err, top5_err]
            )

        # Copy the stats from GPU to CPU (sync point).
        loss_s, loss_h, top1_err, top5_err = (
            loss_s.item(),
            loss_h.item(),
            top1_err.item(),
            top5_err.item()
        )
        batch_size = inputs_source[0].size(0)*max(cfg.NUM_GPUS, 1)

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss_s,
            lr,
            batch_size,
        )
        # write to tensorboard format if available.
        if writer is not None:
            dict2write = {
                "Train/loss_s": loss_s,
                "Train/loss_h": loss_h,
                "Train/lr": lr,
                "Train/Top1_err": top1_err,
                "Train/Top5_err": top5_err,
            }
            writer.add_scalars(dict2write, global_step=data_size * cur_epoch + cur_iter)

            if cfg.TENSORBOARD.DIST_VIS.ENABLE and (data_size * cur_epoch + cur_iter)%cfg.TENSORBOARD.DIST_VIS.LOG_PERIOD==1:
                writer.add_confusion_matrix(
                    torch.argmax(torch.cat(train_meter.all_source_weak, dim=0), dim=1), 
                    torch.cat(train_meter.all_source_labels, dim=0), 
                    tag="Confusion/Labeled", 
                    global_step=data_size * cur_epoch + cur_iter
                )
                writer.add_confusion_matrix(
                    torch.argmax(torch.cat(train_meter.all_target_weak, dim=0), dim=1), 
                    torch.cat(train_meter.all_target_labels, dim=0), 
                    tag="Confusion/Unlabeled", 
                    global_step=data_size * cur_epoch + cur_iter
                )
                all_lab_preds = torch.cat(train_meter.all_source_weak, dim=0)
                all_lab_feats = torch.cat(train_meter.all_source_strong, dim=0)
                all_unl_preds = torch.cat(train_meter.all_target_weak, dim=0)
                all_unl_feats = torch.cat(train_meter.all_target_strong, dim=0)
                all_lab_labels = torch.cat(train_meter.all_source_labels, dim=0)
                all_unl_labels = torch.cat(train_meter.all_target_labels, dim=0)

                dict2save = {
                    "all_lab_preds": all_lab_preds.detach().cpu(),
                    "all_lab_feats": all_lab_feats.detach().cpu(),
                    "all_unl_preds": all_unl_preds.detach().cpu(),
                    "all_unl_feats": all_unl_feats.detach().cpu(),
                    "all_lab_labels": all_lab_labels.detach().cpu(),
                    "all_unl_labels": all_unl_labels.detach().cpu(),
                    "prototypes": prototypes.detach().cpu(),
                }
                np.save(cfg.OUTPUT_DIR + f'/step{data_size * cur_epoch + cur_iter}.npy', dict2save)

            if cfg.TENSORBOARD.SAMPLE_VIS.ENABLE and (data_size * cur_epoch + cur_iter)%cfg.TENSORBOARD.SAMPLE_VIS.LOG_PERIOD==0:
                writer.add_video_pred(
                    lab_inputs[0], 
                    torch.argmax(lab_preds, dim=1), 
                    lab_labels,
                    tag="Sample/Source",
                    global_step = data_size * cur_epoch + cur_iter,
                )
                writer.add_video_pred(
                    unl_inputs[0], 
                    torch.argmax(unl_preds, dim=1), 
                    unl_labels,
                    tag="Sample/Target",
                    global_step = data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.update_predictions(
            lab_preds, lab_feats, lab_labels, 
            unl_preds, unl_feats, unl_labels, prototypes,
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()
        del inputs_source
        del inputs_target_unl
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            del inputs_target_lab

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
            all_preds = [pred.clone().detach() for pred in train_meter.all_source_strong]
            all_labels = [label.clone().detach() for label in train_meter.all_source_labels]
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

        preds, feats = model(inputs)

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
    cfg.EXTRACT.ENABLE = True
    cfg.SWIN.TEMP = cfg.MME.TEMP
    cfg.SWIN.ETA = cfg.MME.LAMBDA
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    if hasattr(model, "module"):
        module = model.module
    else:
        module = model
    sub_modules = []
    for name, sub_module in module.named_modules():
        if name!="head":
            sub_modules.append(sub_module)
    backbone = nn.Sequential(*sub_modules)
    classifier = module.get_submodule("head")
    optimizer_f = optim.construct_optimizer(backbone, cfg)
    optimizer_c = optim.construct_optimizer(classifier, cfg)
    optimizers = [optimizer_f, optimizer_c]
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, None,
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
        target_lab_cfg.TRAIN.BATCH_SIZE = source_cfg.TRAIN.BATCH_SIZE
        target_lab_loader = loader.construct_loader(target_lab_cfg, "lab")
        target_unl_cfg = copy.deepcopy(cfg) 
        target_unl_cfg.DATA.IMDB_FILES.TRAIN = cfg.ADAPTATION.TARGET
        target_unl_cfg.DATA.IMDB_FILES.VAL = cfg.ADAPTATION.SOURCE
        target_unl_cfg.TRAIN.BATCH_SIZE = cfg.ADAPTATION.BETA * source_cfg.TRAIN.BATCH_SIZE
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
        target_unl_cfg.TRAIN.BATCH_SIZE = cfg.ADAPTATION.BETA * source_cfg.TRAIN.BATCH_SIZE
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
    train_meter = AdaMeter(len(train_loaders[0]), cfg)
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
            optimizers, 
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
                optimizer_f, 
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
