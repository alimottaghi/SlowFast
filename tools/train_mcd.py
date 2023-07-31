#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import copy
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
from slowfast.utils.meters import AdaptationMeter, ValMeter, EpochTimer

logger = logging.get_logger(__name__)


def classifier_discrepancy(predictions1: torch.Tensor, predictions2: torch.Tensor) -> torch.Tensor:
    r"""The `Classifier Discrepancy` in
    `Maximum Classiﬁer Discrepancy for Unsupervised Domain Adaptation (CVPR 2018) <https://arxiv.org/abs/1712.02560>`_.

    The classfier discrepancy between predictions :math:`p_1` and :math:`p_2` can be described as:

    .. math::
        d(p_1, p_2) = \dfrac{1}{K} \sum_{k=1}^K | p_{1k} - p_{2k} |,

    where K is number of classes.

    Args:
        predictions1 (torch.Tensor): Classifier predictions :math:`p_1`. Expected to contain raw, normalized scores for each class
        predictions2 (torch.Tensor): Classifier predictions :math:`p_2`
    """
    return torch.mean(torch.abs(predictions1 - predictions2))


def entropy(predictions: torch.Tensor) -> torch.Tensor:
    r"""Entropy of N predictions :math:`(p_1, p_2, ..., p_N)`.
    The definition is:

    .. math::
        d(p_1, p_2, ..., p_N) = -\dfrac{1}{K} \sum_{k=1}^K \log \left( \dfrac{1}{N} \sum_{i=1}^N p_{ik} \right)

    where K is number of classes.

    .. note::
        This entropy function is specifically used in MCD and different from the usual :meth:`~dalib.modules.entropy.entropy` function.

    Args:
        predictions (torch.Tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
    """
    return -torch.mean(torch.log(torch.mean(predictions, 0) + 1e-6))


class ImageClassifierHead(nn.Module):
    r"""Classifier Head for MCD.

    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024

    Shape:
        - Inputs: :math:`(minibatch, F)` where F = `in_features`.
        - Output: :math:`(minibatch, C)` where C = `num_classes`.
    """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim=1024, pool_layer=None):
        super(ImageClassifierHead, self).__init__()
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool_layer(inputs))


def train_epoch(
    train_loaders, 
    models, 
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
    
    # Enable train mode.
    backbone, classifier1, classifier2 = models[0], models[1], models[2]
    backbone.train()
    classifier1.train()
    classifier2.train()

    optimizer_f, optimizer_c = optimizers[0], optimizers[1]

    train_meter.iter_tic()
    data_size = len(source_loader)
    target_unl_iter = iter(target_unl_loader)
    if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
        target_lab_iter = iter(target_lab_loader)

    for cur_iter, (source_inputs, source_labels, _, _) in enumerate(source_loader):
        try:
            target_unl_inputs, target_unl_labels, _, _ = next(target_unl_iter)
        except StopIteration:
            target_unl_iter = iter(target_unl_loader)
            target_unl_inputs, target_unl_labels, _, _ = next(target_unl_iter)
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            try:
                target_lab_inputs, target_lab_labels, _, _ = next(target_lab_iter)
            except StopIteration:
                target_lab_iter = iter(target_lab_loader)
                target_lab_inputs, target_lab_labels, _, _ = next(target_lab_iter)
        
        # Transfer the data to the current GPU device.
        for i in range(len(source_inputs)):
            source_inputs[i] = source_inputs[i].cuda(non_blocking=True)
            target_unl_inputs[i] = target_unl_inputs[i].cuda(non_blocking=True)
        source_labels = source_labels.cuda()
        target_unl_labels = target_unl_labels.cuda()
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            for i in range(len(source_inputs)):
                target_lab_inputs[i] = target_lab_inputs[i].cuda(non_blocking=True)
            target_lab_labels = target_lab_labels.cuda()

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer_f, lr)
        optim.set_lr(optimizer_c, lr)

        train_meter.data_toc()
        source_weak = source_inputs[0]
        source_strong = source_inputs[1]
        target_unl_weak = target_unl_inputs[0]
        target_unl_strong = target_unl_inputs[1]
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            target_lab_weak = target_lab_inputs[0]
            target_lab_strong = target_lab_inputs[1]
        if not cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            weak_input = torch.cat((source_weak, target_unl_weak), dim=0)
            strong_input = torch.cat((source_strong, target_unl_strong), dim=0)
        else:
            weak_input = torch.cat((source_weak, target_unl_weak, target_lab_weak), dim=0)
            strong_input = torch.cat((source_strong, target_unl_strong, target_lab_strong), dim=0)

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):

            x = [weak_input]
            labels_s = source_labels
            trade_off_entropy = 0.01

            # Step A train all networks to minimize loss on source domain
            optimizer_f.zero_grad()
            optimizer_c.zero_grad()

            p, g = backbone(x)
            y_1 = classifier1(g)
            y_2 = classifier2(g)
            if not cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
                y1_s, y1_t = torch.split(y_1, [len(source_weak),len(target_unl_weak)])
                y2_s, y2_t = torch.split(y_2, [len(source_weak),len(target_unl_weak)])
            else:
                y1_s, y1_t, y1_t_lab = torch.split(y_1, [len(source_weak),len(target_unl_weak),len(target_lab_weak)])
                y2_s, y2_t, y2_t_lab = torch.split(y_2, [len(source_weak),len(target_unl_weak),len(target_lab_weak)])

            y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
            loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + \
                (entropy(y1_t) + entropy(y2_t)) * trade_off_entropy
            if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
                loss_target_lab = F.cross_entropy(y1_t_lab, target_lab_labels) + F.cross_entropy(y2_t_lab, target_lab_labels)
                loss += loss_target_lab
            loss.backward()
            optimizer_f.step()
            optimizer_c.step()

            # Step B train classifier to maximize discrepancy
            optimizer_f.zero_grad()
            optimizer_c.zero_grad()

            _, feats = backbone(x)
            y_1 = classifier1(feats)
            y_2 = classifier2(feats)
            if not cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
                y1_s, y1_t = torch.split(y_1, [len(source_weak),len(target_unl_weak)])
                y2_s, y2_t = torch.split(y_2, [len(source_weak),len(target_unl_weak)])
            else:
                y1_s, y1_t, y1_t_lab = torch.split(y_1, [len(source_weak),len(target_unl_weak),len(target_lab_weak)])
                y2_s, y2_t, y2_t_lab = torch.split(y_2, [len(source_weak),len(target_unl_weak),len(target_lab_weak)])
            y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
            loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + \
                (entropy(y1_t) + entropy(y2_t)) * trade_off_entropy - \
                classifier_discrepancy(y1_t, y2_t) * 1
            if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
                loss_target_lab = F.cross_entropy(y1_t_lab, target_lab_labels) + F.cross_entropy(y2_t_lab, target_lab_labels)
                loss += loss_target_lab
            loss.backward()
            optimizer_c.step()

            # Step C train genrator to minimize discrepancy
            for k in range(4):
                optimizer_f.zero_grad()
                _, feats = backbone(x)
                y_1 = classifier1(feats)
                y_2 = classifier2(feats)
                if not cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
                    y1_s, y1_t = torch.split(y_1, [len(source_weak),len(target_unl_weak)])
                    y2_s, y2_t = torch.split(y_2, [len(source_weak),len(target_unl_weak)])
                else:
                    y1_s, y1_t, y1_t_lab = torch.split(y_1, [len(source_weak),len(target_unl_weak),len(target_lab_weak)])
                    y2_s, y2_t, y2_t_lab = torch.split(y_2, [len(source_weak),len(target_unl_weak),len(target_lab_weak)])
                y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
                mcd_loss = classifier_discrepancy(y1_t, y2_t) * 1
                mcd_loss.backward()
                optimizer_f.step()

        # Compute the errors.
        top1_err, top5_err = None, None
        preds = y1_s
        num_topks_correct = metrics.topks_correct(preds, source_labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, mcd_loss, top1_err, top5_err = du.all_reduce(
                [loss, mcd_loss, top1_err, top5_err]
            )

        # Copy the stats from GPU to CPU (sync point).
        loss, mcd_loss, top1_err, top5_err = (
            loss.item(),
            mcd_loss.item(),
            top1_err.item(),
            top5_err.item()
        )
        batch_size = source_inputs[0].size(0)*max(cfg.NUM_GPUS, 1)

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            batch_size,
        )
        # write to tensorboard format if available.
        if writer is not None:
            dict2write = {
                "Train/loss": loss,
                "Train/mcd_loss": mcd_loss,
                "Train/lr": lr,
                "Train/Top1_err": top1_err,
                "Train/Top5_err": top5_err,
            }
            writer.add_scalars(dict2write, global_step=data_size * cur_epoch + cur_iter)

            # if cfg.TENSORBOARD.DIST_VIS.ENABLE and (data_size * cur_epoch + cur_iter)%cfg.TENSORBOARD.DIST_VIS.LOG_PERIOD==1:
            #     writer.add_confusion_matrix(
            #         torch.argmax(torch.cat(train_meter.all_source_strong, dim=0), dim=1), 
            #         torch.cat(train_meter.all_source_labels, dim=0), 
            #         tag="Confusion Matrix/Train Source", 
            #         global_step=data_size * cur_epoch + cur_iter
            #     )
            #     writer.add_confusion_matrix(
            #         torch.argmax(torch.cat(train_meter.all_target_weak, dim=0), dim=1), 
            #         torch.cat(train_meter.all_target_labels, dim=0), 
            #         tag="Confusion Matrix/Train Target", 
            #         global_step=data_size * cur_epoch + cur_iter
            #     )

            if cfg.TENSORBOARD.SAMPLE_VIS.ENABLE and (data_size * cur_epoch + cur_iter)%cfg.TENSORBOARD.SAMPLE_VIS.LOG_PERIOD==0:
                writer.add_video_pred(
                    source_strong, 
                    torch.argmax(preds, dim=1), 
                    source_labels,
                    tag="Sample/Source",
                    global_step = data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.update_predictions(
            y_1, y_2, source_labels, 
            y_1, y_2, target_unl_labels, y_2,
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()
        del source_inputs
        del target_unl_inputs
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            del target_lab_inputs

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
        # all_preds = [pred.clone().detach() for pred in train_meter.all_source_strong]
        # all_labels = [label.clone().detach() for label in train_meter.all_source_labels]
        # all_preds = [pred.cpu() for pred in all_preds]
        # all_labels = [label.cpu() for label in all_labels]
        # writer.plot_eval(
        #     preds=all_preds, 
        #     labels=all_labels, 
        #     global_step=cur_epoch, 
        #     tag="Confusion/Train"
        # )
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
    val_loader, models, val_meter, cur_epoch, cfg, writer=None
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
    backbone, classifier1, classifier2 = models[0], models[1], models[2]
    backbone.eval()
    classifier1.eval()
    classifier2.eval()
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

        _, feats = backbone(inputs)
        preds1, preds2 = classifier1(feats), classifier2(feats)

        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                preds, labels = du.all_gather([preds, labels])
        else:
            # Compute the errors.
            num_topks_correct1 = metrics.topks_correct(preds1, labels, (1, 2))
            num_topks_correct2 = metrics.topks_correct(preds2, labels, (1, 2))

            # Combine the errors across the GPUs.
            top1_err1, top5_err1 = [
                (1.0 - x / preds1.size(0)) * 100.0 for x in num_topks_correct1
            ]
            top1_err2, top5_err2 = [
                (1.0 - x / preds2.size(0)) * 100.0 for x in num_topks_correct2
            ]
            if cfg.NUM_GPUS > 1:
                top1_err1, top5_err1 = du.all_reduce([top1_err1, top5_err1])
                top1_err2, top5_err2 = du.all_reduce([top1_err2, top5_err2])

            # Copy the errors from GPU to CPU (sync point).
            top1_err1, top5_err1 = top1_err1.item(), top5_err1.item()
            top1_err2, top5_err2 = top1_err2.item(), top5_err2.item()

            if top1_err1 > top1_err2:
                top1_err = top1_err1
                top5_err = top5_err1
                preds = preds1
            else:
                top1_err = top1_err2
                top5_err = top5_err2
                preds = preds2

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
        # all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
        # all_labels = [
        #     label.clone().detach() for label in val_meter.all_labels
        # ]
        # if cfg.NUM_GPUS:
        #     all_preds = [pred.cpu() for pred in all_preds]
        #     all_labels = [label.cpu() for label in all_labels]
        # writer.plot_eval(
        #     preds=all_preds, 
        #     labels=all_labels, 
        #     global_step=cur_epoch, 
        #     tag="Confusion/Val"
        # )

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
    num_class = cfg.MODEL.NUM_CLASSES
    cfg.MODEL.NUM_CLASSES = 0
    backbone = build_model(cfg)
    cfg.MODEL.NUM_CLASSES = num_class
    classifier1 = ImageClassifierHead(in_features=backbone.module.num_features,  num_classes=num_class, pool_layer=nn.Identity())
    classifier1 = classifier1.cuda(device=torch.cuda.current_device())
    classifier2 = ImageClassifierHead(in_features=backbone.module.num_features,  num_classes=num_class, pool_layer=nn.Identity())
    classifier2 = classifier2.cuda(device=torch.cuda.current_device())
    models = [backbone, classifier1, classifier2]
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(backbone, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer_f = optim.construct_optimizer(backbone, cfg)
    optimizer_c = torch.optim.SGD(
            [
                {"params": classifier1.parameters()},
                {"params": classifier2.parameters()},
            ],
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
    )
    optimizers = [optimizer_f, optimizer_c]
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, backbone, None,
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
    train_meter = AdaptationMeter(len(train_loaders[0]), cfg)
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
            models, 
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
            and len(get_bn_modules(backbone)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                backbone,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(backbone)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR, 
                backbone, 
                optimizer_f, 
                cur_epoch, 
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(
                val_loader, 
                models, 
                val_meter, 
                cur_epoch, 
                cfg, 
                writer,
            )

    if writer is not None:
        writer.close()
    raise SystemExit('Training Ends')
