#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import math
import copy
from collections import OrderedDict
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
from slowfast.utils.meters import AdaptationMeter, ValMeter, EpochTimer

logger = logging.get_logger(__name__)


def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(X, Y.T)
    return distances


def train_epoch(
    train_loaders, 
    models, 
    optimizers, 
    scaler, 
    train_meter, 
    cur_epoch, 
    cfg, 
    writer=None
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

    model, model_momentum = models[0], models[1]
    optimizer_f, optimizer_c = optimizers[0], optimizers[1]
    
    # Enable train mode.
    model.train()
    model_momentum.eval()

    train_meter.iter_tic()
    data_size = len(source_loader)
    target_unl_iter = iter(target_unl_loader)
    if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
        target_lab_iter = iter(target_lab_loader)

    # Loss functions needed.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    loss_fun_none = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="none")

    for cur_iter, (inputs_source, labels_source, _, _) in enumerate(source_loader):
        try:
            inputs_target_unl, labels_target_unl, _, _ = next(target_unl_iter)
        except StopIteration:
            target_unl_iter = iter(target_unl_loader)
            inputs_target_unl, labels_target_unl, _, _ = next(target_unl_iter)
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            try:
                inputs_target_lab, labels_target_lab, _, _ = next(target_lab_iter)
            except StopIteration:
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
        optim.set_lr(optimizer_f, lr)
        optim.set_lr(optimizer_c, lr)
        mu = (0.5 + math.cos(math.pi * (cfg.SOLVER.MAX_EPOCH - cur_epoch - float(cur_iter) / data_size) / cfg.SOLVER.MAX_EPOCH) / 2)

        # Load bank.
        if len(train_meter.all_lab_preds) > 10:
            lab_feats_bank = torch.cat(train_meter.all_lab_feats, dim=0).detach()
            unl_feats_bank = torch.cat(train_meter.all_unl_feats, dim=0).detach()
            lab_probs_bank = torch.cat(train_meter.all_lab_preds, dim=0).detach()
            unl_probs_bank = torch.cat(train_meter.all_unl_preds, dim=0).detach()
            feats_bank = torch.cat((lab_feats_bank, unl_feats_bank), dim=0)
            probs_bank = torch.cat((lab_probs_bank, unl_probs_bank), dim=0)
        else:
            if cfg.NUM_GPUS > 1:
                num_features = model.module.num_features
            else:
                num_features = model.num_features
            lab_feats_bank = torch.zeros(10, num_features).cuda()
            unl_feats_bank = torch.zeros(10, num_features).cuda()
            lab_probs_bank = torch.zeros(10, cfg.MODEL.NUM_CLASSES).cuda()
            unl_probs_bank = torch.zeros(10, cfg.MODEL.NUM_CLASSES).cuda()
            feats_bank = torch.cat((lab_feats_bank, unl_feats_bank), dim=0)
            probs_bank = torch.cat((lab_probs_bank, unl_probs_bank), dim=0)
        
        # Generate alignment for predictions.
        if cfg.ADAEMBED.ALIGNMENT and len(train_meter.all_lab_preds) > 10:
            p_source = F.softmax(lab_probs_bank, dim=1).mean(dim=0)
            p_target = F.softmax(unl_probs_bank, dim=1).mean(dim=0)
            alignment = (1e-6 + p_source) / (1e-6 + p_target)
        else:
            alignment = torch.ones(cfg.MODEL.NUM_CLASSES).cuda()

        # Generate the sampling rate for pseudo labels.
        if cfg.ADAEMBED.SAMPLING and len(train_meter.all_pseudo_labels) > 10:
            all_pseudo_labels = train_meter.all_pseudo_labels.copy()
            all_pseudo_labels.extend([torch.tensor(i).cuda() for i in range(cfg.MODEL.NUM_CLASSES)])
            all_pseudo_labels = torch.stack(all_pseudo_labels, dim=0).detach()
            freq = torch.bincount(all_pseudo_labels)
            p_sample = torch.minimum((torch.min(freq) + 1) / (freq + 1), torch.ones_like(freq))
        else:
            p_sample = torch.ones(cfg.MODEL.NUM_CLASSES).cuda()

        # Generate tau.
        if cfg.ADAEMBED.THRESHOLDING and len(train_meter.all_lab_preds) > 10:
            if cfg.ADAEMBED.THRESHOLDING=='uniform':
                c_tau = cfg.ADAEMBED.TAU * torch.ones(cfg.MODEL.NUM_CLASSES).cuda()
            elif cfg.ADAEMBED.THRESHOLDING=='relative':
                max_preds = F.softmax(lab_probs_bank, dim=1).max(dim=1)[0]
                c_tau = cfg.ADAEMBED.TAU * max_preds.mean(dim=0) * torch.ones(cfg.MODEL.NUM_CLASSES).cuda()
                c_tau = torch.minimum(c_tau, 0.99 * torch.ones_like(c_tau))
            elif cfg.ADAEMBED.THRESHOLDING=='adaptive':
                all_preds = F.softmax(lab_probs_bank, dim=1)
                max_preds = all_preds.max(dim=1)[0]
                class_preds = torch.argmax(all_preds, dim=1)
                c_tau = torch.stack([cfg.ADAEMBED.TAU * max_preds[class_preds==i].mean(dim=0) for i in range(cfg.MODEL.NUM_CLASSES)])
                c_tau = torch.minimum(c_tau, 0.99 * torch.ones_like(c_tau))
                c_tau = torch.maximum(c_tau, 0.8 * torch.ones_like(c_tau))
            elif cfg.ADAEMBED.THRESHOLDING=='elastic':
                all_preds = F.softmax(lab_probs_bank, dim=1)
                max_preds = all_preds.max(dim=1)[0]
                class_preds = torch.argmax(all_preds, dim=1)
                c_tau = torch.stack([max_preds[class_preds==i].mean(dim=0) for i in range(cfg.MODEL.NUM_CLASSES)])
                c_tau = cfg.ADAEMBED.TAU * c_tau / torch.max(c_tau)
        else:
            c_tau = torch.ones(cfg.MODEL.NUM_CLASSES).cuda()

        train_meter.data_toc()
        source_weak = inputs_source[1]
        source_strong = inputs_source[0]
        target_unl_weak = inputs_target_unl[1]
        target_unl_strong = inputs_target_unl[0]
        if cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            target_lab_weak = inputs_target_lab[1]
            target_lab_strong = inputs_target_lab[0]
        
        if not cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
            inputs_weak = torch.cat((source_weak, target_unl_weak), dim=0)
            inputs_strong = torch.cat((source_strong, target_unl_strong), dim=0)
        else:
            inputs_weak = torch.cat((source_weak, target_lab_weak, target_unl_weak), dim=0)
            inputs_strong = torch.cat((source_strong, target_lab_strong, target_unl_strong), dim=0)

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # Forward pass.
            optimizer_f.zero_grad()
            optimizer_c.zero_grad()

            # Compute the predictions.
            logists_strong, feats_strong = model([inputs_strong])
            with torch.no_grad():
                logists_weak, feats_weak = model_momentum([inputs_weak])
            
            if not cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE:
                feats_sls, feats_tus = torch.split(feats_strong, [len(source_strong),len(target_unl_strong)])
                logits_sls, logits_tus = torch.split(logists_strong, [len(source_strong),len(target_unl_strong)])
                feats_slw, feats_tuw = torch.split(feats_weak, [len(source_weak),len(target_unl_weak)])
                logits_slw, logits_tuw = torch.split(logists_weak, [len(source_weak),len(target_unl_weak)])
                lab_feats = feats_slw
                lab_preds = logits_slw
                lab_labels = labels_source
                unl_feats = feats_tuw
                unl_preds = logits_tuw
                unl_labels = labels_target_unl
                lab_feats_strong = feats_sls
                lab_preds_strong = logits_sls
            else:
                feats_sls, feats_tls, feats_tus = torch.split(feats_strong, [len(source_strong),len(target_lab_strong),len(target_unl_strong)])
                logits_sls, logits_tls, logits_tus = torch.split(logists_strong, [len(source_strong),len(target_lab_strong),len(target_unl_strong)])
                feats_slw, feats_tlw, feats_tuw = torch.split(feats_weak, [len(source_weak),len(target_lab_weak),len(target_unl_weak)])
                logits_slw, logits_tlw, logits_tuw = torch.split(logists_weak, [len(source_weak),len(target_lab_weak),len(target_unl_weak)])
                lab_feats = torch.cat((feats_slw, feats_tlw), dim=0)
                lab_preds = torch.cat((logits_slw, logits_tlw), dim=0)
                lab_labels = torch.cat((labels_source, labels_target_lab), dim=0)
                unl_feats = feats_tuw
                unl_preds = logits_tuw
                unl_labels = labels_target_unl
                lab_feats_strong = torch.cat((feats_sls, feats_tls), dim=0)
                lab_preds_strong = torch.cat((logits_sls, logits_tls), dim=0)

            if cfg.NUM_GPUS > 1:
                prototypes = model.module.head.weight.clone().detach()
            else:
                prototypes = model.head.weight.clone().detach()

            # Compute the loss.
            loss_s = loss_fun(lab_preds_strong, lab_labels)

            # Pseudo label generation
            preds_tuw = F.softmax(logits_tuw, dim=1)
            if cfg.ADAEMBED.PSEUDO_TYPE=='AdaMatch':
                preds_tuw_refined = F.normalize(alignment * preds_tuw, p=1, dim=1)
                pseudo_labels = torch.argmax(preds_tuw_refined, dim=1)
                pred_mask = preds_tuw_refined.max(dim=1)[0] >= c_tau[pseudo_labels]
                sampling_mask = torch.bernoulli(p_sample[pseudo_labels]) > 0
            elif cfg.ADAEMBED.PSEUDO_TYPE=='AdaEmbed':
                mem_feat = F.normalize(feats_bank, dim=1)
                mem_probs = F.softmax(probs_bank, dim=1)
                k_feats = F.normalize(feats_tuw, dim=1)
                distances = get_distances(k_feats, mem_feat)
                refined_probs = []
                for i in range(len(k_feats)):
                    dists = distances[i, :]
                    _, idxs = dists.sort()
                    idxs = idxs[:cfg.ADAEMBED.NUM_NEIGHBORS]
                    probs = mem_probs[idxs, :].mean(0)
                    refined_probs.append(probs)
                refined_probs = torch.stack(refined_probs)
                preds_tuw_refined = F.normalize(alignment * refined_probs, p=1, dim=1)
                pseudo_labels = torch.argmax(preds_tuw_refined, dim=1)
                pred_mask = preds_tuw_refined.max(dim=1)[0] >= c_tau[pseudo_labels]
                sampling_mask = torch.zeros_like(pseudo_labels).bool()
                proto_distances = get_distances(prototypes, k_feats)
                max_dist = []
                for i in range(len(prototypes)):
                    dists = proto_distances[i, :]
                    _, idxs = dists.sort()
                    idxs = idxs[:cfg.ADAEMBED.NUM_NEIGHBORS]
                    max_dist.append(dists[idxs].max())
                for i in range(len(k_feats)):
                    dists = proto_distances[:, i]
                    _, idxs = dists.sort()
                    idx = idxs[0]
                    if dists[idx] < max_dist[idx]:
                        sampling_mask[i] = True
            else:
                preds_tuw_refined = preds_tuw
                pseudo_labels = torch.argmax(preds_tuw_refined, dim=1)
                pred_mask = preds_tuw_refined.max(dim=1)[0] >= c_tau[pseudo_labels]
                sampling_mask = torch.bernoulli(p_sample[pseudo_labels]) > 0
            mask = pred_mask * sampling_mask
            loss_t = loss_fun_none(logits_tus, pseudo_labels)
            loss_t = (mask * loss_t).mean(dim=0)

            # Prototype loss
            if cfg.ADAEMBED.LAMBDA_P > 0:
                q = F.normalize(lab_feats_strong, dim=1)
                k = prototypes[lab_labels]
                l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
                l_neg = torch.einsum("nc,ck->nk", [q, prototypes.T])
                logits_ins = torch.cat([l_pos, l_neg], dim=1) / cfg.ADAEMBED.TEMP
                labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()
                mask_ins = torch.ones_like(logits_ins, dtype=torch.bool)
                mask_ins[:, 1:] = lab_labels.reshape(-1, 1) != torch.arange(cfg.MODEL.NUM_CLASSES).cuda()  # (B, K)
                logits_ins = torch.where(mask_ins, logits_ins, torch.tensor([float("-inf")]).cuda())
                loss_p = F.cross_entropy(logits_ins, labels_ins)
            else:
                loss_p = torch.zeros_like(loss_s)

            # Contrastive loss
            if cfg.ADAEMBED.LAMBDA_C > 0:
                q = F.normalize(feats_tus, dim=1)
                k = F.normalize(feats_tuw, dim=1)
                mem_feat = F.normalize(unl_feats_bank, dim=1)
                mem_label = torch.argmax(unl_probs_bank, dim=1)
                l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
                l_neg = torch.einsum("nc,ck->nk", [q, mem_feat.T])
                logits_ins = torch.cat([l_pos, l_neg], dim=1) / cfg.ADAEMBED.TEMP
                labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()
                mask_ins = torch.ones_like(logits_ins, dtype=torch.bool)
                mask_ins[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_label  # (B, K)
                logits_ins = torch.where(mask_ins, logits_ins, torch.tensor([float("-inf")]).cuda())
                loss_c = F.cross_entropy(logits_ins, labels_ins)
            else:
                loss_c = torch.zeros_like(loss_s)

            loss = loss_s + cfg.ADAEMBED.LAMBDA_T * mu * loss_t + \
                + cfg.ADAEMBED.LAMBDA_P * mu * loss_p + cfg.ADAEMBED.LAMBDA_C * mu * loss_c
            loss.backward()
            optimizer_f.step()
            optimizer_c.step()

            # Train classifier to maximize discrepancy.
            optimizer_f.zero_grad()
            optimizer_c.zero_grad()

            unl_preds_rev, _ = model([target_unl_weak], reverse=True)
            new_preds = F.softmax(unl_preds_rev, dim=1)
            loss_h = torch.mean(torch.sum(new_preds * (torch.log(new_preds + 1e-5)), 1))
            loss2 = cfg.ADAEMBED.LAMBDA_H * mu * loss_h
            loss2.backward()
            optimizer_f.step()
            optimizer_c.step()

        # Update the momentum encoder.
        for ema_param, param in zip(model_momentum.parameters(), model.parameters()):
            ema_param.data.mul_(cfg.ADAEMBED.EMA).add_(param.data, alpha=1-cfg.ADAEMBED.EMA)
        # model_params = OrderedDict(model.named_parameters())
        # momentum_params = OrderedDict(model_momentum.named_parameters())
        # model_buffers = OrderedDict(model.named_buffers())
        # momentum_buffers = OrderedDict(model_momentum.named_buffers())
        # assert model_params.keys() == momentum_params.keys()
        # assert model_buffers.keys() == momentum_buffers.keys()
        # for name, param in model_params.items():
        #     momentum_params[name].sub_((1.0 - cfg.ADAEMBED.EMA) * (momentum_params[name] - param))
        # for name, buffer in model_buffers.items():
        #     momentum_buffers[name].copy_(buffer)

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
            loss, loss_s, loss_t, loss_p, loss_c, loss_h, top1_err, top5_err = du.all_reduce(
                [loss, loss_s, loss_t, loss_p, loss_c, loss_h, top1_err, top5_err]
            )
            num_pred_mask, num_sampling_mask, num_pseudo, num_correct = du.all_reduce(
                [num_pred_mask, num_sampling_mask, num_pseudo, num_correct], 
                average=False,
            )

        # Copy the stats from GPU to CPU (sync point).
        loss, loss_s, loss_t, loss_p, loss_c, loss_h, top1_err, top5_err = (
            loss.item(),
            loss_s.item(),
            loss_t.item(),
            loss_p.item(), 
            loss_c.item(),
            loss_h.item(),
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
                "Train/loss_p": loss_p,
                "Train/loss_c": loss_c,
                "Train/loss_h": -loss_h,
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

                # if cfg.ADAPTATION.SOURCE==['real_train'] and cfg.ADAPTATION.TARGET==['clipart_train'] and (not cfg.ADAPTATION.SEMI_SUPERVISED.ENABLE):
                #     all_lab_preds = torch.cat(train_meter.all_lab_preds, dim=0)
                #     all_lab_feats = torch.cat(train_meter.all_lab_feats, dim=0)
                #     all_unl_preds = torch.cat(train_meter.all_unl_preds, dim=0)
                #     all_unl_feats = torch.cat(train_meter.all_unl_feats, dim=0)
                #     all_lab_labels = torch.cat(train_meter.all_lab_labels, dim=0)
                #     all_unl_labels = torch.cat(train_meter.all_unl_labels, dim=0)
                #     dict2save = {
                #         "all_lab_preds": all_lab_preds.detach().cpu(),
                #         "all_lab_feats": all_lab_feats.detach().cpu(),
                #         "all_unl_preds": all_unl_preds.detach().cpu(),
                #         "all_unl_feats": all_unl_feats.detach().cpu(),
                #         "all_lab_labels": all_lab_labels.detach().cpu(),
                #         "all_unl_labels": all_unl_labels.detach().cpu(),
                #         "prototypes": prototypes.detach().cpu(),
                #     }
                #     np.save(cfg.OUTPUT_DIR + f'/step{data_size * cur_epoch + cur_iter}.npy', dict2save)

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
            lab_feats, lab_preds, lab_labels, 
            unl_feats, unl_preds, unl_labels, preds_tuw_refined,
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

        preds, _ = model(inputs)

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
    cfg.SWIN.TEMP = cfg.ADAEMBED.TEMP
    cfg.SWIN.ETA = 1.0
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)
    model_momentum = copy.deepcopy(model)
    for param in model_momentum.parameters():
        param.requires_grad = False
    models = [model, model_momentum]

    # Construct the optimizer.
    optimizer_f, optimizer_c = optim.construct_optimizer(model, cfg)
    optimizers = [optimizer_f, optimizer_c]
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer_f,
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
            writer
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
                model_momentum, 
                val_meter, 
                cur_epoch, 
                cfg, 
                writer,
            )

    if writer is not None:
        writer.close()
    raise SystemExit('Training Ends')
