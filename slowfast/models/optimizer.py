#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch
import logging

import slowfast.utils.logging as logging
import slowfast.utils.lr_policy as lr_policy


logger = logging.get_logger(__name__)


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    bn_parameters = []
    base_parameters = []
    head_parameters = []
    zero_parameters = []
    skip = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    for name, m in model.named_modules():
        if "head" in name:
            for p in m.parameters():
                if not p.requires_grad:
                    continue
                head_parameters.append(p)
        elif name in skip:
            for p in m.parameters(recurse=False):
                if not p.requires_grad:
                    continue
                zero_parameters.append(p)
        elif isinstance(m, torch.nn.modules.batchnorm._NormBase):
            for p in m.parameters(recurse=False):
                if not p.requires_grad:
                    continue
                bn_parameters.append(p)
        else:
            for p in m.parameters(recurse=False):
                if not p.requires_grad:
                    continue
                base_parameters.append(p)

    optim_params_base = [
        # {"params": bn_parameters, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": base_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        # {"params": zero_parameters, "weight_decay": 0.0},
    ]
    optim_params_base = [x for x in optim_params_base if len(x["params"])]
    optim_params_head = [
        {"params": head_parameters, "weight_decay": 10 * cfg.SOLVER.WEIGHT_DECAY},
    ]
    optim_params_head = [x for x in optim_params_head if len(x["params"])]
    optim_params = optim_params_base + optim_params_head

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(head_parameters) + len(base_parameters) + len(
        bn_parameters
    ) + len(zero_parameters
    ), "parameter size does not match: {} + {} + {} != {}".format(
        len(non_bn_parameters),
        len(bn_parameters),
        len(zero_parameters),
        len(list(model.parameters())),
    )

    logger.info("Break down the paramters into {} groups: {} for bn, {} for base, {} for head, and {} for zero-wd".format(
            len(optim_params),
            len(bn_parameters),
            len(base_parameters),
            len(head_parameters),
            len(zero_parameters),
    ))

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        if cfg.ADAPTATION.ENABLE:
            optimizer_f = torch.optim.SGD(
            optim_params_base,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
            )
            optimizer_c = torch.optim.SGD(
            optim_params_head,
            lr=10*cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
            )
            return optimizer_f, optimizer_c
        else:
            return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        if cfg.ADAPTATION.ENABLE:
            optimizer_f = torch.optim.Adam(
            optim_params_base,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
            optimizer_c = torch.optim.Adam(
            optim_params_head,
            lr=10*cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
            return optimizer_f, optimizer_c
        else:
            return torch.optim.Adam(
                optim_params,
                lr=cfg.SOLVER.BASE_LR,
                betas=(0.9, 0.999),
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        if cfg.ADAPTATION.ENABLE:
            optimizer_f = torch.optim.AdamW(
            optim_params_base,
            lr=cfg.SOLVER.BASE_LR,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
            optimizer_c = torch.optim.AdamW(
            optim_params_head,
            lr=10*cfg.SOLVER.BASE_LR,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
            return optimizer_f, optimizer_c
        else:
            return torch.optim.AdamW(
                optim_params,
                lr=cfg.SOLVER.BASE_LR,
                eps=1e-08,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
