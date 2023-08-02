#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.config import CfgNode

"""Add custom configs and default values"""

def add_custom_config(_C):
    # Add your own customized configs.

    # Data Options
    _C.DATA.IMDB_FILES = CfgNode()
    _C.DATA.IMDB_FILES.TRAIN = ["train"]
    _C.DATA.IMDB_FILES.VAL = ["val"]
    _C.DATA.IMDB_FILES.TEST = ["test"]
    _C.DATA.SAMPLE_RATIO = 1.0
    _C.SOLVER.EPOCH_ITER_MAX = 1000

    # Augmentation Options
    _C.AUG.GRAYSCALE = 0.2
    _C.AUG.GAUSSIAN_BLUR = 0.5

    # use a different num_frames during test time. If -1, use DATA.NUM_FRAMES
    _C.EXTRACT = CfgNode()
    _C.EXTRACT.NUM_FRAMES = 16
    _C.EXTRACT.ENABLE = False
    _C.EXTRACT.PATH_TO_FEAT_OUT_DIR = ""
    _C.EXTRACT.EXTRACT_RELEVANT_FRAMES_ONLY = False
    _C.EXTRACT.SAMPLING_RATE = 1

    # Swin Options
    _C.SWIN = CfgNode()
    _C.SWIN.PATCH_SIZE = (4, 4, 4)
    _C.SWIN.IN_CHANS = 3
    _C.SWIN.EMBED_DIM = 192
    _C.SWIN.DEPTHS = [2, 2, 18, 2]
    _C.SWIN.NUM_HEADS = [6, 12, 24, 48]
    _C.SWIN.WINDOW_SIZE = (2, 7, 7)
    _C.SWIN.MLP_RATIO = 4.0
    _C.SWIN.QKV_BIAS = True
    _C.SWIN.QK_SCALE = None
    _C.SWIN.DROP_RATE = 0.0
    _C.SWIN.ATTN_DROP_RATE = 0.0
    _C.SWIN.DROP_PATH_RATE = 0.2
    _C.SWIN.NORM_LAYER = "layernorm"
    _C.SWIN.PATCH_NORM = False
    _C.SWIN.FROZEN_STAGES = -1
    _C.SWIN.SPATIAL_TYPE = "avg"
    _C.SWIN.DROP_HEAD = 0.5
    _C.SWIN.USE_CHECKPOINT = False
    _C.SWIN.APE = False
    _C.SWIN.FUSED_WINDOW_PROCESS = False
    _C.SWIN.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
    _C.SWIN.PRETRAINED_FILEPATH = ""
    _C.SWIN.PRETRAINED_2D = False
    _C.SWIN.FEW_SHOT = False
    _C.SWIN.TEMP = 0.05
    _C.SWIN.ETA = 1.0

    # TimeSformer Options
    _C.TIMESFORMER = CfgNode()
    _C.TIMESFORMER.ATTENTION_TYPE = 'divided_space_time'
    _C.TIMESFORMER.PRETRAINED_MODEL = ''
    _C.TRAIN.FINETUNE = False
    _C.MODEL.SINGLE_PATHWAY_ARCH = ["2d", "c2d", "i3d", "slow", "x3d", "mvit", "vit"]

    # Tensorboard Options
    _C.TENSORBOARD.SYNC_WANDB = False
    _C.TENSORBOARD.SAMPLE_VIS = CfgNode()
    _C.TENSORBOARD.SAMPLE_VIS.ENABLE = True
    _C.TENSORBOARD.SAMPLE_VIS.LOG_PERIOD = 1000
    _C.TENSORBOARD.EPOCH_LOG = CfgNode()
    _C.TENSORBOARD.EPOCH_LOG.ENABLE = True
    _C.TENSORBOARD.DIST_VIS = CfgNode()
    _C.TENSORBOARD.DIST_VIS.ENABLE = True
    _C.TENSORBOARD.DIST_VIS.LOG_PERIOD = 1000

    # Adaptation Options
    _C.ADAPTATION = CfgNode()
    _C.ADAPTATION.ENABLE = False
    _C.ADAPTATION.ADAPTATION_TYPE = "finetune"
    _C.ADAPTATION.SOURCE = []
    _C.ADAPTATION.TARGET = []
    _C.ADAPTATION.ALPHA = 1.0
    _C.ADAPTATION.BETA = 1.0
    _C.ADAPTATION.BANK_SIZE = 1000
    _C.ADAPTATION.SEMI_SUPERVISED = CfgNode()
    _C.ADAPTATION.SEMI_SUPERVISED.ENABLE = False
    _C.ADAPTATION.SEMI_SUPERVISED.LAB_RATIO = 0.1
    _C.ADAPTATION.SEMI_SUPERVISED.NUM_SHOTS = 0

    # AdaMatch Options
    _C.ADAMATCH = CfgNode()
    _C.ADAMATCH.ENABLE = False
    _C.ADAMATCH.SOURCE = ["OR3"]
    _C.ADAMATCH.TARGET = ["OR4"]
    _C.ADAMATCH.TARGET_ANNOT = None
    _C.ADAMATCH.ALPHA = 1
    _C.ADAMATCH.MU = 1.0
    _C.ADAMATCH.THRESHOLDING = CfgNode()
    _C.ADAMATCH.THRESHOLDING.ENABLE = True
    _C.ADAMATCH.THRESHOLDING.TYPE = "uniform"
    _C.ADAMATCH.THRESHOLDING.TAU = 0.9
    _C.ADAMATCH.ALIGNMENT = CfgNode()
    _C.ADAMATCH.ALIGNMENT.ENABLE = True
    _C.ADAMATCH.SAMPLING = CfgNode()
    _C.ADAMATCH.SAMPLING.ENABLE = True
    _C.ADAMATCH.SEMI_SUPERVISED = CfgNode()
    _C.ADAMATCH.SEMI_SUPERVISED.ENABLE = False
    _C.ADAMATCH.SEMI_SUPERVISED.LAB_RATIO = 0.1
    _C.ADAMATCH.BASELINE = None

    # MME Options
    _C.MME = CfgNode()
    _C.MME.LAMBDA = 0.1
    _C.MME.TEMP = 0.05
    _C.MME.ETA = 1.0

    # AdaEmbed Options
    _C.ADAEMBED = CfgNode()
    _C.ADAEMBED.LAMBDA_H = 0.1
    _C.ADAEMBED.LAMBDA_T = 1.0
    _C.ADAEMBED.LAMBDA_P = 0.0
    _C.ADAEMBED.LAMBDA_C = 0.1
    _C.ADAEMBED.TEMP = 0.05
    _C.ADAEMBED.EMA = 0.999
    _C.ADAEMBED.TAU = 0.9
    _C.ADAEMBED.NUM_NEIGHBORS = 10
    _C.ADAEMBED.PSEUDO_TYPE = "AdaMatch"
    _C.ADAEMBED.THRESHOLDING = "uniform"
    _C.ADAEMBED.ALIGNMENT = True
    _C.ADAEMBED.SAMPLING = True
