#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .custom_video_model_builder import *  # noqa
from .video_model_builder import ResNet, SlowFast  # noqa
from .vit import vit_base_patch16_224, TimeSformer  # noqa
from .swin_transfomer import SwinTransformer
from .swin_transfomer3d import SwinTransformer3D

try:
    from .ptv_model_builder import (
        PTVCSN,
        PTVX3D,
        PTVR2plus1D,
        PTVResNet,
        PTVSlowFast,
    )  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
