import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.checkpoint as checkpoint
from slowfast.models.vit_utils import DropPath, to_2tuple, trunc_normal_
import numpy as np

from .build import MODEL_REGISTRY


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@MODEL_REGISTRY.register()
class ResNet34(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Initialize the ResNet34 model
        self.resnet34 = models.resnet34(weights='IMAGENET1K_V1')
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.num_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Identity()

        # Dropout layer before the head
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE
        self.dropout = nn.Dropout(self.dropout_rate)

        # Replace the classifier
        self.head = nn.Linear(self.num_features, self.num_classes, bias=not cfg.SWIN.FEW_SHOT) if self.num_classes > 0 else nn.Identity()

        # Additional properties from Swin Transformer config (if needed)
        self.extracting = cfg.EXTRACT.ENABLE
        self.few_shot = cfg.SWIN.FEW_SHOT
        self.temp = cfg.SWIN.TEMP
        self.eta = cfg.SWIN.ETA

        # Adapted initialization from Swin Transformer
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # Using ResNet34 features
        x = self.resnet34(x)
        # x = self.resnet34.conv1(x)
        # x = self.resnet34.bn1(x)
        # x = self.resnet34.relu(x)
        # x = self.resnet34.maxpool(x)

        # x = self.resnet34.layer1(x)
        # x = self.resnet34.layer2(x)
        # x = self.resnet34.layer3(x)
        # x = self.resnet34.layer4(x)

        # x = self.resnet34.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return x

    def forward(self, x, extract=False, reverse=False):
        if isinstance(x, list) and len(x) == 1:
            x = x[0]
            
        x = self.forward_features(x)
        feats = x

        if reverse:
            x = grad_reverse(x, self.eta)
        
        if self.few_shot:
            x = F.normalize(x)
            cls_score = self.head(x) / self.temp
        else:
            cls_score = self.head(x)

        if self.extracting or extract:
            return cls_score, feats
        else:
            return cls_score

    def flops(self):
        # Compute FLOPs: This is an example, you might need to adjust the calculation
        flops = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                flops += ... # Compute FLOPs for Conv2D
            elif isinstance(m, nn.Linear):
                flops += ... # Compute FLOPs for Linear
        return flops
