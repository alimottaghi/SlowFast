import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict
from .build import MODEL_REGISTRY

class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name=''):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        #self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)

@MODEL_REGISTRY.register()
class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    def __init__(self, cfg, spatial_squeeze=True, in_channels=3):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """
        super(InceptionI3d, self).__init__()
        self.dropout_keep_prob = cfg.MODEL.DROPOUT_RATE
        self._num_classes = cfg.MODEL.NUM_CLASSES       
        self._spatial_squeeze = spatial_squeeze
        self.extract_features = cfg.EXTRACT.ENABLE
        self.in_channels = in_channels
        #super(InceptionI3d, self).__init__()
        self._construct_network(cfg)

    def _construct_network(self,cfg):
        #self._num_classes = num_classes
        #self._spatial_squeeze = spatial_squeeze
        #self._final_endpoint = final_endpoint
        #self.extract_features = cfg.EXTRACT.ENABLE
        #self.logits = None

        self.c1 = Unit3D(in_channels=self.in_channels, output_channels=64, kernel_shape=[7, 7, 7],stride=(2, 2, 2), padding=(3,3,3))
        
        self.p1 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
       
        self.c2 = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0)
        
        self.c3 = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1)

        self.p2 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),padding=0)
        
        self.i1 = InceptionModule(192, [64,96,128,16,32,32])

        self.i2 = InceptionModule(256, [128,128,192,32,96,64])

        self.p3 = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)

        self.i3 = InceptionModule(128+192+96+64, [192,96,208,16,48,64])
        
        self.i4 = InceptionModule(192+208+48+64, [160,112,224,24,64,64])

        self.i5 = InceptionModule(160+224+64+64, [128,128,256,24,64,64])
        
        self.i6 = InceptionModule(128+256+64+64, [112,144,288,32,64,64])
        
        self.i7 = InceptionModule(112+288+64+64, [256,160,320,32,128,128])
        
        self.p4 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        
        self.i8 = InceptionModule(256+320+128+128, [256,160,320,32,128,128])
        
        self.i9 = InceptionModule(256+320+128+128, [384,192,384,48,128,128])
        
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(self.dropout_keep_prob)
        #self.logits = nn.Conv1d(384+384+128+128,self.num_classes,1)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True)
        #self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = nn.Conv1d(384+384+128+128, self._num_classes, 1) 
       
    def forward(self, x):
        #print(f"Input shape: {x.size()}")
        x = self.c1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.p2(x)
        x = self.i1(x)
        x = self.i2(x)
        x = self.p3(x)
        x = self.i3(x)
        x = self.i4(x)
        x = self.i5(x)
        x = self.i6(x)
        x = self.i7(x)
        x = self.p4(x)
        x = self.i8(x)
        x = self.i9(x)
        #print(f"Feautes before pool: {x.size()}")
        x = self.avg_pool(x)
        #print(f"Features after pool: {x.size()}")
        #print(f"Features after squeeze: {torch.max(torch.squeeze(x),dim=2)[0].size()}") 
        logits = self.logits(self.dropout(x))
        #print(f"Logits shape: {logits.size()}")
        if self._spatial_squeeze:
            logits = logits.squeeze(3).squeeze(3)
        #print(f"Logits after spatial squeeze {logits.size()}")
        logits = torch.max(logits, dim=2)[0]
        #print(f"Logits after frame-wise max: {logits.size()}")
        if self.extract_features:
            return logits, torch.max(torch.squeeze(x),dim=2)[0] if len(torch.squeeze(x).size())>2 else torch.squeeze(x)
        else:
            return logits
       

   # def extract_features(self, x):
   #     for end_point in self.VALID_ENDPOINTS:
   #         if end_point in self.end_points:
   #             x = self._modules[end_point](x)
   #     return self.avg_pool(x)
        
