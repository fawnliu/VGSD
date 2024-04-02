# The model for location-aware single image reflection removal.

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

from collections import OrderedDict

class Conv2DLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, k_size, stride, padding=None, dilation=1, norm=None, act=None, bias=False):
        super(Conv2DLayer, self).__init__()
        # use default padding value or (kernel size // 2) * dilation value
        if padding is not None:
            padding = padding
        else:
            padding = dilation * (k_size - 1) // 2

        self.add_module('conv2d', nn.Conv2d(in_channels, out_channels, k_size, stride, padding, dilation=dilation, bias=bias))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
        if act is not None:
            self.add_module('act', act)


class SElayer(nn.Module):
    # The SE_layer(Channel Attention.) implement, reference to:
    # Squeeze-and-Excitation Networks
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True), 
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)

        return x * y


class ResidualBlock(nn.Module):
    # The ResBlock implements: the conv & skip connections here.
    # Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf.
    # Which contains SE-layer implements.

    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, se_reduction=None, res_scale=1, act=nn.ReLU(True)):
        super(ResidualBlock, self).__init__()

        print('Residual block: ', channel, norm, dilation, se_reduction, res_scale)

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SElayer(channel, se_reduction)
    
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se_layer:
            x = self.se_layer(x)
        x = x * self.res_scale
        out = x + res
        return out


class ChannelAttention(nn.Module):
    # The channel attention block
    # Original relize of CBAM module.
    # Sigma(MLP(F_max^c) + MLP(F_avg^c)) -> output channel attention feature.
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    
        self.fc_1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu = nn.ReLU(True)
        self.fc_2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_output = self.fc_2(self.relu(self.fc_1(self.avg_pool(x))))
        max_output = self.fc_2(self.relu(self.fc_1(self.max_pool(x))))
        out = avg_output + max_output
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    # The spatial attention block.
    # Simgoid(conv([F_max^s; F_avg^s])) -> output spatial attention feature.
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in [3, 7], 'kernel size must be 3 or 7.'
        padding_size = 1 if kernel_size == 3 else 3

        self.conv = nn.Conv2d(2, 1, padding=padding_size, bias=False, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        pool_out = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(pool_out)
        return self.sigmoid(x)


class CBAMlayer(nn.Module):
    # THe CBAM module(Channel & Spatial Attention feature) implement
    # reference from paper: CBAM(Convolutional Block Attention Module)
    def __init__(self, channel, reduction=16):
        super(CBAMlayer, self).__init__()
        self.channel_layer = ChannelAttention(channel, reduction)
        self.spatial_layer = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_layer(x) * x
        x = self.spatial_layer(x) * x
        return x 


class ResidualCbamBlock(nn.Module):
    # The ResBlock which contain CBAM attention module.

    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, cbam_reduction=None, act=nn.ReLU(True)):
        super(ResidualCbamBlock, self).__init__()

        print('ResidualCbam block: ', channel, norm, dilation, cbam_reduction)

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.cbam_layer = None
        if cbam_reduction is not None:
            self.cbam_layer = CBAMlayer(channel, cbam_reduction)
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.cbam_layer:
            x = self.cbam_layer(x)
        
        out = x + res
        return out 
