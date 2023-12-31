import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResNet_block3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5,
                 kernel_size=3, stride=1, padding=1, use_conv_shortcut=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv3d(self.in_channels, self.out_channels,
                               self.kernel_size, self.stride,
                               self.padding)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = Normalize(self.out_channels) # nn.BatchNorm3d(out_channels)
        self.nonlinearity1 = nn.ReLU()
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels,
                               self.kernel_size, self.stride,
                               self.padding)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = Normalize(self.out_channels) # nn.BatchNorm3d(out_channels)
        self.nonlinearity2 = nn.ReLU()

        self.use_conv_shortcut = use_conv_shortcut

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.norm1(x)
        x = self.nonlinearity1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.norm2(x)
        x = self.nonlinearity2(x)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                inp = self.conv_shortcut(inp)
            else:
                inp = self.nin_shortcut(inp)

        x = x + inp

        return x


class Attention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        self.norm = Normalize(self.in_channels)
        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=1,
                               stride=1, padding=0)
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=1,
                             stride=1, padding=0)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1,
                               stride=1, padding=0)
        self.projection = nn.Conv3d(in_channels, in_channels, kernel_size=1,
                                    stride=1, padding=0)

    def forward(self, x):
        x = self.norm(x)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        B, C, D, H, W = q.shape
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)
        attn_wts = torch.bmm(q.permute(0, 2, 1), k)
        attention = torch.bmm(v, F.softmax(attn_wts * (C ** -0.5),
                                           dim=2).permute(0, 2, 1))
        attention = attention.view(B, C, D, H, W)
        attention = self.projection(attention)
        x = x + attention

        return x


class UpSample3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = 1

        self.conv = nn.Conv3d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding)
        # Transposed Conv checkerboard artifacts: 
        # https://distill.pub/2016/deconv-checkerboard/

        # self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2, mode='trilinear',
        #                   align_corners=True)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        # x = self.bn(x)
        return x


class DownSample3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=2, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv3d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding)

        # self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return x
    
# def groupNorm(x, num_channels, num_groups=32):
#     return nn.GroupNorm(num_groups = num_groups, num_channels = num_channels)

def Normalize(in_channels):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32

    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def VQ(enc_embed : torch.Tensor, codebook : torch.Tensor):
    #encoder embed: batch_size x block_count x Nz
    #codebook: KxNz
    similarity = enc_embed @ codebook.T
    codebook_idxs = torch.argmax(similarity, dim=-1)
    return codebook_idxs