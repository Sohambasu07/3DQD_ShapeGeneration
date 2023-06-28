import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResNet_block3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1)
        self.dropout1 = nn.Dropout3d(dropout_rate)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1)
        self.dropout2 = nn.Dropout3d(dropout_rate)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x + input
        return x
    
class Attention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=1, 
                               stride=1, padding=0)
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=1, 
                             stride=1, padding=0)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1, 
                               stride=1, padding=0)
        self.projection = nn.Conv3d(in_channels, in_channels, kernel_size=1, 
                                    stride=1, padding=0)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        B, C, D, H, W = q.shape
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)
        attn_wts = torch.bmm(q.permute(0, 2, 1), k)
        attention = torch.bmm(v, F.softmax(attn_wts*(C**-0.5), 
                                           dim=2).permute(0, 2, 1))
        attention = attention.view(B, C, D, H, W)
        attention = self.projection(attention)
        x = x + attention

        return x
    
    class UpSample3D(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                                  stride=1, padding=1) 
            # Transposed Conv checkerboard artifacts: https://distill.pub/2016/deconv-checkerboard/
            # self.bn = nn.BatchNorm3d(out_channels)

        def forward(self, x):
            x = F.interpolate(x, scale_factor=2, mode='trilinear', 
                              align_corners=True)
            x = self.conv(x)
            # x = self.bn(x)
            return x
        

    class DownSample3D(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                                  stride=2, padding=1)
            # self.bn = nn.BatchNorm3d(out_channels)

        def forward(self, x):
            x = self.conv(x)
            # x = self.bn(x)
            return x
        