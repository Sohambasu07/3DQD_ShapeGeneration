import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pvqvae.modules import ResNet_block3D, Attention3D, UpSample3D, DownSample3D, Normalize

class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, conv1_outchannels=64, dropout_rate=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.conv1_outchannels = conv1_outchannels
        self.outch = conv1_outchannels
        self.inch = conv1_outchannels
        self.dropout_rate = dropout_rate

        self.conv1 = torch.nn.Conv3d(in_channels, conv1_outchannels,
                                        kernel_size=3, stride=1, padding=1)

        self.layers = nn.ModuleList()
        for i in range(3):
            if i == 1:
                self.outch *= 2
            self.layers.append(ResNet_block3D(self.inch, self.outch, dropout_rate=self.dropout_rate))
            self.layers.append(DownSample3D(self.outch, self.outch, padding=1))
            self.inch = self.outch

        self.outch *= 2
        self.layers.append(ResNet_block3D(self.inch, self.outch, dropout_rate=self.dropout_rate))
        self.layers.append(ResNet_block3D(self.outch, self.outch, dropout_rate=self.dropout_rate))
        self.layers.append(Attention3D(self.outch))
        self.layers.append(ResNet_block3D(self.outch, self.outch, dropout_rate=self.dropout_rate))

        self.norm = Normalize(self.outch)
        self.swish = nn.SiLU()
        self.conv_out = nn.Conv3d(self.outch, self.outch, kernel_size=3, 
                                  stride=1, padding=1)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)

        for module in self.layers:
            x = module(x)

        x = self.norm(x)
        x = self.swish(x)
        x = self.conv_out(x)
        return x
