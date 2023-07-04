import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pvqvae.modules import ResNet_block3D, Attention3D, UpSample3D, DownSample3D, groupNorm

class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, conv1_outchannels=64):
        super().__init__()
        self.in_channels = in_channels
        self.conv1_outchannels = conv1_outchannels
        self.outch = conv1_outchannels

        self.conv1 = torch.nn.Conv3d(in_channels, conv1_outchannels,
                                        kernel_size=3, stride=1, padding=1)
            
        self.modules = nn.ModuleList()
        for i in range(3):
            if i == 1:
                self.outch *= 2
            self.modules.append(ResNet_block3D(self.outch, self.outch))
            self.modules.append(DownSample3D(self.outch, self.outch))

        self.outch *= 2
        self.modules.append(ResNet_block3D(self.outch, self.outch))
        self.modules.append(ResNet_block3D(self.outch, self.outch))
        self.modules.append(Attention3D(self.outch))
        self.modules.append(ResNet_block3D(self.outch, self.outch))

        self.conv_out = nn.Conv3d(self.outch, self.outch, kernel_size=3, 
                                  stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for module in self.modules:
            x = module(x)
        x = nn.SiLU(x)
        x = self.conv_out(x)
        return x