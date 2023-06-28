import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pvqvae.modules import ResNet_block3D, Attention3D, UpSample3D, DownSample3D

class Encoder3D(nn.Module):
    def __init__(self, in_channels, conv1_outchannels=64):
        super().__init__()
        self.in_channels = in_channels
        self.conv1_outchannels = conv1_outchannels

        self.conv1 = torch.nn.Conv3d(in_channels, conv1_outchannels,
                                        kernel_size=3, stride=1, padding=1)
            

