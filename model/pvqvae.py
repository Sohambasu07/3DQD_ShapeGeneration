import torch
import torch.nn as nn
import torch.nn.functional as F

class PVQVAEModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def shape2patch(self, x, patch_size=8, stride=8):
        B, C, D, H, W = x.shape
        x = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride).unfold(4, patch_size, stride)
        x = x.view(B, C, patch_size**3, -1)
        x = (x.permute(0, 2, 1, 3, 4, 5)).view(B*patch_size**3, C, D, H, W)
        return x