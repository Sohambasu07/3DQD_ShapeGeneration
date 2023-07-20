import torch
import torch.nn as nn

from model.pvqvae.modules import ResNet_block3D, Attention3D, UpSample3D, DownSample3D, Normalize


class Decoder3D(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()

        ch = 64
        ch_mult = [1,2,2,4]
        self.num_resolutions = len(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]  # 64 * 4
        self.dropout_rate = dropout_rate

        self.conv_in = torch.nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.res1 = ResNet_block3D(256, 256, dropout_rate=self.dropout_rate)
        self.attn1 = Attention3D(256)
        self.res2 = ResNet_block3D(256, 256, dropout_rate=self.dropout_rate)
        # self.up1 = UpSample3D(256, 256, kernel_size=3, stride=2, padding=0)
        self.up1 = UpSample3D(256, 256, kernel_size=3)

        self.res3 = ResNet_block3D(256, 128, dropout_rate=self.dropout_rate)
        self.attn2 = Attention3D(128)
        self.up2 = UpSample3D(128, 128, kernel_size=3)

        self.res4 = ResNet_block3D(128, 64, dropout_rate=self.dropout_rate)
        self.up3 = UpSample3D(64, 64, kernel_size=3)
        self.res5 = ResNet_block3D(64, 64, dropout_rate=self.dropout_rate)
        self.res6 = ResNet_block3D(64, 64, dropout_rate=self.dropout_rate)
        self.norm = Normalize(64)
        self.swish = nn.SiLU()

        self.conv_out = torch.nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_in(x)

        x = self.res1(x)
        x = self.attn1(x)
        x = self.res2(x)
        x = self.up1(x)

        x = self.res3(x)
        x = self.attn2(x)
        x = self.up2(x)
        x = self.res4(x)
        x = self.up3(x)

        x = self.res5(x)
        x = self.res6(x)

        x = self.norm(x)
        x = self.swish(x)

        x = self.conv_out(x)

        return x
