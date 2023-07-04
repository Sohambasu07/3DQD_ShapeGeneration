import torch
import torch.nn as nn

from encoder import Encoder3D
from decoder import Decoder3D
from vector_quantizer import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self, embed_dim, num_embeddings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder3D(in_channels=1)
        self.decoder = Decoder3D()
        self.vq = VectorQuantizer(e_dim=embed_dim, n_embed=num_embeddings)

    def forward(self, x):
        encoded = self.encoder(x)
        self.decoder(encoded)
