import torch
import torch.nn as nn

from model.pvqvae.encoder import Encoder3D
from model.pvqvae.decoder import Decoder3D
from model.pvqvae.vector_quantizer import VectorQuantizer
from utils import shape2patch

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embed_dim, codebook_dropout=False, codebook_dropout_prob=0.3,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder3D(in_channels=1)
        self.decoder = Decoder3D()
        self.vq = VectorQuantizer(n_embed=num_embeddings, e_dim=embed_dim,
                                    codebook_dropout=codebook_dropout, codebook_dropout_prob=codebook_dropout_prob)

    def forward(self, patched_tsdf, is_training=True):
        # x = torch.reshape(x, (1, 1, *x.shape))
        # patched_tsdf = shape2patch(x)
        encoded = self.encoder(patched_tsdf)
        z_q, vq_loss, commitment_loss, _ = self.vq(encoded, is_training)
        x_head = self.decoder(z_q)

        return x_head, vq_loss, commitment_loss
    



