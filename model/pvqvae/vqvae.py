import torch
import torch.nn as nn

from model.pvqvae.encoder import Encoder3D
from model.pvqvae.decoder import Decoder3D
from model.pvqvae.vector_quantizer import VectorQuantizer
from utils import shape2patch, fold_to_voxels

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embed_dim, codebook_dropout=False, 
                 codebook_dropout_prob=0.3, resnet_dropout_rate=0.5,
                 replace_threshold=0.01, replace_batches=40, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.encoder = Encoder3D(in_channels=1, dropout_rate=resnet_dropout_rate)
        self.decoder = Decoder3D(dropout_rate=resnet_dropout_rate)
        self.vq = VectorQuantizer(n_embed=num_embeddings, e_dim=embed_dim,
                                    codebook_dropout=codebook_dropout, 
                                    codebook_dropout_prob=codebook_dropout_prob,
                                    replace_threshold=replace_threshold,
                                    replace_batches=replace_batches)

    def forward(self, patched_tsdf, is_training=True):
        # x = torch.reshape(x, (1, 1, *x.shape))
        # patched_tsdf = shape2patch(x)
        encoded = self.encoder(patched_tsdf)
        z_q, _ = self.vq(encoded, is_training)
        # vq_loss = torch.tensor(0.0, dtype=torch.float32).to(z_q.device)
        # commitment_loss = torch.tensor(0.0, dtype=torch.float32).to(z_q.device)

        #z_q.shape [512, 256, 1, 1, 1]
        #voxel_z_q [1, 256, 8, 8, 8]
        batch_size = z_q.shape[0] // 512
        voxel_z_q = fold_to_voxels(x_cubes=z_q, batch_size=batch_size, ncubes_per_dim=8)
        x_head = self.decoder(voxel_z_q)

        return x_head
    



