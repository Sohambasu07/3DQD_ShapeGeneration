import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed=512, e_dim=256, beta=0.25):
        super().__init__()
        self.n_embed = n_embed
        self.e_dim = e_dim
        self.beta = beta

        # Create a lookup table with n_embed items, each of size e_dim
        self.embedding = nn.Embedding(self.n_embed, self.e_dim)

    def forward(self, z):
        z_flattened = z.view(-1, self.e_dim)

        # # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        #     torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
        #     torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        # Don't understand this in the paper implementation. Is this the correct way?

        # Get all embeddings
        indices = torch.arange(self.n_embed).to('cuda')
        embeddings = self.embedding(indices)

        # Calculate dot product similarity between z and embeddings
        similarity = torch.mm(z_flattened, embeddings.t())

        codebook_idxs = torch.argmax(similarity, dim=-1)
        z_q = self.embedding(codebook_idxs).view(z.shape)

        # Third term in the loss but implemented differently from what is in the paper
        vq_loss = torch.mean((z_q - z.detach()) ** 2) 
        commitment_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, vq_loss, commitment_loss, codebook_idxs
