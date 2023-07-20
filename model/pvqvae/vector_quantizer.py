import torch
import torch.nn as nn
from einops import rearrange


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed=512, e_dim=256, beta=0.25,
                 codebook_dropout=False, codebook_dropout_prob=0.3,
                 replace_threshold=0.01, replace_batches=40, eps=1e-12):
        super().__init__()
        self.n_embed = n_embed
        self.e_dim = e_dim
        self.beta = beta
        self.codebook_dropout = codebook_dropout
        self.codebook_dropout_prob = codebook_dropout_prob
        self.replace_threshold = replace_threshold
        self.replace_batches = replace_batches
        self.eps = eps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.codebook_hist = torch.zeros(self.n_embed).to(self.device)

        # Create a lookup table with n_embed items, each of size e_dim
        self.embedding = nn.Embedding(self.n_embed, self.e_dim)


    def update_codebook(self):
        with torch.no_grad():
            most_frequent_idx = torch.argmax(self.codebook_hist)
            most_frequent_embed = self.embedding(most_frequent_idx)
            # _, most_used_idxs = torch.mode(self.codebook_hist)
            new_weights = torch.zeros_like(self.embedding.weight.data)
            new_weights[:] = most_frequent_embed
            self.embedding.weight = nn.Parameter(new_weights)


    def forward(self, z, is_training=True):
        z_flattened = z.view(-1, self.e_dim)
        # print(z_flattened.shape)

        if self.codebook_dropout and is_training:
            # generate a random permutation of indices for the tensor
            indices = torch.randperm(self.n_embed)

            # select the first 70% of the indices
            num_selected = int((1 - self.codebook_dropout_prob) * self.n_embed)
            indices, _ = torch.sort(indices[:num_selected])
            indices = indices.to(self.device)
            embeddings = self.embedding(indices)
        else:
            # Get all embeddings
            embeddings = self.embedding.weight

        # # Calculate dot product similarity between z and embeddings
        # similarity_old = torch.mm(z_flattened, embeddings.t())

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        similarity = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                     torch.sum(embeddings ** 2, dim=1) - 2 * \
                     torch.einsum('bd,dn->bn', z_flattened, rearrange(embeddings, 'n d -> d n'))

        codebook_idxs = torch.argmin(similarity, dim=-1)
        # print(codebook_idxs.shape)
        # print(codebook_idxs)
        z_q = self.embedding(codebook_idxs).view(z.shape)

        if is_training:
            if self.codebook_dropout:
                codebook_idxs = indices[codebook_idxs]

            self.codebook_hist[codebook_idxs] += 1

            # vq_loss = torch.mean((z_q - z.detach()) ** 2)
        # commitment_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

        # preserve gradients
        # z_q = z + (z_q - z).detach()

        if is_training:
            noise = torch.rand_like(z_q)
            z_q = z + torch.norm(z - z_q) * noise / torch.norm(noise)

        return z_q, codebook_idxs

    def replace_codebook_entries(self):

        with torch.no_grad():

            unused_indices = torch.where((self.codebook_hist.cpu() / self.replace_batches) < self.replace_threshold)[0]
            used_indices = torch.where((self.codebook_hist.cpu() / self.replace_batches) >= self.replace_threshold)[0]

            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]

            if used_count == 0:
                # print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                self.embedding.weight += self.eps * torch.randn(self.embedding.weight.size(),
                                                                device=self.device).clone()
            else:
                used = self.embedding.weight[used_indices].clone()
                if used_count < unused_count:
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                else:
                    used_codebooks = used

                self.embedding.weight[unused_indices] *= 0
                self.embedding.weight[unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                    (unused_count, self.e_dim), device=self.device).clone()

            # print(f'************* Replaced ' + str(unused_count) + f' codebooks *************')
            self.codebook_hist[:] = 0.0
