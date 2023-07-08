import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tsdf_dataset import ShapeNet
from model.pvqvae.encoder import Encoder3D
from model.pvqvae.decoder import Decoder3D
from model.pvqvae.vqvae import VQVAE

if __name__ == '__main__':
    shapenet_dataset = ShapeNet(r'./dataset')
    data_loader = DataLoader(shapenet_dataset, batch_size=1, shuffle=True)
    tsdf_sample = next(iter(data_loader))
    # print(tsdf_sample[0].shape)
    model_path = tsdf_sample[1][0]
    tsdf = tsdf_sample[0][0]

    # tsdf = tsdf.reshape(-1, 1, 8, 8, 8)  # [8, 1, 8, 8, 8] i.e. 8 patches for one mesh

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tsdf = tsdf.to(device)


    # Create the models
    # encoder = Encoder3D().to(device)
    # decoder = Decoder3D().to(device)
    #
    # # Define the loss function
    # criterion = nn.MSELoss()
    #
    # # Define the optimizer
    # # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #

    # print(encoder(tsdf).shape)
    embed_dim = 256
    num_embed = 128
    model = VQVAE(embed_dim, num_embed).to(device)
    x_head, vq_loss = model(tsdf)

    # # Training loop
    # def train(model, dataloader, criterion, optimizer, device):
    #     model.train()  # Set the model to train mode
    #
    #     for batch_idx, (data, _) in enumerate(dataloader):
    #         data = data.view(-1, 784).to(device)  # Reshape and move data to device
    #
    #         optimizer.zero_grad()  # Clear the gradients
    #
    #         reconstructed_data = model(data)  # Forward pass
    #         loss = criterion(reconstructed_data, data)  # Compute the loss
    #
    #         loss.backward()  # Backpropagation
    #         optimizer.step()  # Update the weights
    #
    #         if batch_idx % 100 == 0:
    #             print(f"Batch {batch_idx}/{len(dataloader)} Loss: {loss.item()}")