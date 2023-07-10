import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tsdf_dataset import ShapeNet
from model.pvqvae.encoder import Encoder3D
from model.pvqvae.decoder import Decoder3D
from model.pvqvae.vqvae import VQVAE
from utils import shape2patch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb

# Training loop
def train(model, dataloader, criterion, learning_rate, optimizer, num_epoch=5,  device='cuda'):
    writer = SummaryWriter()

    wandb.login()

    wandb.init(
        # set the wandb project where this run will be logged
        project="3dqd-pvqvae",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "PVQVAE",
        "dataset": "ShapeNetv2",
        "optimizer": optimizer.__class__.__name__,
        "epochs": num_epoch,
        }
    )

    model.train()  # Set the model to train mode
    for epoch in range(num_epoch):
        print(f'Epoch {epoch} -------------------------------------')
        avr_tot_loss_buffer = []
        avr_recon_loss_buffer = []
        avr_vq_loss_buffer = []
        for batch_idx, tsdf_sample in enumerate(dataloader):
            model_path = tsdf_sample[1][0]
            tsdf = tsdf_sample[0][0]

            optimizer.zero_grad()  # Clear the gradients

            tsdf = tsdf.to(device)
            tsdf = torch.reshape(tsdf, (1, 1, *tsdf.shape))
            patched_tsdf = shape2patch(tsdf)
            reconstructed_data, vq_loss = model(patched_tsdf)  # Forward pass
            recon_loss = criterion(reconstructed_data, patched_tsdf)  # Compute the loss

            total_loss = recon_loss + vq_loss
            total_loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights

            avr_tot_loss_buffer.append(total_loss.item())
            avr_recon_loss_buffer.append(recon_loss.item())
            avr_vq_loss_buffer.append(vq_loss.item())
            if batch_idx % 50 == 0:
                iter_no = epoch * len(data_loader) + batch_idx
                avr_tot_loss = np.mean(avr_tot_loss_buffer)
                avr_recon_loss = np.mean(avr_recon_loss_buffer)
                avr_vq_loss = np.mean(avr_vq_loss_buffer)
                print(f"Epoch {epoch}/{num_epoch} - Batch {batch_idx}/{len(dataloader)} Total Loss: {avr_tot_loss} Recon Loss: {avr_recon_loss}, Vq Loss: {avr_vq_loss}")
                writer.add_scalar('Total loss/Train', avr_tot_loss, iter_no)
                writer.add_scalar('Recon loss/Train', avr_recon_loss, iter_no)
                writer.add_scalar('VQ loss/Train', avr_vq_loss, iter_no)
        wandb.log({"Total loss/Train": avr_tot_loss, "Recon loss/Train": avr_recon_loss, "VQ loss/Train": avr_vq_loss})
    writer.close()

if __name__ == '__main__':
    shapenet_dataset = ShapeNet(r'./dataset')
    data_loader = DataLoader(shapenet_dataset, batch_size=1, shuffle=True)


    # tsdf = tsdf.reshape(-1, 1, 8, 8, 8)  # [8, 1, 8, 8, 8] i.e. 8 patches for one mesh

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Create the models
    # encoder = Encoder3D().to(device)
    # decoder = Decoder3D().to(device)
    # print(encoder(tsdf).shape)

    # tsdf_sample = next(iter(data_loader))
    # model_path = tsdf_sample[1][0]
    # tsdf = tsdf_sample[0][0]
    # tsdf = tsdf.to(device)


    embed_dim = 256
    num_embed = 128
    model = VQVAE(embed_dim, num_embed).to(device)
    # x_head, vq_loss = model(tsdf)

    criterion = nn.MSELoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch = 5

    train(model, data_loader, criterion, learning_rate, optimizer, epoch, device)
    