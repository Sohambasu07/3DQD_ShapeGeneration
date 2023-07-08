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

# Training loop
def train(model, dataloader, criterion, optimizer, num_epoch=5,  device='cuda'):
    writer = SummaryWriter()

    model.train()  # Set the model to train mode
    for epeoch in range(num_epoch):
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

            if batch_idx % 50 == 0:
                iter_no = epeoch * len(data_loader) + batch_idx
                print(f"Batch {iter_no}/{len(dataloader)} Total Loss: {total_loss.item()} Recon Loss: {recon_loss.item()}, Vq Loss: {vq_loss.item()}")
                writer.add_scalar('Total loss/Train', total_loss.item(), iter_no)
                writer.add_scalar('Recon loss/Train', recon_loss.item(), iter_no)
                writer.add_scalar('VQ loss/Train', vq_loss.item(), iter_no)
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch = 5

    train(model, data_loader, criterion, optimizer, epoch, device)
    