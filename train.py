import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from tsdf_dataset import ShapeNet
from model.pvqvae.vqvae import VQVAE
from utils import shape2patch, patch2shape
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import wandb
import logging
from torchinfo import summary

# Training loop
def train(model, train_dataloader, val_dataloader, 
          criterion, learning_rate, optimizer, num_epoch=5,  device='cuda'):
    
    logging.basicConfig(level=logging.INFO)
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

    logging.info("Starting training")

    val_loss_bench = 100000
    for epoch in range(num_epoch):
        # Training
        model.train()  # Set the model to train mode

        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epoch))

        avr_tot_loss_buffer = []
        avr_recon_loss_buffer = []
        avr_vq_loss_buffer = []
        tqdm_dataloader = tqdm(train_dataloader)
        for batch_idx, tsdf_sample in enumerate(tqdm_dataloader):
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
            iter_no = epoch * len(train_dataloader) + batch_idx
            avr_tot_loss = np.mean(avr_tot_loss_buffer)
            avr_recon_loss = np.mean(avr_recon_loss_buffer)
            avr_vq_loss = np.mean(avr_vq_loss_buffer)
            if batch_idx % 50 == 0:
            #     iter_no = epoch * len(data_loader) + batch_idx
            #     avr_tot_loss = np.mean(avr_tot_loss_buffer)
            #     avr_recon_loss = np.mean(avr_recon_loss_buffer)
            #     avr_vq_loss = np.mean(avr_vq_loss_buffer)
            #     print(f"Epoch {epoch}/{num_epoch} - Batch {batch_idx}/{len(dataloader)} Total Loss: {avr_tot_loss} Recon Loss: {avr_recon_loss}, Vq Loss: {avr_vq_loss}")
                writer.add_scalar('Total loss/Train', avr_tot_loss, iter_no)
                writer.add_scalar('Recon loss/Train', avr_recon_loss, iter_no)
                writer.add_scalar('VQ loss/Train', avr_vq_loss, iter_no)
            wandb.log({"Total loss/Train": avr_tot_loss, "Recon loss/Train": avr_recon_loss, "VQ loss/Train": avr_vq_loss})
            tqdm_dataloader.set_postfix_str("Total Loss: {:.4f} Recon Loss: {:.4f}, Vq Loss: {:.4f}".format(
                                                                                avr_tot_loss, avr_recon_loss, avr_vq_loss))
        
        print()
        
        # Validation
        model.eval()

        val_total_loss = 0

        val_total_loss_buffer = []
        val_recon_loss_buffer = []
        val_vq_loss_buffer = []

        vtqdm_dataloader = tqdm(val_dataloader)
        for batch_idx, tsdf_sample in enumerate(vtqdm_dataloader):
            model_path = tsdf_sample[1][0]
            tsdf = tsdf_sample[0][0]

            tsdf = tsdf.to(device)
            tsdf = torch.reshape(tsdf, (1, 1, *tsdf.shape))
            patched_tsdf = shape2patch(tsdf)
            with torch.no_grad():
                reconstructed_data, val_vq_loss = model(patched_tsdf)
                val_recon_loss = criterion(reconstructed_data, patched_tsdf)
            val_total_loss = val_recon_loss + val_vq_loss

            val_total_loss_buffer.append(val_total_loss.item())
            val_recon_loss_buffer.append(val_recon_loss.item())
            val_vq_loss_buffer.append(val_vq_loss.item())

            val_avr_tot_loss = np.mean(val_total_loss_buffer)
            val_avr_recon_loss = np.mean(val_recon_loss_buffer)
            val_avr_vq_loss = np.mean(val_vq_loss_buffer)
            
            writer.add_scalar('Total loss/Val', val_avr_tot_loss, epoch)
            writer.add_scalar('Recon loss/Val', val_avr_recon_loss, epoch)
            writer.add_scalar('VQ loss/Val', val_avr_vq_loss, epoch)

            wandb.log({"Total loss/Val": val_avr_tot_loss, "Recon loss/Val": val_avr_recon_loss, "VQ loss/Val": val_avr_vq_loss})
            vtqdm_dataloader.set_postfix_str("Val Total Loss: {:.4f} Val Recon Loss: {:.4f}, Val Vq Loss: {:.4f}".format(
                                                                                val_avr_tot_loss, val_avr_recon_loss, val_avr_vq_loss))
        if val_avr_tot_loss < val_loss_bench:
            val_loss_bench = val_total_loss
            torch.save(model.state_dict(), './best_model.pth')
            logging.info("Model saved")

    writer.close()

if __name__ == '__main__':
    shapenet_dataset = ShapeNet(r'./dataset',
                                split_ratio={'train': 0.8, 'val': 0.1, 'test': 0.1})    
    train_dataset, val_dataset, test_dataset = shapenet_dataset.split_dataset()


    #Saving the test dataset paths

    test_indices = test_dataset.indices
    test_datapaths = [shapenet_dataset.paths[i] for i in test_indices]

    test_savepaths = './test_dataset_paths.txt'

    # if os.path.exists(test_savepaths):
    #     os.remove(test_savepaths)

    with open(test_savepaths, 'w') as f:
        for item in test_datapaths:
            f.write("%s\n" % item)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Create model object
    embed_dim = 256
    num_embed = 128
    model = VQVAE(embed_dim, num_embed).to(device)
    summary(model, input_size=(512, 1, 8, 8, 8))
    # x_head, vq_loss = model(tsdf)

    criterion = nn.MSELoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch = 5

    train(model, train_dataloader=train_loader, 
                 val_dataloader=val_loader, 
                 criterion=criterion, 
                 learning_rate=learning_rate, 
                 optimizer=optimizer, 
                 num_epoch=epoch, 
                 device=device)
    