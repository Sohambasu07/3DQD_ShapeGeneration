import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse

from tsdf_dataset import ShapeNet
from model.pvqvae.vqvae import VQVAE
from utils import shape2patch, patch2shape, log_reconstructed_mesh
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import wandb
import logging
from torchinfo import summary
import matplotlib.pyplot as plt
from PIL import Image

# Training loop
def train(model, train_dataloader, val_dataloader, 
          criterion, learning_rate, optimizer, num_epoch=5, 
          L1_lambda = 0.001, device='cuda', experiment_params='', 
          replace_codebook=True, replace_batches=40):
    
    logging.basicConfig(level=logging.INFO)
    writer = SummaryWriter(comment=f'nsvq-{experiment_params}-L1_lambda={L1_lambda}')

    # wandb.login()

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="3dqd-pvqvae",
        
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": learning_rate,
    #     "architecture": "PVQVAE",
    #     "dataset": "ShapeNetv2",
    #     "optimizer": optimizer.__class__.__name__,
    #     "epochs": num_epoch,
    #     }
    # )

    agg_codebook_hist = torch.zeros(model.vq.n_embed).cpu()

    logging.info("Starting training")

    val_loss_bench = 100000.0
    for epoch in range(num_epoch):
        # Training
        model.train()  # Set the model to train mode

        replace_batches *= (epoch + 1)

        if replace_batches >= len(train_dataloader):
            replace_codebook = False

        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epoch))

        avr_tot_loss_buffer = []
        avr_recon_loss_buffer = []
        avr_vq_loss_buffer = []
        avr_com_loss_buffer = []
        avr_reg_loss_buffer = []

        tqdm_dataloader = tqdm(train_dataloader)
        for batch_idx, tsdf_sample in enumerate(tqdm_dataloader):

            # Replacing codebook entries
            if replace_codebook:
                if batch_idx != 0 and batch_idx % replace_batches == 0:
                    model.vq.replace_codebook_entries()
                #     # logging.info("Codebook entries replaced")

            model_path = tsdf_sample[1][0]
            tsdf = tsdf_sample[0][0]

            optimizer.zero_grad()  # Clear the gradients

            tsdf = tsdf.to(device)
            tsdf = torch.reshape(tsdf, (1, 1, *tsdf.shape))
            patched_tsdf = shape2patch(tsdf)
            reconstructed_data, vq_loss, com_loss = model(patched_tsdf)  # Forward pass
            # reconstructed_data = patch2shape(patched_recon_data)
            recon_loss = criterion(reconstructed_data, tsdf)  # Compute the loss

            # Adding regularization

            L1_penalty = nn.L1Loss()
            L1_regloss = 0.0
            for param in model.parameters():
                L1_regloss += L1_penalty(param, torch.zeros_like(param))
            L1_regloss = L1_lambda * L1_regloss

            total_loss = recon_loss + L1_regloss
            total_loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights
            with torch.no_grad():
                avr_tot_loss_buffer.append(total_loss.item())
                avr_recon_loss_buffer.append(recon_loss.item())
                avr_vq_loss_buffer.append(vq_loss.item())
                avr_com_loss_buffer.append(com_loss.item())
                avr_reg_loss_buffer.append(L1_regloss.item())

                iter_no = epoch * len(train_dataloader) + batch_idx
                avr_tot_loss = np.mean(avr_tot_loss_buffer)
                avr_recon_loss = np.mean(avr_recon_loss_buffer)
                avr_vq_loss = np.mean(avr_vq_loss_buffer)
                avr_com_loss = np.mean(avr_com_loss_buffer)
                avr_reg_loss = np.mean(avr_reg_loss_buffer)

                if batch_idx  % 50 == 0:
                #     iter_no = epoch * len(data_loader) + batch_idx
                #     avr_tot_loss = np.mean(avr_tot_loss_buffer)
                #     avr_recon_loss = np.mean(avr_recon_loss_buffer)
                #     avr_vq_loss = np.mean(avr_vq_loss_buffer)
                #     print(f"Epoch {epoch}/{num_epoch} - Batch {batch_idx}/{len(dataloader)} Total Loss: {avr_tot_loss} Recon Loss: {avr_recon_loss}, Vq Loss: {avr_vq_loss}")
                    writer.add_scalar('Total loss/Train', avr_tot_loss, iter_no)
                    writer.add_scalar('Recon loss/Train', avr_recon_loss, iter_no)
                    writer.add_scalar('VQ loss/Train', avr_vq_loss, iter_no)
                    writer.add_scalar('Commit loss/Train', avr_com_loss, iter_no)
                    writer.add_scalar('Regularization loss/Train', avr_reg_loss, iter_no)
                    agg_codebook_hist = log_codebook_idx_histogram(model, writer, iter_no, agg_codebook_hist)

                    # wandb.log({'Codebook index hist': wandb.Histogram((model.vq.codebook_hist).cpu().numpy())})

            # wandb.log({"Total loss/Train": avr_tot_loss, 
            #            "Recon loss/Train": avr_recon_loss, 
            #            "VQ loss/Train": avr_vq_loss,
            #            "Commit loss/Train": avr_com_loss,
            #            "Regularization loss/Train": avr_reg_loss})
            
            tqdm_dataloader.set_postfix_str("Total Loss: {:.4f}, Recon Loss: {:.4f}, Vq Loss: {:.4f}, Commit Loss: {:.4f}, Reg Loss: {:.4f}"\
                                            .format(avr_tot_loss, avr_recon_loss, 
                                            avr_vq_loss, avr_com_loss, avr_reg_loss))
        with torch.no_grad():
            log_reconstructed_mesh(tsdf, reconstructed_data, writer, model_path, epoch)
        
        print()
        
        # Validation
        model.eval()

        val_avr_tot_loss = 0.0

        val_total_loss_buffer = []
        val_recon_loss_buffer = []
        val_vq_loss_buffer = []
        val_com_loss_buffer = []

        vtqdm_dataloader = tqdm(val_dataloader)
        for batch_idx, tsdf_sample in enumerate(vtqdm_dataloader):
            model_path = tsdf_sample[1][0]
            tsdf = tsdf_sample[0][0]

            tsdf = tsdf.to(device)
            tsdf = torch.reshape(tsdf, (1, 1, *tsdf.shape))
            patched_tsdf = shape2patch(tsdf)
            with torch.no_grad():
                reconstructed_data, val_vq_loss, val_com_loss = model(patched_tsdf, is_training=False)
                val_recon_loss = criterion(reconstructed_data, tsdf)
            val_total_loss = val_recon_loss + val_vq_loss + val_com_loss

            val_total_loss_buffer.append(val_total_loss.item())
            val_recon_loss_buffer.append(val_recon_loss.item())
            val_vq_loss_buffer.append(val_vq_loss.item())
            val_com_loss_buffer.append(val_com_loss.item())

            val_avr_tot_loss = np.mean(val_total_loss_buffer)
            val_avr_recon_loss = np.mean(val_recon_loss_buffer)
            val_avr_vq_loss = np.mean(val_vq_loss_buffer)
            val_avr_com_loss = np.mean(val_com_loss_buffer)

            vtqdm_dataloader.set_postfix_str("Val Total Loss: {:.4f}, Val Recon Loss: {:.4f}, Val Vq Loss: {:.4f}, Commit Loss: {:.4f}"\
                                             .format(val_avr_tot_loss, val_avr_recon_loss, 
                                                     val_avr_vq_loss, val_avr_com_loss))
            
        writer.add_scalar('Total loss/Val', val_avr_tot_loss, epoch)
        writer.add_scalar('Recon loss/Val', val_avr_recon_loss, epoch)
        writer.add_scalar('VQ loss/Val', val_avr_vq_loss, epoch)
        writer.add_scalar('Commit loss/Val', val_avr_com_loss, epoch)    
        # wandb.log({"Total loss/Val": val_avr_tot_loss, 
        #            "Recon loss/Val": val_avr_recon_loss, 
        #            "VQ loss/Val": val_avr_vq_loss,
        #            "Commit loss/Val": val_avr_com_loss})

        torch.save(model.state_dict(), './final_model.pth')
        logging.info("Model saved")
        logging.info(val_avr_tot_loss)
        logging.info(val_loss_bench)
            
        if val_avr_tot_loss < val_loss_bench:
            val_loss_bench = val_avr_tot_loss
            torch.save(model.state_dict(), './best_model.pth')
            logging.info("Best Model saved")
            logging.info(val_avr_tot_loss)
            logging.info(val_loss_bench)

    writer.close()

def log_codebook_idx_histogram(model, writer, iter_no, agg_codebook_hist):
    fig, ax = plt.subplots()
    agg_codebook_hist += model.vq.codebook_hist.cpu()
    # ax.bar(np.arange(len(model.vq.codebook_hist)), model.vq.codebook_hist.cpu())
    ax.bar(np.arange(len(agg_codebook_hist)), agg_codebook_hist)
    tmp_file = 'histog.png'
    fig.savefig(tmp_file, format='png')
    plt.close(fig)
    codebook_hist =  np.asarray(Image.open(tmp_file))
    writer.add_image('Codebook index hist', codebook_hist[:,:,:3], iter_no, dataformats="HWC")
    # wandb.log({'Codebook index hist': wandb.Image(codebook_hist[:,:,:3])})
    return agg_codebook_hist

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a PVQVAE model')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='Path to the dataset')
    parser.add_argument('--splits', nargs='+', type=float, default=[0.8, 0.1, 0.1], help='Train, Val, Test splits')
    parser.add_argument('--num_embed', type=int, default=128, help='Number of embeddings')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--codebook_dropout', type=bool, default=False, help='Whether to use codebook dropout')
    parser.add_argument('--codebook_dropout_prob', type=float, default=0.3, help='Codebook dropout probability')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--L1_lambda', type=float, default=0.01, help='L1 regularization lambda')
    parser.add_argument('--resnet_dropout', type=float, default=0.5, help='Dropout rate for Resnet blocks')
    parser.add_argument('--replace_codebook', type=bool, default=True, help='Whether to replace codebook entries')
    parser.add_argument('--replace_threshold', type=float, default=0.2, help='Codebook replacement threshold')
    parser.add_argument('--replace_batches', type=int, default=40, help='Number of batches to replace codebook entries')

    args = parser.parse_args()


    shapenet_dataset = ShapeNet(dataset_dir=args.dataset_path,
                                split_ratio={'train': args.splits[0], 'val': args.splits[1], 'test': args.splits[2]})    
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
    num_embed = args.num_embed
    embed_dim = args.embed_dim
    codebook_dropout = args.codebook_dropout
    codebook_dropout_prob = args.codebook_dropout_prob
    experiment_params = f'num_embe={num_embed}-embed_dim={embed_dim}-vq_drop={codebook_dropout}={codebook_dropout_prob}'

    model = VQVAE(num_embeddings=num_embed, 
                  embed_dim=embed_dim, 
                  codebook_dropout=codebook_dropout, 
                  codebook_dropout_prob=codebook_dropout_prob,
                  resnet_dropout_rate=args.resnet_dropout,
                  replace_batches=args.replace_batches,
                  replace_threshold=args.replace_threshold).to(device)
    
    summary(model, input_size=(512, 1, 8, 8, 8))
    # x_head, vq_loss = model(tsdf)

    # Set Hyperparameters
    criterion = nn.MSELoss()
    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch = args.num_epoch
    L1_lambda = args.L1_lambda

    train(model, train_dataloader=train_loader, 
                 val_dataloader=val_loader, 
                 criterion=criterion, 
                 learning_rate=learning_rate, 
                 optimizer=optimizer, 
                 num_epoch=epoch,
                 L1_lambda=L1_lambda, 
                 device=device,
                 experiment_params=experiment_params,
                 replace_codebook=args.replace_codebook,
                 replace_batches=args.replace_batches)
    