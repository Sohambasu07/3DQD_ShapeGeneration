import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from tsdf_dataset import ShapeNet
from model.pvqvae.vqvae import VQVAE
import argparse

from utils import shape2patch, display_tsdf

def evaluate(test_dataloader, model, criterion, device='cuda'):

    print("Starting evaluation")

    model.eval()

    test_recon_loss_buffer = []

    tqdm_dataloader = tqdm(test_dataloader)
    for batch_idx, tsdf_sample in enumerate(tqdm_dataloader):
        model_path = tsdf_sample[1][0]
        tsdf = tsdf_sample[0][0]

        tsdf = tsdf.to(device)
        tsdf = torch.reshape(tsdf, (1, 1, *tsdf.shape))
        patched_tsdf = shape2patch(tsdf)
        with torch.no_grad():
            reconstructed_data = model(patched_tsdf, is_training=False)
            test_recon_loss = criterion(reconstructed_data, tsdf)

        test_total_loss = test_recon_loss

        # test_total_loss_buffer.append(test_total_loss.item())
        test_recon_loss_buffer.append(test_recon_loss.item())
        # test_vq_loss_buffer.append(test_vq_loss.item())
        # test_com_loss_buffer.append(test_com_loss.item())

        # test_avr_tot_loss = np.mean(test_total_loss_buffer)
        test_avr_recon_loss = np.mean(test_recon_loss_buffer)
        # test_avr_vq_loss = np.mean(test_vq_loss_buffer)
        # test_avr_com_loss = np.mean(test_com_loss_buffer)

        
        tqdm_dataloader.set_postfix_str("Test Recon Loss: {:.4f}".format(test_avr_recon_loss))
        
        if batch_idx == np.random.randint(0, len(test_dataloader)-1):
            # rec_data = patch2shape(reconstructed_data)
            # rec_data = rec_data.squeeze().squeeze()
            # print(rec_data.min(), rec_data.max())
            rec_data = reconstructed_data.squeeze().squeeze()
            rec_data = rec_data.cpu()
            print(rec_data.shape)
            # rec_data = rec_data.cpu()
            print((rec_data.max()+rec_data.min())/2.0)
            display_tsdf(rec_data, mc_level=(rec_data.max()+rec_data.min())/2.0)#(rec_data.max()+rec_data.min())/2.0)
        
        # print(model.vq.codebook_hist)


        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate PVQVAE')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--splits', nargs='+', type=float, default=[0.8, 0.1, 0.1], help='Train, Val, Test splits')
    parser.add_argument('--model_path', type=str, default='./best_model.pth', help='Path to model')
    parser.add_argument('--num_embed', type=int, default=128, help='Number of embeddings')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')

    args = parser.parse_args()

    # Load test dataset
    shapenet_dataset = ShapeNet(dataset_dir=args.dataset_path, split={'train': False, 'val': False, 'test': True},
                                split_ratio={'train': args.splits[0], 'val': args.splits[1], 'test': args.splits[2]})

    test_paths = []
    with open('./test_dataset_paths.txt', 'r') as f:
        for line in f:
            if line!='\n':
                test_paths.append(line.strip())

    test_indices = [shapenet_dataset.paths.index(path) for path in test_paths]
    test_dataset = Subset(shapenet_dataset, test_indices)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Create model object
    embed_dim = args.embed_dim
    num_embed = args.num_embed
    model = VQVAE( num_embed, embed_dim).to(device)

    # Load model
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Load criterion
    criterion = torch.nn.MSELoss()

    evaluate(test_loader, model, criterion, device=device)