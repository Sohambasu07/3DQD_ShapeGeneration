import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from tsdf_dataset import ShapeNet
from model.pvqvae.vqvae import VQVAE
import argparse
import pickle 

from utils import shape2patch, patch2shape, display_tsdf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference PVQVAE')
    parser.add_argument('--mesh_path', type=str, default='./dataset/chair/chair_71.pkl', help='path to input mesh')
    parser.add_argument('--model_path', type=str, default='./best_model.pth', help='Path to model')


    args = parser.parse_args()

    with open(args.mesh_path, 'rb') as f:
        tsdf = pickle.load(f)

    tsdf, _ = tsdf['tsdf'], tsdf['model_path']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    embed_dim = 256
    num_embed = 128
    model = VQVAE(embed_dim, num_embed).to(device)

    # Load model
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    tsdf = torch.from_numpy(tsdf)
    tsdf = tsdf.to(device)
    input_tsdf = torch.reshape(tsdf, (1, 1, *tsdf.shape))
    patched_tsdf = shape2patch(input_tsdf)

    with torch.no_grad():
        patch_recon_data, test_vq_loss, test_com_loss = model(patched_tsdf, is_training=False)
        reconstructed_data = patch2shape(patch_recon_data)
        test_recon_loss = torch.mean((reconstructed_data - tsdf) ** 2)
        print(f'{test_recon_loss=}')
        print(input_tsdf.shape)
        print(reconstructed_data.shape)
        display_tsdf(tsdf.cpu(), 0)
        reconstructed_data = torch.squeeze(reconstructed_data)
        display_tsdf(reconstructed_data.cpu(), (reconstructed_data.max() + reconstructed_data.min())/ 2)

