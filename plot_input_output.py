import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from tsdf_dataset import ShapeNet
from model.pvqvae.vqvae import VQVAE
import argparse
import pickle
import matplotlib.pyplot as plt


from utils import shape2patch, patch2shape, display_tsdf, get_tsdf_vertices_faces

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference PVQVAE')
    parser.add_argument('--model_path', type=str, default='./best_model.pth', help='Path to model')
    parser.add_argument('--num_embed', type=int, default=512, help='Number of embeddings')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')


    args = parser.parse_args()
    mesh_paths = ['./dataset/table/table_129.pkl', './dataset/table/table_952.pkl',
                  './dataset/bench/bench_410.pkl', './dataset/bench/bench_198.pkl',
                  './dataset/chair/chair_629.pkl', './dataset/chair/chair_325.pkl',
                  './dataset/bed/bed_105.pkl', './dataset/bed/bed_155.pkl']
    tsdfs = []
    for mesh_path in mesh_paths:
        with open(mesh_path, 'rb') as f:
            tsdfs.append(pickle.load(f)['tsdf'])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    num_embed = args.num_embed
    embed_dim = args.embed_dim
    model = VQVAE(num_embeddings=num_embed, embed_dim=embed_dim).to(device)

    # Load model
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    model.eval()
    plot_width = 4
    plot_height = len(tsdfs) * 2 // plot_width
    stacked_fig, stacked_axs = plt.subplots(plot_height, plot_width, figsize=(10, 5), subplot_kw={'projection': '3d'})
    print(stacked_axs.shape)

    # stacked_axs[0].set_title(f'0%')


    for i in range(plot_height):
        for j in range(0, plot_width, 2):
            idx = i * plot_width // 2 + j // 2
            print(idx)
            tsdf = tsdfs[idx]
            tsdf = torch.from_numpy(tsdf)
            vertices, faces = get_tsdf_vertices_faces(tsdf, mc_level=(tsdf.max() + tsdf.min()) / 2.0)
            stacked_axs[i, j].plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces)
            stacked_axs[i, j].set_axis_off()
            stacked_axs[i, j].view_init(azim=-135, vertical_axis='y')

            tsdf = tsdf.to(device)
            input_tsdf = torch.reshape(tsdf, (1, 1, *tsdf.shape))
            patched_tsdf = shape2patch(input_tsdf)

            with torch.no_grad():
                # reconstructed_data, test_vq_loss, test_com_loss = model(patched_tsdf, is_training=False)
                reconstructed_data = model(patched_tsdf, is_training=False)
                test_recon_loss = torch.mean((reconstructed_data - tsdf) ** 2)
                print(f'{test_recon_loss=}')
                reconstructed_data = torch.squeeze(reconstructed_data).cpu()
                vertices, faces = get_tsdf_vertices_faces(reconstructed_data, mc_level=(reconstructed_data.max() + reconstructed_data.min()) / 2.0)
                stacked_axs[i, j +1].plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces)
                stacked_axs[i, j + 1].set_axis_off()
                stacked_axs[i, j + 1].view_init(azim=-135, vertical_axis='y')

    stacked_fig.suptitle('Input - Output')
    stacked_fig.tight_layout()
    stacked_fig.savefig('Input-Output.png', dpi=500)
    plt.show()