import torch
import argparse
import keyboard

from model.pvqvae.vqvae import VQVAE
from utils import fold_to_voxels, display_tsdf

def display_embedding(model, z_q_empty_space, embedding_idx):
    embedding = model.vq.embedding.weight[embedding_idx]

    # z_q = z_q_empty_space.clone()
    z_q = z_q_empty_space
    middle_idx = z_q_empty_space.shape[0] // 2
    z_q[middle_idx] = embedding.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    voxel_z_q = fold_to_voxels(x_cubes=z_q, batch_size=1, ncubes_per_dim=8)
    rec_data = model.decoder(voxel_z_q).squeeze().squeeze()
    rec_data = rec_data.detach().cpu()

    display_tsdf(rec_data, mc_level=(rec_data.max() + rec_data.min()) / 2.0)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display model embeddings')
    parser.add_argument('--load_model_path', type=str, default='./best_model.pth', help='Path to the saved model')
    parser.add_argument('--num_embed', type=int, default=128, help='Number of embeddings')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Create model object
    num_embed = args.num_embed
    embed_dim = args.embed_dim
    model = VQVAE(num_embeddings=num_embed, embed_dim=embed_dim).to(device)

    model.load_state_dict(torch.load(args.load_model_path))
    print("Model loaded")
    model.eval()
    # Create tsdf without zero crossing to get embedding for empty space. Already patched
    tsdf_empty_space = torch.rand(size=(512, 1, 8, 8, 8), device=device)

    encoded_empty_space = model.encoder(tsdf_empty_space)
    z_q_empty_space = model.vq.inference(encoded_empty_space)

    embedding_idx = 0
    display_embedding(model, z_q_empty_space, embedding_idx)

    while True:
        if keyboard.is_pressed('left'):
            embedding_idx -= 1
            print(f'{embedding_idx=}')
            display_embedding(model, z_q_empty_space, embedding_idx)
    
        elif keyboard.is_pressed('right'):
            embedding_idx += 1
            print(f'{embedding_idx=}')
            display_embedding(model, z_q_empty_space, embedding_idx)

