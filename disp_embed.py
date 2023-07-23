import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import argparse
from model.pvqvae.vqvae import VQVAE
from utils import fold_to_voxels, get_tsdf_vertices_faces


def get_mesh_components(model, z_q_empty_space, embedding_idx):
    embedding = model.vq.embedding.weight[embedding_idx]

    # z_q = z_q_empty_space.clone()
    z_q = z_q_empty_space
    middle_idx = z_q_empty_space.shape[0] // 2
    z_q[middle_idx] = embedding.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    voxel_z_q = fold_to_voxels(x_cubes=z_q, batch_size=1, ncubes_per_dim=8)
    rec_data = model.decoder(voxel_z_q).squeeze().squeeze()
    rec_data = rec_data.detach().cpu()

    return get_tsdf_vertices_faces(rec_data, mc_level=(rec_data.max() + rec_data.min()) / 2.0)

def display_3d_mesh(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces)

    # Hide axes for a better view
    ax.set_axis_off()
    ax.set_title('Embedding Index: 0')

    plt.ion()  # Turn on interactive mode
    plt.show()

    return fig, ax

def update_mesh(event, ax, fig, model, z_q_empty_space):

    global embedding_idx
    if event.key == 'right':
        embedding_idx += 1
    elif event.key == 'left':
        embedding_idx -= 1

    print(embedding_idx)
    vertices, faces = get_mesh_components(model, z_q_empty_space, embedding_idx)
    # Redraw the updated mesh
    ax.clear()
    ax.set_axis_off()
    ax.set_title(f'Embedding Index: {embedding_idx}')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces)
    fig.canvas.draw()


def main():
    parser = argparse.ArgumentParser(description='Display model embeddings')
    parser.add_argument('--load_model_path', type=str, default='./final_model.pth', help='Path to the saved model')
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

    vertices, faces = get_mesh_components(model, z_q_empty_space, embedding_idx)

    fig, ax = display_3d_mesh(vertices, faces)

    # Connect the keyboard event to the update_mesh function
    fig.canvas.mpl_connect('key_press_event', lambda event: update_mesh(event, ax, fig, model, z_q_empty_space))

    # print("Use arrow keys to move the mesh. Press 'q' to quit.")
    while True:
        plt.pause(0.1)  # Pause to update the display

        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' key to quit
            print("Quitting...")
            break

embedding_idx = 0
if __name__ == "__main__":
    main()
