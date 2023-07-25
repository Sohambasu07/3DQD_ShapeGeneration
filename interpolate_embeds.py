import matplotlib.pyplot as plt
import keyboard
import torch
import argparse
import pickle 
from model.pvqvae.vqvae import VQVAE
from utils import fold_to_voxels, get_tsdf_vertices_faces, shape2patch, display_tsdf


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
    ax.set_title(f'Mesh1 to Mesh2: 0%')

    plt.ion()  # Turn on interactive mode
    plt.show()

    return fig, ax

def update_mesh(event, ax, fig, stacked_axs, stacked_fig, model, start_z_q, z_q_step_size, num_steps):

    global embedding_idx
    if event.key == 'right':
        if embedding_idx < num_steps:
            embedding_idx += 1
    elif event.key == 'left':
        if embedding_idx > 0:
            embedding_idx -= 1

    print(embedding_idx)
    interpolated_z_q = start_z_q + z_q_step_size * embedding_idx
    vertices, faces = get_mesh_components(model, interpolated_z_q, embedding_idx)
    # Redraw the updated mesh
    ax.clear()
    ax.set_axis_off()
    ax.set_title(f'Mesh1 to Mesh2: {100 * embedding_idx / num_steps:.1f}%')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces)
    fig.canvas.draw()

    if embedding_idx % 2 == 0:
        plot_id = embedding_idx // 2
        # stacked_axs[plot_id].get_figure().canvas.draw()  # Ensure the first figure is drawn (optional)
        # stacked_axs[plot_id].get_figure().canvas.renderer = stacked_axs[plot_id].get_figure().canvas.get_renderer()  # Set the renderer

        # Copy the content from the first original figure to the left subplot of the new figure
        stacked_axs[plot_id].plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces)
        stacked_axs[plot_id].set_title(f'{100 * embedding_idx / num_steps}%')
        stacked_fig.canvas.draw()


def main():
    parser = argparse.ArgumentParser(description='Display model embeddings')
    parser.add_argument('--load_model_path', type=str, default='./best_model.pth', help='Path to the saved model')
    parser.add_argument('--num_embed', type=int, default=512, help='Number of embeddings')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--mesh1_path', type=str, default='./dataset/chair/chair_82.pkl', help='path to input mesh')
    parser.add_argument('--mesh2_path', type=str, default='./dataset/table/table_70.pkl', help='path to input mesh')
    parser.add_argument('--num_steps', type=int, default=10, help='number of step for interpolation')

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
    with open(args.mesh1_path, 'rb') as f:
        tsdf1 = pickle.load(f)

    with open(args.mesh2_path, 'rb') as f:
        tsdf2 = pickle.load(f)

    # display_tsdf(torch.from_numpy(tsdf1['tsdf']), 0)
    z_q_tsdf1 = get_tsdf_vq(device, model, tsdf1)
    z_q_tsdf2 = get_tsdf_vq(device, model, tsdf2)

    num_steps = args.num_steps
    z_q_step_size = (z_q_tsdf2 - z_q_tsdf1) / num_steps

    vertices, faces = get_mesh_components(model, z_q_tsdf1, embedding_idx)

    fig, ax = display_3d_mesh(vertices, faces)
    ax.view_init(azim=-135, vertical_axis='y')

    stacked_fig, stacked_axs = plt.subplots(1, num_steps // 2 + 1, figsize=(20, 5), subplot_kw={'projection': '3d'})
    for ax_ in stacked_axs:
        ax_.set_axis_off()
        ax_.view_init(azim=-135, vertical_axis='y')
    stacked_axs[0].plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces)
    stacked_axs[0].set_title(f'0%')
    stacked_fig.suptitle('Interpolated Embeddings: Mesh1 to Mesh2')


    # Connect the keyboard event to the update_mesh function
    fig.canvas.mpl_connect('key_press_event', lambda event: update_mesh(event, ax, fig, stacked_axs, stacked_fig, model,  z_q_tsdf1, z_q_step_size, num_steps))

    # print("Use arrow keys to move the mesh. Press 'q' to quit.")
    while True:
        plt.pause(0.01)  # Pause to update the display
        if keyboard.is_pressed('q'):  # Press 'q' key to quit
            print("Quitting...")
            break

def get_tsdf_vq(device, model, tsdf):
    tsdf = tsdf['tsdf']
    tsdf = torch.from_numpy(tsdf)
    tsdf = tsdf.to(device)
    input_tsdf = torch.reshape(tsdf, (1, 1, *tsdf.shape))
    patched_tsdf = shape2patch(input_tsdf)

    encoded_tsd = model.encoder(patched_tsdf)
    z_q_tsdf = model.vq.inference(encoded_tsd)
    return z_q_tsdf

embedding_idx = 0
if __name__ == "__main__":
    main()
