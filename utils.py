import numpy as np
import torch
import trimesh
import skimage
import pickle
import argparse
import wandb
from einops import rearrange

def shape2patch(x, patch_size=8, stride=8):
        #x shape (1, 1, 64, 64, 64)
        B, C, D, H, W = x.shape
        x = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride).unfold(4, patch_size, stride)
        # print(x.shape)
        x = x.reshape(B, C, patch_size**3, patch_size, patch_size, patch_size)
        x = (x.permute(0, 2, 1, 3, 4, 5)).view(B*patch_size**3, C, patch_size, patch_size, patch_size)
        # print(x.shape)
        return x
    
def patch2shape(x_head, patch_size=8, output_size=64):
    #x_head shape (512, 1, 8, 8, 8)
    x_head = x_head[:, 0]
    num_patches = output_size // patch_size
    # print(x_head.shape)
    # fold = torch.nn.Fold(output_size=(output_size, output_size, output_size), kernel_size=(patch_size,patch_size,patch_size))/
    # folded_x_head = torch.split(x_head, dim=0, split_size_or_sections=8)
    folded_x_head = x_head.reshape(num_patches, num_patches, num_patches, patch_size, patch_size, patch_size)
    folded_x_head = folded_x_head.permute(0, 3, 1, 4, 2, 5).reshape(output_size, output_size, output_size)
    # folded_x_head = fold(x_head)
    folded_x_head = torch.unsqueeze(torch.unsqueeze(folded_x_head, dim=0), dim=0)
    return folded_x_head

def fold_to_voxels(x_cubes, batch_size, ncubes_per_dim):
    x = rearrange(x_cubes, '(b p) c d h w -> b p c d h w', b=batch_size) 
    x = rearrange(x, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                    p1=ncubes_per_dim, p2=ncubes_per_dim, p3=ncubes_per_dim)
    return x

def display_tsdf(tsdf, mc_level=0.0):
    tsdf = tsdf.numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(tsdf, level=mc_level)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh.show()

def log_reconstructed_mesh(original_tsdf, rec_tsdf, tensorboard_writer, model_path, iter_no):
    original_tsdf = original_tsdf.squeeze().squeeze()
    original_tsdf = original_tsdf.cpu().numpy()
    mc_level = (original_tsdf.max()+original_tsdf.min())/2.0
    vertices, faces, normals, _ = skimage.measure.marching_cubes(original_tsdf, level=mc_level)
    vertices = np.expand_dims(vertices, axis=0)
    faces = np.expand_dims(faces, axis=0)
    tensorboard_writer.add_mesh(model_path, vertices= vertices.copy(), faces=faces.copy(), global_step=iter_no)
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    # obj = trimesh.exchange.obj.export_obj(mesh)
    # wandb.log({model_path: wandb.Object3D(obj)})


    rec_tsdf = rec_tsdf.squeeze().squeeze()
    rec_tsdf = rec_tsdf.cpu().numpy()
    mc_level = (rec_tsdf.max()+rec_tsdf.min())/2.0
    vertices, faces, normals, _ = skimage.measure.marching_cubes(rec_tsdf, level=mc_level)
    vertices = np.expand_dims(vertices, axis=0)
    faces = np.expand_dims(faces, axis=0)
    tensorboard_writer.add_mesh(model_path, vertices= vertices.copy(), faces=faces.copy(), global_step=iter_no+1)
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utils')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='Path to dataset')
    
    args = parser.parse_args()

    # test_x = torch.zeros((512, 1, 8, 8, 8))
    # test_x = torch.randn((1, 1, 64, 64, 64))
    #load a saved tsdf file and display to verify
    file_path = args.dataset_path + '/plane/plane_3.pkl'

    with open(file_path, 'rb') as f:
        tsdf_sample = pickle.load(f)

    test_x= tsdf_sample['tsdf']

    test_x = torch.from_numpy(test_x)
    display_tsdf(test_x, mc_level=0.0)

    test_x = torch.unsqueeze(torch.unsqueeze(test_x, dim=0), dim=0)
    test_x = shape2patch(test_x)
    folded_x = patch2shape(test_x)

    display_tsdf(folded_x.squeeze().squeeze().cpu(), mc_level=0.0)

    if test_x.all() == folded_x.all():
        print("Success")
    print(folded_x.shape)
