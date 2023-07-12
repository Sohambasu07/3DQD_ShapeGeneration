import numpy as np
import torch
import trimesh
import skimage

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
    return folded_x_head

def display_tsdf(tsdf):
    tsdf = tsdf.numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(tsdf, level=0.05)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh.show()


if __name__ == '__main__':
    # test_x = torch.zeros((512, 1, 8, 8, 8))
    test_x = torch.randn((1, 1, 64, 64, 64))

    from tsdf_dataset import ShapeNet
    from torch.utils.data import DataLoader
 
    shapenet_dataset = ShapeNet(r'./dataset')
    data_loader = DataLoader(shapenet_dataset, batch_size=2, shuffle=True)
    tsdf_sample = next(iter(data_loader))
    model_path = tsdf_sample[1][0]
    tsdf_x = tsdf_sample[0][0]
    display_tsdf(tsdf_x)

    tsdf_x = torch.reshape(tsdf_x, (1, 1, *tsdf_x.shape))
    print(tsdf_x.shape)

    test_x = shape2patch(tsdf_x)
    folded_x = patch2shape(test_x)
    display_tsdf(folded_x)

    folded_x = torch.unsqueeze(torch.unsqueeze(folded_x, dim=0), dim=0)

    if test_x.all() == folded_x.all():
        print("Success")
    print(folded_x.shape)
