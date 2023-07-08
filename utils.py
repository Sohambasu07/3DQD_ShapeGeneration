import numpy as np
import trimesh
import trimesh.proximity as prox
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import math
import open3d as o3d
from mesh_to_sdf import mesh_to_voxels
import torch


def createCmap(sequence, cmap='viridis'):
    cmap = cmx.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=min(sequence), vmax=max(sequence))
    colormap = cmap(cNorm(sequence))
    return colormap

def viz_matplotlib(mesh, points, sdf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=sdf)
    plt.show()

def viz_trimesh(mesh, points, sdf, colormap):
    scene = trimesh.scene.Scene(mesh)
    point_cloud = trimesh.points.PointCloud(points, colors=colormap)
    # print(point_cloud.shape)
    scene.add_geometry(point_cloud)
    scene.show()
    # print(sdf.shape)

def truncated_sdf(sdf, threshold):
    # sdf[sdf > threshold] = threshold
    # sdf[sdf < -threshold] = -threshold
    # return sdf
    return np.clip(sdf, -threshold, threshold)

# def trimesh_obj_to_tsdf(obj_path, threshold, points3D, max_triangle_count =5000):
#     scene = trimesh.load_mesh(obj_path)
#     if type(scene) == trimesh.scene.scene.Scene:
#         meshes = scene.dump(concatenate=True)
#     else:
#         meshes = scene
#     merged_mesh = trimesh.util.concatenate(meshes)

#     print(len(merged_mesh.triangles))
#     if len(merged_mesh.triangles) > max_triangle_count:
#         merged_mesh = decimate_mesh(obj_path, max_triangle_count)

#     sdf = prox.signed_distance(merged_mesh, points3D)
#     tsdf = truncated_sdf(sdf, threshold)
#     return tsdf

def obj_to_tsdf(obj_path, threshold, patch_size=64, max_triangle_count =5000):
    scene = trimesh.load_mesh(obj_path)
    if type(scene) == trimesh.scene.scene.Scene:
        meshes = scene.dump(concatenate=True)
    else:
        meshes = scene
    merged_mesh = trimesh.util.concatenate(meshes)

    if len(merged_mesh.triangles) > max_triangle_count:
        merged_mesh = decimate_mesh(obj_path, max_triangle_count)

    # sdf = prox.signed_distance(merged_mesh, points3D)
    sdf = mesh_to_voxels(merged_mesh, patch_size, pad=False)
    tsdf = truncated_sdf(sdf, threshold)
    return tsdf

def decimate_mesh(path, target_triangles):
    mesh_in = o3d.io.read_triangle_mesh(path)
    mesh_in.compute_vertex_normals()

    reduced_mesh = mesh_in.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

    # o3d.visualization.draw_geometries([reduced_mesh]) 

    vertices = reduced_mesh.vertices
    triangles = reduced_mesh.triangles
    reduced_trimesh = trimesh.Trimesh(vertices, triangles)
    return reduced_trimesh

def shape2patch(x, patch_size=8, stride=8):
        B, C, D, H, W = x.shape
        x = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride).unfold(4, patch_size, stride)
        x = x.reshape(B, C, patch_size**3, patch_size, patch_size, patch_size)
        x = (x.permute(0, 2, 1, 3, 4, 5)).view(B*patch_size**3, C, patch_size, patch_size, patch_size)
        return x
    
def patch2shape(x_head, patch_size=8, output_size=64):
    #x_head shape (512, 1, 8, 8, 8)
    x_head = x_head[:, 0]
    print(x_head.shape)
    # fold = torch.nn.Fold(output_size=(output_size, output_size, output_size), kernel_size=(patch_size,patch_size,patch_size))/
    folded_x_head = torch.split(x_head, dim=0, split_size_or_sections=8)
    # folded_x_head = fold(x_head)
    return folded_x_head


if __name__ == '__main__':
    test_x = torch.zeros((512, 1, 8, 8, 8))
    folded_x = patch2shape(test_x)
    print(len(folded_x))
