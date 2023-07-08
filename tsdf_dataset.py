import numpy as np
import trimesh
import math
from utils import createCmap, viz_trimesh
import pickle
import os
import json
import skimage

from torch.utils.data import Dataset, DataLoader

class ShapeNet(Dataset):
    def __init__(self, dataset_dir):

        with open(os.path.join(dataset_dir, 'dataset_info.json')) as fp:
            meta_data = json.load(fp)

        self.num_points = meta_data['num_points']
        # self.points3D = meta_data['points3D']
        self.class_ids = meta_data['class_ids']
    
        self.paths = []
        for class_folder in os.listdir(dataset_dir):
            class_dir = os.path.join(dataset_dir, class_folder)
            if os.path.isdir(class_dir):
                for tsdf_sample in os.listdir(class_dir):
                    sample_path = os.path.join(class_dir, tsdf_sample)
                    self.paths.append(sample_path)

    
    def __getitem__(self, index):
        with open(self.paths[index], 'rb') as f:
            tsdf = pickle.load(f)

        return tsdf['tsdf'], tsdf['model_path']
    
    def __len__(self):
        return len(self.paths)
    


if __name__ == '__main__':
    #load a saved tsdf file and display to verify
    # with open('dataset/plane/plane_3.pkl', 'rb') as f:
    #     tsdf_sample = pickle.load(f)

    # tsdf= tsdf_sample['tsdf']
    # model_path = tsdf_sample['model_path']

    shapenet_dataset = ShapeNet(r'./dataset')
    data_loader = DataLoader(shapenet_dataset, batch_size=2, shuffle=True)
    tsdf_sample = next(iter(data_loader))
    model_path = tsdf_sample[1][0]
    tsdf = tsdf_sample[0][0]
    print(model_path)
    print(tsdf.shape)
    


    # num_points = 512

    # mesh = trimesh.load_mesh(model_path)
    # meshes = mesh.dump(concatenate=True)
    # merged_mesh = trimesh.util.concatenate(meshes)
    # merged_mesh.show()

    # grid = np.linspace(-1, 1, math.ceil(num_points**(1/3)))
    # points3D = np.array([np.array([x, y, z]) for x in grid for y in grid for z in grid])

    # colormap = createCmap(tsdf)

    # viz_trimesh(merged_mesh, points3D, tsdf, colormap)

    tsdf = tsdf.numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(tsdf, level=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh.show()


