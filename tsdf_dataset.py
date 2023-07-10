import trimesh
import pickle
import os
import json
import skimage

from torch.utils.data import Dataset, DataLoader, random_split
from utils import display_tsdf

class ShapeNet(Dataset):
    def __init__(self, dataset_dir, split = {'train': True, 'val': True, 'test': True}, 
                                    split_ratio = {'train': 0.8, 'val': 0.1, 'test': 0.1}, 
                                    ):
        
        self.split = split
        self.split_ratio = split_ratio

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
    
    def split_dataset(self):
        split_dataset = random_split(self, [int(len(self)*self.split_ratio['train']), 
                                            int(len(self)*self.split_ratio['val']), 
                                            int(len(self)*self.split_ratio['test'])])
        return split_dataset[0], split_dataset[1], split_dataset[2]
    


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
    
    display_tsdf(tsdf)
    


