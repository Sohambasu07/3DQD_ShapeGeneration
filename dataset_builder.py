import os
from utils import obj_to_tsdf
import numpy as np
import pickle
import json
import math
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="root folder of the 3D models", default='../ShapeNetCore.v2/')
    args = parser.parse_args()

    root_folder = args.path
    save_root_folder = 'dataset'
    if not os.path.exists(save_root_folder):
        os.mkdir(save_root_folder)

    class_ids = {'plane': '02691156', 'chair': '03001627', 'table': '04379243'}
    discarded_samples = ['de45798ef57fe2d131b4f9e586a6d334']
    num_points = 16*16*16
    # num_points = 512
    threshold = 0.8
    gen = np.linspace(-1, 1, math.ceil(num_points**(1/3)))
    points3D = np.array([np.array([x, y, z]) for x in gen for y in gen for z in gen])

    dataset_info = {'num_points':num_points, 'threshold':threshold, 'points3D':points3D.tolist(), 'class_ids':class_ids}
    with open(os.path.join(save_root_folder, 'dataset_info.json'), 'w') as fp:
        json.dump(dataset_info, fp)


    tsdf_no = 0
    for cls_name in class_ids:

        class_save_folder = os.path.join(save_root_folder, cls_name)
        if not os.path.exists(class_save_folder):
            os.mkdir(class_save_folder)
        cls_path = os.path.join(root_folder, class_ids[cls_name])
        for sample_id in os.listdir(cls_path):

            sample_path = os.path.join(cls_path, sample_id + '/models/model_normalized.obj')
            if not os.path.isfile(sample_path) or sample_id in discarded_samples:
                print(f"Invalid file: {sample_path}")
                continue
        
            tsdf_save_path = os.path.join(class_save_folder, f'{cls_name}_{tsdf_no}.pkl')
            tsdf_no += 1
            if os.path.exists(tsdf_save_path):
                print(f"Already created: {tsdf_save_path}")
                continue
            else:
                print(sample_path)

            tsdf = obj_to_tsdf(sample_path, threshold, points3D)

            tsdf_sample = {'tsdf':tsdf, 'model_path':sample_path}

            with open(tsdf_save_path, 'wb') as f:
                pickle.dump(tsdf_sample, f)

            
        