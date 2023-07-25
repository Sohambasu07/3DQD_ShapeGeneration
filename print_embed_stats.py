import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from model.pvqvae.vqvae import VQVAE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from tsdf_dataset import ShapeNet
from tqdm import tqdm
from utils import shape2patch, patch2shape, display_tsdf
from functools import reduce


def main():
    parser = argparse.ArgumentParser(description='Plot model embeddings')
    parser.add_argument('--load_model_path', type=str, default='./best_model.pth', help='Path to the saved model')
    parser.add_argument('--num_embed', type=int, default=512, help='Number of embeddings')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--splits', nargs='+', type=float, default=[0.8, 0.1, 0.1], help='Train, Val, Test splits')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Create model object
    num_embed = args.num_embed
    embed_dim = args.embed_dim
    model = VQVAE(num_embeddings=num_embed, embed_dim=embed_dim).to(device)

    model.load_state_dict(torch.load(args.load_model_path, map_location=torch.device('cpu')))
    print("Model loaded")
    model.eval()

    # Get empty space embeddings
    samples_per_type = 2
    test_dataset = [0.2 * torch.rand(size=(512, 1, 8, 8, 8), device=device) for i in range(samples_per_type)]
    test_dataset.extend([-0.2 * torch.rand(size=(512, 1, 8, 8, 8), device=device) for i in range(samples_per_type)])
    test_dataset.extend([0.2 * torch.ones(size=(512, 1, 8, 8, 8), device=device) for i in range(samples_per_type)])

    class_names = ['rand', 'neg_rand', 'ones']
    class_names_repeated = [item for item in class_names for i in range(samples_per_type)]
    class_histograms = {name: torch.zeros(num_embed) for name in class_names}

    tqdm_dataloader = tqdm(test_dataset)
    for batch_idx, tsdf_sample in enumerate(tqdm_dataloader):
        patched_tsdf = tsdf_sample
        with torch.no_grad():
            model(patched_tsdf, is_training=False)
            # print(f'{model.vq.codebook_hist.sum()=}')
            sample_class = class_names_repeated[batch_idx]
            class_histograms[sample_class] += model.vq.codebook_hist.cpu()
            model.vq.reset_histogram()

    empty_space_idx = []
    for class_name in class_histograms:
        # fig, ax = plt.subplots()
        codebook_hist = class_histograms[class_name]

        count_dict = {}
        for index, count in enumerate(codebook_hist):
            if count == samples_per_type:
                count_dict[index] = count
                empty_space_idx.append(index)

        # print(f'\n{class_name}: {count_dict=}')
        # ax.bar(np.arange(len(codebook_hist)), codebook_hist)
        # plt.show()

    # print(f'{empty_space_idx=}')
    empty_space_idx.sort()
    empty_space_idx = np.array(empty_space_idx)
    # ------------------
    # Explore embeddings for shapenet classes

    # Load test dataset
    shapenet_dataset = ShapeNet(dataset_dir=args.dataset_path, split={'train': False, 'val': False, 'test': True},
                                split_ratio={'train': args.splits[0], 'val': args.splits[1], 'test': args.splits[2]})

    test_paths = []
    with open('./test_dataset_paths.txt', 'r') as f:
        for line in f:
            if line != '\n':
                test_paths.append(line.strip())

    # print(f'{test_paths=}')
    test_indices = [shapenet_dataset.paths.index(path) for path in test_paths]
    test_dataset = Subset(shapenet_dataset, test_indices)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Collect histogram of each class
    # class_names = ['chair', 'table', 'plane']
    class_names = ['chair', 'table', 'bed', 'bench']
    class_histograms = {name: torch.zeros(num_embed) for name in class_names}
    samples_per_class_dict = {k: 0 for k in class_names}

    tqdm_dataloader = tqdm(test_loader)
    for batch_idx, tsdf_sample in enumerate(tqdm_dataloader):
        tsdf_path = tsdf_sample[2][0]
        tsdf = tsdf_sample[0][0]
        # find the class of the sample
        sample_class = ''
        for class_name in class_names:
            if class_name in tsdf_path:
                sample_class = class_name
                samples_per_class_dict[class_name] += 1
                break
        if sample_class == '':
            continue

        tsdf = tsdf.to(device)
        tsdf = torch.reshape(tsdf, (1, 1, *tsdf.shape))
        patched_tsdf = shape2patch(tsdf)
        with torch.no_grad():
            model(patched_tsdf, is_training=False)
            class_histograms[sample_class] += model.vq.codebook_hist.cpu()
            model.vq.reset_histogram()

    # take embeddings of each class that are used in at least 50% of samples for that class
    used_percentage = 0.5
    most_used_indeces_by_class = {}
    indeces_sets = []
    for class_name in class_histograms:
        hist = class_histograms[class_name]

        hist_of_used_percentages = hist / samples_per_class_dict[class_name]
        most_used_indeces = (hist_of_used_percentages >= used_percentage).nonzero(as_tuple=True)[0]
        # print(f'{hist=}\n{hist_of_used_percentages=}\n{most_used_indeces=}')
        # sorted_hist, indeces = torch.sort(hist, descending=True)
        # most_used_indeces = indeces[:int(used_percentage * num_embed)].numpy()
        most_used_indeces_by_class[class_name] = most_used_indeces.numpy()
        # indeces_sets.append(set(most_used_indeces))
        indeces_sets.append(most_used_indeces)

        # fig, ax = plt.subplots()
        # codebook_hist = class_histograms[class_name]
        #
        # count_dict = {}
        # for index, count in enumerate(codebook_hist):
        #     if count == samples_per_type:
        #         count_dict[index] = count
        #
        # print(f'\n{class_name}: {count_dict=}')
        # ax.bar(np.arange(len(codebook_hist)), codebook_hist)
        # plt.show()

    # find the common indeces for using a different color
    # common_indeces = set.intersection(*indeces_sets)
    # print(indeces_sets)
    common_indeces = np.intersect1d(*indeces_sets)
    common_indeces = np.setdiff1d(common_indeces, empty_space_idx)
    print(f'empty space: {len(empty_space_idx)} embeddings. {empty_space_idx}')
    print(f'Embeddings common bw all classes: {len(common_indeces)} emb. {common_indeces}')

    for class_name in most_used_indeces_by_class:
        other_classes_most_used = [most_used_indeces_by_class[cls] for cls in most_used_indeces_by_class
                                                 if cls != class_name]
        other_classes_most_used.append(empty_space_idx)
        union_of_all_other_classes = reduce(np.union1d, other_classes_most_used)
        idx_only_this_class = np.setdiff1d(most_used_indeces_by_class[class_name], union_of_all_other_classes)
        print(f'{class_name}: {len(idx_only_this_class)} embeddings. {idx_only_this_class}')

    # Get infrequently used and dead embeddings
    dead_threshold = 0.01
    combined_hist = torch.stack(list(class_histograms.values())).sum(dim=0)
    total_samples = sum(samples_per_class_dict.values())
    dead_idx = ((combined_hist / total_samples) <= dead_threshold).nonzero(as_tuple=True)[0]
    print(f'dead embeddings: {len(dead_idx)} embeddings. {dead_idx}')


    # print(f'{most_used_indeces_by_class}=')

    # # plot the embeddings
    # colors = plt.get_cmap('plasma', len(class_names) + 1)
    # class_colors = {class_name: colors(i) for i, class_name in enumerate(class_names)}
    # class_colors['common'] = colors(len(class_names))
    # marker_list = ['*', 'x', 'P', 's']
    # markers = {class_name: marker_list[i] for i, class_name in enumerate(class_names)}
    # for class_name in most_used_indeces_by_class:
    #     most_used_indeces = most_used_indeces_by_class[class_name]
    #     not_common_idxs = []
    #     for idx in most_used_indeces:
    #         if idx not in common_indeces:
    #             not_common_idxs.append(idx)
    #     print(f'{not_common_idxs=}')
    #     most_used_embeds = reduced_embeds[not_common_idxs]
    #     # filter outlier embeddings
    #     # most_used_embeds = most_used_embeds[most_used_embeds[:, 0] < 0.04]
    #     plt.scatter(most_used_embeds[:, 0], most_used_embeds[:, 1], marker=markers[class_name],
    #                 color=class_colors[class_name], alpha=0.8, label=class_name)
    #
    # # plot the common points with a different color
    # common_most_used_embeds = reduced_embeds[list(common_indeces)]
    # plt.scatter(common_most_used_embeds[:, 0], common_most_used_embeds[:, 1], marker='o', color=class_colors['common'],
    #             alpha=1, label='common')
    #
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('Most Used Quantizer Embeddings')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
