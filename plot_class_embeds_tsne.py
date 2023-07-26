import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from model.pvqvae.vqvae import VQVAE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tsdf_dataset import ShapeNet
from tqdm import tqdm
from utils import shape2patch, patch2shape, display_tsdf


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

    # Load test dataset
    shapenet_dataset = ShapeNet(dataset_dir=args.dataset_path, split={'train': False, 'val': False, 'test': True},
                                split_ratio={'train': args.splits[0], 'val': args.splits[1], 'test': args.splits[2]})

    test_paths = []
    with open('./test_dataset_paths.txt', 'r') as f:
        for line in f:
            if line != '\n':
                test_paths.append(line.strip())

    print(f'{test_paths=}')
    test_indices = [shapenet_dataset.paths.index(path) for path in test_paths]
    test_dataset = Subset(shapenet_dataset, test_indices)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Collect histogram of each class
    # class_names = ['chair', 'table', 'plane']
    class_names = ['chair', 'table', 'bed', 'bench']
    class_histograms = {name: torch.zeros(num_embed) for name in class_names}
    tqdm_dataloader = tqdm(test_loader)
    for batch_idx, tsdf_sample in enumerate(tqdm_dataloader):
        tsdf_path = tsdf_sample[2][0]
        tsdf = tsdf_sample[0][0]
        # find the class of the sample
        sample_class = ''
        for class_name in class_names:
            if class_name in tsdf_path:
                sample_class = class_name
                break
        if sample_class == '':
            continue

        tsdf = tsdf.to(device)
        tsdf = torch.reshape(tsdf, (1, 1, *tsdf.shape))
        patched_tsdf = shape2patch(tsdf)
        with torch.no_grad():
            model(patched_tsdf, is_training=False)
            # print(f'{model.vq.codebook_hist.sum()=}')
            class_histograms[sample_class] += model.vq.codebook_hist.cpu()
            model.vq.reset_histogram()

    # reduce dim of embeddings
    embeddings = model.vq.embedding.weight
    embeddings = embeddings.detach().cpu().numpy()
    # pca = PCA(n_components=2)  # Choose the number of components you want (in this case, 2 for visualization purposes)
    # reduced_embeds = pca.fit_transform(embeddings)
    pca = PCA(n_components=50)  # Choose the number of components you want (in this case, 2 for visualization purposes)
    reduced_embeds = pca.fit_transform(embeddings)
    reduced_embeds = TSNE(n_components=2).fit_transform(reduced_embeds)
    # principal_components = pca.components_  # Returns the principal components
    # explained_variance = pca.explained_variance_ratio_  # Returns the variance explained by each component

    # take most used embeddings of each class
    used_percentage = 0.1
    most_used_indeces_by_class = {}
    indeces_sets = []
    for class_name in class_histograms:
        hist = class_histograms[class_name]
        sorted_hist, indeces = torch.sort(hist, descending=True)
        most_used_indeces = indeces[:int(used_percentage * num_embed)].numpy()
        most_used_indeces_by_class[class_name] = most_used_indeces
        indeces_sets.append(set(most_used_indeces))

        fig, ax = plt.subplots()
        codebook_hist = class_histograms[class_name]
        # print(f'{codebook_hist=}\n{codebook_hist.sum()=}')
        ax.bar(np.arange(len(codebook_hist)), codebook_hist)
        plt.show()
        print(f'{class_name=}')

    # find the common indeces for using a different color
    common_indeces = set.intersection(*indeces_sets)
    print(f'{common_indeces=}')

    # plot the embeddings
    fig = plt.figure()
    colors = plt.get_cmap('plasma', len(class_names) + 1)
    class_colors = {class_name: colors(i) for i, class_name in enumerate(class_names)}
    class_colors['common'] = colors(len(class_names))
    marker_list = ['*', 'x', 'P', 's']
    markers = {class_name: marker_list[i] for i, class_name in enumerate(class_names)}
    for class_name in most_used_indeces_by_class:
        most_used_indeces = most_used_indeces_by_class[class_name]
        not_common_idxs = []
        for idx in most_used_indeces:
            if idx not in common_indeces:
                not_common_idxs.append(idx)
        print(f'{not_common_idxs=}')
        most_used_embeds = reduced_embeds[not_common_idxs]
        # filter outlier embeddings
        # most_used_embeds = most_used_embeds[most_used_embeds[:, 0] < 0.04]
        plt.scatter(most_used_embeds[:, 0], most_used_embeds[:, 1], marker=markers[class_name],
                    color=class_colors[class_name], alpha=0.6, label=class_name)

    # plot the common points with a different color
    common_most_used_embeds = reduced_embeds[list(common_indeces)]
    plt.scatter(common_most_used_embeds[:, 0], common_most_used_embeds[:, 1], marker='o', color=class_colors['common'],
                alpha=1, label='common')

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Most Used Quantizer Embeddings')
    plt.legend()
    fig.savefig('Most Used Quantizer Embeddings_tsne.png', dpi=500)

    plt.show()


if __name__ == "__main__":
    main()
