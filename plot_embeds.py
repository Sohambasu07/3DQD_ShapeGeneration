import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from model.pvqvae.vqvae import VQVAE
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser(description='Plot model embeddings')
    parser.add_argument('--load_model_path', type=str, default='./best_model.pth', help='Path to the saved model')
    parser.add_argument('--num_embed', type=int, default=512, help='Number of embeddings')
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

    embeddings =  model.vq.embedding.weight
    embeddings = embeddings.detach().cpu().numpy()
    print(embeddings.shape)

    pca = PCA(n_components=2)  # Choose the number of components you want (in this case, 2 for visualization purposes)
    data_transformed = pca.fit_transform(embeddings)
    principal_components = pca.components_  # Returns the principal components
    explained_variance = pca.explained_variance_ratio_  # Returns the variance explained by each component

    #filter outliers
    data_transformed = data_transformed[data_transformed[:, 0] < 0.04]
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], marker='o', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Quantizer Embeddings')
    plt.show()

    

embedding_idx = 0
if __name__ == "__main__":
    main()