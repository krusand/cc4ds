
import torch
from torch import nn

import typer

import matplotlib.pyplot as plt
from tqdm import tqdm

from data import corrupt_mnist
from model import MyAwesomeModel

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint: str) -> None:
    """Visualize a trained model."""

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.fc4 = nn.Identity()

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    embeddings, targets = [], []
    with torch.inference_mode():

        for images, labels in test_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(labels)
        embeddings = torch.cat(embeddings).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

    if embeddings.shape[1] > 500: # Use PCA for initial for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10,10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/embedding_visualization.png")

if __name__ == "__main__":
    visualize(model_checkpoint="models/s1_model.pt")
