from pathlib import Path

import typer
from torch.utils.data import Dataset

import torch
import os

import matplotlib.pyplot as plt  # only needed for plotting
import torch
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting

def normalize(images):
    return (images - images.mean() / images.std())

def preprocess_data(raw_dir, processed_dir):
    # exchange with the corrupted mnist dataset
    train_images, train_target = [], []

    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images = torch.load(f"{raw_dir}/test_images.pt")
    test_target = torch.load(f"{raw_dir}/test_target.pt")

    train_images = normalize(train_images.unsqueeze(1).float())
    test_images = normalize(test_images.unsqueeze(1).float())

    train_target = train_target.long()
    test_target = test_target.long()

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    train_images = torch.load(f"data/preprocessed/train_images.pt")
    train_target = torch.load(f"data/preprocessed/train_target.pt")
    test_images = torch.load(f"data/preprocessed/test_images.pt")
    test_target = torch.load(f"data/preprocessed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    
    return train_set, test_set


def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    typer.run(preprocess_data)