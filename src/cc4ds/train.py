
import torch
from torch import nn

import typer

import matplotlib.pyplot as plt
from tqdm import tqdm

from data import corrupt_mnist
from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

def train(lr: float = 1e-3) -> None:
    """
    Trains the model and saves it to the models directory. 
    Additionally saves a plot of training loss pr. epoch
    
    Parameters: 
        lr (float): Learning rate for optimizer

    Returns: 
        None
    """

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # AdamW is faster optimizer compared to Adam/SGD

    epochs = 15
    steps = 0

    train_losses = []
    for _ in tqdm(range(epochs)):
        model.train()
        batch_loss = 0
        for images, labels in train_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()

        train_losses.append(batch_loss)

    torch.save(model.state_dict(), "models/s1_model.pt")

    plt.plot(range(0,epochs),train_losses)
    plt.title("Training loss")
    plt.savefig(f"reports/figures/training_loss.png")



if __name__ == "__main__":
    train()
