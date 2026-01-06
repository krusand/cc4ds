
import torch
from torch import nn

import typer

import matplotlib.pyplot as plt
from tqdm import tqdm

from data import corrupt_mnist
from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    with torch.no_grad():
        batch_test_loss = 0
        batch_accuracy = []
        for images, labels in test_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            log_ps = model(images)
            ps = torch.exp(log_ps)

            top_p, top_class = ps.topk(k=1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            accuracy = torch.mean(equals.type(torch.FloatTensor))
            batch_accuracy.append(accuracy.item())
        accuracy = torch.mean(torch.Tensor(batch_accuracy))
        print(100*accuracy.item(), "%")


if __name__ == "__main__":
    evaluate(model_checkpoint="models/s1_model.pt")
