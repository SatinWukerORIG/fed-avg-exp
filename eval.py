import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitRecognitionModel


def prepare_test_data(batch_size=64):
    data_dir = Path(__file__).resolve().parent / "data"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate_model(model, data_loader, device):
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.flatten(start_dim=1).to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(weights_path=None, batch_size=64):
    if weights_path is None:
        weights_path = Path(__file__).resolve().parent / "trained_weights.pth"
    else:
        weights_path = Path(weights_path)

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Trained weights not found at {weights_path}. Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitRecognitionModel()
    model.load_weights(torch.load(weights_path, map_location="cpu"))
    test_loader = prepare_test_data(batch_size)
    loss, accuracy = evaluate_model(model, test_loader, device)
    print(f"Test loss: {loss:.4f}; test accuracy: {accuracy:.2%}; device: {device}")
    return loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained MNIST model")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parent / "trained_weights.pth",
        help="Path to a model state-dict checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    evaluate(args.weights, args.batch_size)
