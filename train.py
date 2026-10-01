from pathlib import Path
from itertools import islice

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitRecognitionModel


def prepare_data(batch_size=64):
    data_dir = Path(__file__).resolve().parent / "data"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_epoch(model, data_loader, loss_fn, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        batches = data_loader if max_batches is None else islice(data_loader, max_batches)
        for images, labels in batches:
            images = images.flatten(start_dim=1).to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            predictions = model(images)
            loss = loss_fn(predictions, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train(epochs=1, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_data(batch_size)
    model = DigitRecognitionModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batches_per_epoch = max(1, len(train_loader) // 20)
    print(f"Training for {epochs} epochs with {batches_per_epoch} batches per epoch.")

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_fn, device, optimizer, batches_per_epoch
        )
        test_loss, test_accuracy = run_epoch(model, test_loader, loss_fn, device)
        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.2%} "
            f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.2%}"
        )

    weights_path = Path(__file__).resolve().parent / "trained_weights.pth"
    torch.save(model.network.state_dict(), weights_path)
    print(f"Trained weights saved to {weights_path}")
    return model


if __name__ == "__main__":
    train()


