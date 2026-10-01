

from pathlib import Path

import torch


class DigitRecognitionModel(torch.nn.Module):
    def __init__(self, input_size=784, output_size=10):
        super(DigitRecognitionModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_size)
        )
        weights_path = Path(__file__).resolve().parent / "shared_weights.pth"
        self.load_weights(torch.load(weights_path, map_location="cpu"))


    def forward(self, x):
        out = self.network(x)
        return out

    def load_weights(self, weights):
        self.network.load_state_dict(weights)
