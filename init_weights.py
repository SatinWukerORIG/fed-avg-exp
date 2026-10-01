"""
Generates the initial weights for the neural network model using He initialization.
Saves the weights to a file to ensure models are trained on the same weights
"""


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

def initialize_weights():
    model = DigitRecognitionModel()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )
            torch.nn.init.zeros_(layer.bias)

    weights_path = Path(__file__).resolve().parent / "shared_weights.pth"
    torch.save(model.network.state_dict(), weights_path)


if __name__ == "__main__":
    initialize_weights()
    print("Initial weights generated and saved to shared_weights.pth")