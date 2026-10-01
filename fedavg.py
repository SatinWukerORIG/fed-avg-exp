import argparse
from pathlib import Path

import torch


def average_weights(first_path, second_path, output_path="averaged_weights.pth"):
    first_weights = torch.load(first_path, map_location="cpu")
    second_weights = torch.load(second_path, map_location="cpu")

    if first_weights.keys() != second_weights.keys():
        raise ValueError("The weight files must contain the same state-dict keys.")

    averaged_weights = {}
    for key, first_tensor in first_weights.items():
        second_tensor = second_weights[key]
        if first_tensor.shape != second_tensor.shape:
            raise ValueError(f"Weight shape mismatch for {key}.")
        if first_tensor.dtype != second_tensor.dtype:
            raise ValueError(f"Weight dtype mismatch for {key}.")

        if first_tensor.is_floating_point() or first_tensor.is_complex():
            averaged_weights[key] = (first_tensor + second_tensor) / 2
        elif torch.equal(first_tensor, second_tensor):
            averaged_weights[key] = first_tensor.clone()
        else:
            raise ValueError(f"Cannot average differing non-floating weights for {key}.")

    output_path = Path(output_path)
    torch.save(averaged_weights, output_path)
    print(f"Averaged weights saved to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average two model state dicts")
    parser.add_argument("first_weights", type=Path)
    parser.add_argument("second_weights", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "averaged_weights.pth",
    )
    args = parser.parse_args()
    average_weights(args.first_weights, args.second_weights, args.output)
