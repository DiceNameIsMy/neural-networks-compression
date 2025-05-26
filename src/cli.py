import argparse
import re

from src.datasets.dataset import CnnDataset, MlpDataset
from src.datasets.mnist_dataset import MiniMNISTDataset, MNISTDataset
from src.datasets.vertebral_dataset import VertebralDataset

MLP_DATASETS_MAPPING: dict[str, type[MlpDataset]] = {
    "vertebral": VertebralDataset,
}

CNN_DATASETS_MAPPING: dict[str, type[CnnDataset]] = {
    "mnist": MNISTDataset,
    "mini-mnist": MiniMNISTDataset,
}

SUPPORTED_DATASETS = list(MLP_DATASETS_MAPPING.keys()) + list(
    CNN_DATASETS_MAPPING.keys()
)


def is_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


def is_filename(value):
    # Only allow filenames, not paths
    if "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError(f"{value} must be a filename, not a path")
    # Optionally, check for valid filename characters
    if not re.match(r"^[\w\-.]+$", value):
        raise argparse.ArgumentTypeError(
            f"{value} contains invalid filename characters"
        )
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural Architecture Search CLI for quantized neural networks."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Dataset to use: mnist, mini-mnist, or vertebral",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=is_positive_int,
        help="Number of training epochs (positive integer)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=is_positive_int,
        help="Batch size for training (positive integer)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=is_filename,
        help="Output filename for results (no path, just filename)",
    )
    parser.add_argument(
        "-g",
        "--generations",
        type=is_positive_int,
        required=True,
        help="Number of NAS generations (positive integer)",
    )
    parser.add_argument(
        "-H",
        "--histogram",
        type=is_filename,
        help="Output filename for histogram (no path, just filename)",
    )
    parser.add_argument(
        "-p",
        "--pareto",
        type=is_filename,
        help="Output filename for pareto front (no path, just filename)",
    )
    parser.add_argument(
        "-l",
        "--logging",
        choices=["debug", "info", "warning"],
        default="info",
        help="Dataset to use: mnist, mini-mnist, or vertebral",
    )

    args = parser.parse_args()
    return args
