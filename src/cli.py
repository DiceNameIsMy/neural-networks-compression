import argparse
import re

SUPPORTED_DATASETS = [
    # MLP datasets
    "vertebral",
    "cardio",
    "breast-cancer",
    # CNN datasets
    "mnist",
    "mini-mnist",
    "cifar10",
    "mini-cifar10",
]


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
        help="Dataset to train on",
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
        help="Output filename for the resulting population (not path, just filename)",
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
        help="Output filename for histogram (not path, just filename)",
    )
    parser.add_argument(
        "-p",
        "--pareto",
        type=is_filename,
        help="Output filename for pareto front (not path, just filename)",
    )
    parser.add_argument(
        "-l",
        "--logging",
        choices=["debug", "info", "warning"],
        default="info",
        help="Dataset to use: mnist, mini-mnist, or vertebral",
    )

    parser.add_argument(
        "-P",
        "--population",
        type=is_positive_int,
        help="Population size for NAS (positive integer)",
    )
    parser.add_argument(
        "-O",  # Uppercase 'O' to avoid conflict with '-o' for output
        "--offspring",
        type=is_positive_int,
        help="Number of offspring per generation for NAS (positive integer)",
    )

    args = parser.parse_args()

    if args.population and args.offspring:
        if args.population <= args.offspring:
            parser.error(
                "Population size must be greater than the number of offspring."
            )
    elif not args.population and args.offspring:
        parser.error(
            "Population size must be specified if offspring count is provided."
        )

    return args
