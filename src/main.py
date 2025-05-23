import argparse
import logging
import re


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


def main():
    parser = argparse.ArgumentParser(
        description="Neural Architecture Search CLI for quantized neural networks."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["mnist", "mini-mnist", "vertebral"],
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

    if args.logging == "debug":
        level = logging.DEBUG
    elif args.logging == "info":
        level = logging.INFO
    elif args.logging == "warning":
        level = logging.WARNING
    else:
        raise ValueError(f"Unknown logging level: {args.logging}")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    from src.run import run_nas_cli

    run_nas_cli(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_file=args.output,
        generations=args.generations,
        histogram=args.histogram,
        pareto=args.pareto,
    )


if __name__ == "__main__":
    main()
