import argparse

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
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")

        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")


def is_filename(value):
    # Only allow filenames, not paths
    is_filename = "/" in value or "\\" in value
    if not is_filename:
        raise argparse.ArgumentTypeError(f"{value} must be a filename, not a path")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural Architecture Search CLI for quantized neural networks.",
        epilog="Example: %(prog)s nas -d vertebral -e 20 -g 10 -p 20 -o 12 -s",
    )

    subparsers = parser.add_subparsers(
        title="modes", dest="mode", help="Select a mode to run", required=True
    )

    parser_run_nas = subparsers.add_parser(
        "nas",
        help="Run NAS pipeline to train models.",
    )
    configure_nas_mode_parser(parser_run_nas)

    parser_export_model = subparsers.add_parser(
        "export",
        help="Export a trained model. (TODO)",
    )
    configure_export_model_parser(parser_export_model)

    args = parser.parse_args()

    # Extra validation
    if args.mode == "nas":
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


def configure_nas_mode_parser(parser: argparse.ArgumentParser):
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
        "-g",
        "--generations",
        type=is_positive_int,
        required=True,
        help="Number of NAS generations (positive integer)",
    )
    parser.add_argument(
        "-p",
        "--population",
        type=is_positive_int,
        help="Population size for NAS (positive integer)",
    )
    parser.add_argument(
        "-o",
        "--offspring",
        type=is_positive_int,
        help="Number of offspring per generation for NAS (positive integer)",
    )
    parser.add_argument(
        "-O",
        "--output",
        type=is_filename,
        help="Output filename for the resulting population (not path, just filename)",
    )
    parser.add_argument(
        "-s",
        "--store-models",
        action="store_true",
        help="Store best models resulting from NAS (default: False)",
    )
    parser.add_argument(
        "-H",
        "--histogram",
        type=is_filename,
        help="Output filename for histogram (not path, just filename)",
    )
    parser.add_argument(
        "-P",
        "--pareto",
        type=is_filename,
        help="Output filename for pareto front (not path, just filename)",
    )
    parser.add_argument(
        "-l",
        "--logging",
        choices=["debug", "info", "warning"],
        default="info",
    )


def configure_export_model_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-f",
        "--filename",
        type=is_filename,
        required=True,
        help="Filename of the model to export (not path, just filename)",
    )
    parser.add_argument(
        "-l",
        "--logging",
        choices=["debug", "info", "warning"],
        default="info",
    )
