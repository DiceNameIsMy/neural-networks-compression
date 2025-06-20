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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural Architecture Search CLI for quantized neural networks.",
        epilog="Example: %(prog)s nas -d vertebral -e 20 -g 10 -p 20 -o 12 -s",
    )

    parser.add_argument(
        "-l",
        "--logging",
        choices=["debug", "info", "warning"],
        default="info",
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

    parser_experiment1 = subparsers.add_parser(
        "experiment1",
        help="Run experiment No. 1",
    )
    configure_experiment1_parser(parser_experiment1)

    subparsers.add_parser(
        "experiment2",
        help="Run experiment No. 2 (TODO)",
    )
    subparsers.add_parser(
        "experiment3",
        help="Run experiment No. 3 (TODO)",
    )

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
        "--patience",
        type=is_positive_int,
        default=5,
        help="Early stopping patience (positive integer, default: 5)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=is_positive_int,
        required=True,
        help="Batch size for training (positive integer)",
    )
    parser.add_argument(
        "--evaluations",
        type=is_positive_int,
        default=1,
        help="Number of evaluations per architecture (positive integer, default: 1)",
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
        help="Output folder for the resulting population",
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
        action="store_true",
        default=False,
        help="Plot a histogram of accuracies",
    )
    parser.add_argument(
        "-P",
        "--pareto",
        action="store_true",
        default=False,
        help="Plot a pareto front",
    )


def configure_export_model_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Filename of the model to export",
    )


def configure_experiment1_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-O",
        "--output",
        help="Folder to store outputs (optional). If not specified, uses ./reports/<iso_timestamp>/",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=is_positive_int,
        required=True,
        help="Number of epochs to run per model training (positive integer)",
    )
    parser.add_argument(
        "--patience",
        type=is_positive_int,
        required=True,
        help="Early stopping patience (positive integer, default: 5)",
    )
    parser.add_argument(
        "--evaluations",
        type=is_positive_int,
        required=True,
        help="Number of evaluations for accuracy evaluation (positive integer)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate plots for the experiment results (default: False)",
    )
    parser.add_argument(
        "--size",
        choices=["full", "mini"],
        default="mini",
        help="Dataset size to use for the experiment. Default is 'mini'",
    )
