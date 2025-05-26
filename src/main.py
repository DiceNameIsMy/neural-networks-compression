import logging

from src.cli import SUPPORTED_DATASETS, parse_args


def main():
    args = parse_args()

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

    # Importing pytorch takes some time. It's not always needed because of
    # --help or invalid arguments.
    #
    # To speed up execution in these cases, we import modules that use
    # them inside the main function; after CLI args are parsed.

    ensure_cli_datasets_are_supported()

    from src.run import run_nas_pipeline

    run_nas_pipeline(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_file=args.output,
        generations=args.generations,
        histogram=args.histogram,
        pareto=args.pareto,
    )


def ensure_cli_datasets_are_supported():
    """
    Ensure that if a CLI supports a dataset, it really does so.
    """

    from src.run import CNN_DATASETS_MAPPING, MLP_DATASETS_MAPPING

    mappable_datasets = set(CNN_DATASETS_MAPPING.keys()).union(
        MLP_DATASETS_MAPPING.keys()
    )

    for dataset in SUPPORTED_DATASETS:
        if dataset not in mappable_datasets:
            raise ValueError(
                f"Dataset '{dataset}' is supported but not present in any mapping."
            )


if __name__ == "__main__":
    main()
