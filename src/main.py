import logging

from src.cli import parse_args


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
    # To speed up execution in these cases, we import modules that use them here.

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
