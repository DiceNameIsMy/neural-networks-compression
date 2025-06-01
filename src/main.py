import logging
import sys

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
        stream=sys.stderr,
    )

    if args.mode == "nas":
        run_nas_mode(args)
    elif args.mode == "export":
        export_model_mode(args)
    elif args.mode == "experiment1":
        from src.experiment1 import run_experiment1

        run_experiment1(
            output_folder=args.output,
            evaluations=args.evaluations,
            epochs=args.epochs,
            plot=args.plot,
            dataset_size=args.size,
        )
    else:
        raise ValueError(f"Not implemented mode: {args.mode}.")


def run_nas_mode(args):
    # Importing pytorch takes some time. It's not always needed because of
    # --help or invalid arguments.
    #
    # To speed up execution in these cases, we import modules that use
    # them inside this function; after CLI args are parsed.

    from src.run import run_nas_pipeline

    run_nas_pipeline(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        population_size=args.population,
        offspring_count=args.offspring,
        generations=args.generations,
        store_models=args.store_models,
        output_file=args.output,
        histogram=args.histogram,
        pareto=args.pareto,
    )


def export_model_mode(args):
    filename: str = args.filename
    dataset, accuracy, chromosome = filename.strip(".pth").strip(".pt").split("_")
    logging.info(
        f"Exporting model for dataset={dataset}, "
        f"accuracy={accuracy}, chromosome={chromosome}"
    )
    # TODO: Read the stored model
    # TODO: Instantiate a model from the chromosome & dataset.
    #       Probably use a proper NAS problem there a corresponding method is defined.
    # TODO: Export the model somehow?
    pass


if __name__ == "__main__":
    main()
