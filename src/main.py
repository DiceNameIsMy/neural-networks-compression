import logging
import os
import sys

from src.cli import parse_args
from src.reporting import get_reporting_folder


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

    output_folder = get_reporting_folder(args.output)

    configure_logging(level, output_folder)

    logger = logging.getLogger(__name__)
    logger.info(f"Executing program: {' '.join(sys.argv)}")

    if args.mode == "nas":
        run_nas_mode(args, output_folder)
    elif args.mode == "export":
        export_model_mode(args)
    elif args.mode == "experiment1":
        from src.experiment1 import run_experiment1

        run_experiment1(
            output_folder=output_folder,
            evaluations=args.evaluations,
            epochs=args.epochs,
            plot=args.plot,
            dataset_size=args.size,
        )
    else:
        raise ValueError(f"Not implemented mode: {args.mode}.")


def configure_logging(level: int, output_folder: str):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        stream=sys.stderr,
    )

    os.makedirs(output_folder, exist_ok=True)
    log_filename = os.path.join(output_folder, "output.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    # Add file handler to root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def run_nas_mode(args, output_folder: str):
    # Importing pytorch takes some time. It's not always needed because of
    # --help or invalid arguments.
    #
    # To speed up execution in these cases, we import modules that use
    # them inside this function; after CLI args are parsed.

    from src.run import run_nas_pipeline

    run_nas_pipeline(
        dataset=args.dataset,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        evaluations_per_arch=args.evaluations,
        population_size=args.population,
        offspring_count=args.offspring,
        generations=args.generations,
        store_models=args.store_models,
        output_folder=output_folder,
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
