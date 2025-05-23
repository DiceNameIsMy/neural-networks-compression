import argparse
import logging
import re
import sys

import pandas as pd
from pymoo.optimize import minimize

from src.constants import SEED
from src.datasets.dataset import CnnDataset, MlpDataset
from src.datasets.mnist_dataset import MiniMNISTDataset, MNISTDataset
from src.datasets.vertebral_dataset import VertebralDataset
from src.nas.cnn_nas_problem import CnnNasProblem
from src.nas.mlp_nas_problem import MlpNasProblem
from src.nas.nas import NasParams
from src.nas.plot import hist_accuracies, plot_pareto_front

mlp_datasets_mapping: dict[str, type[MlpDataset]] = {
    "vertebral": VertebralDataset,
}

cnn_datasets_mapping: dict[str, type[CnnDataset]] = {
    "mnist": MNISTDataset,
    "mini-mnist": MiniMNISTDataset,
}


def run_nas_cli(
    dataset: str,
    epochs: int | None,
    batch_size: int | None,
    output_file: str | None,
    generations: int,
    histogram: str | None,
    pareto: str | None,
):
    # TODO: batch_size

    cnn_problem: CnnNasProblem | MlpNasProblem

    if dataset in ["mnist", "mini-mnist"]:
        CnnDatasetClass = cnn_datasets_mapping[dataset]
        cnn_nas_params = NasParams(
            epochs=epochs or 1,
            patience=5,
            amount_of_evaluations=1,
            population_size=10,
            population_offspring_count=4,
            algorithm_generations=generations,
            population_store_file=output_file
            or (CnnDatasetClass.__name__ + "_population.csv"),
        )
        cnn_problem = CnnNasProblem(cnn_nas_params, CnnDatasetClass)
        df = run_nas(cnn_problem, cnn_nas_params)

    elif dataset in ["vertebral"]:
        MlpDatasetClass = mlp_datasets_mapping[dataset]
        mlp_params = NasParams(
            epochs=epochs or 10,
            patience=5,
            amount_of_evaluations=1,
            population_size=20,
            population_offspring_count=8,
            algorithm_generations=generations,
            population_store_file=output_file
            or MlpDatasetClass.__name__ + "_population.csv",
        )
        mlp_problem = MlpNasProblem(mlp_params, MlpDatasetClass)
        df = run_nas(mlp_problem, mlp_params)

    else:
        print(f"Unknown dataset: {dataset}", file=sys.stderr)
        sys.exit(1)

    if histogram is not None:
        hist_fig = hist_accuracies(df["Accuracy"])
        hist_fig.savefig(histogram)

    if pareto is not None:
        pareto_fig = plot_pareto_front(df["Accuracy"], df["Complexity"])
        pareto_fig.savefig(pareto)

    print(df.to_string(index=False, columns=["Accuracy", "Complexity", "Chromosome"]))


def run_nas(
    problem: CnnNasProblem | MlpNasProblem, nas_params: NasParams
) -> pd.DataFrame:
    algorithm = nas_params.get_algorithm()
    termination = nas_params.get_termination()

    res = minimize(problem, algorithm, verbose=True, seed=SEED, termination=termination)

    if nas_params.population_store_file is not None:
        nas_params.store_population(res, nas_params.population_store_file)

    df = problem.result_as_df(res)
    return df


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
        format="%(levelname)s: %(message)s",
        force=True,
    )

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
