import logging

import pandas as pd
from pymoo.optimize import minimize

from src.constants import SEED
from src.datasets.dataset import CnnDataset, MlpDataset
from src.datasets.mnist_dataset import MiniMNISTDataset, MNISTDataset
from src.datasets.vertebral_dataset import VertebralDataset
from src.nas.cnn_nas_problem import CnnNasProblem
from src.nas.mlp_nas_problem import MlpNasProblem
from src.nas.nas_params import NasParams
from src.nas.plot import hist_accuracies, plot_pareto_front

logger = logging.getLogger(__name__)

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
    # TODO: Use batch_size
    # TODO: Parametrize other parameters too

    if dataset in ["mnist", "mini-mnist"]:
        CnnDatasetClass = cnn_datasets_mapping[dataset]
        nas_params = NasParams(
            epochs=epochs or 1,
            patience=5,
            amount_of_evaluations=1,
            population_size=10,
            population_offspring_count=4,
            algorithm_generations=generations,
            population_store_file=output_file
            or (CnnDatasetClass.__name__ + "_population.csv"),
        )
        problem = CnnNasProblem(nas_params, CnnDatasetClass)

    elif dataset in ["vertebral"]:
        MlpDatasetClass = mlp_datasets_mapping[dataset]
        nas_params = NasParams(
            epochs=epochs or 10,
            patience=5,
            amount_of_evaluations=1,
            population_size=20,
            population_offspring_count=8,
            algorithm_generations=generations,
            population_store_file=output_file
            or MlpDatasetClass.__name__ + "_population.csv",
        )
        problem = MlpNasProblem(nas_params, MlpDatasetClass)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logger.info("Starting NAS")
    df = run_nas(problem, nas_params)
    logger.info("NAS has finished")

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
