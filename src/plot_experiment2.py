import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.express as px

from src.datasets.vertebral_dataset import VertebralDataset
from src.models.eval import NNArchitectureComplexityEvaluator
from src.nas.chromosome import ChromosomeConfig
from src.nas.mlp_chromosome import MLPChromosome
from src.nas.mlp_nas_problem import MlpNasProblem
from src.nas.nas_params import NasParams


def plot_report(folder: str, title: str, store: bool = True):
    df = report_to_df(folder)
    fig = scatter_population(df, title=title)

    if store:
        fig.write_image(os.path.join(folder, "population.png"), format="png")

    return fig


def report_to_df(folder: str):
    population = NasParams.load_population(os.path.join(folder, "population.csv"))
    if population is None:
        raise ValueError("Population is empty or not found.")

    cfg = ChromosomeConfig(MLPChromosome)
    nas_params = NasParams(batch_size=32)
    nas_problem = MlpNasProblem(nas_params, VertebralDataset)

    data = []

    for raw_ch in population:
        ch = cfg.decode(raw_ch)
        acc = find_acc(raw_ch, folder)
        cost = NNArchitectureComplexityEvaluator(
            nas_problem.get_nn_params(ch)
        ).evaluate_complexity()

        data.append(
            {
                "acc": acc,
                "cost": cost,
                **asdict(ch),
            }
        )

    df = pd.DataFrame(data)
    return df


def find_acc(chromosome: np.ndarray, report_folder: str) -> float:
    models = os.listdir(os.path.join(report_folder, "models"))
    str_ch = "-".join([str(x) for x in chromosome])

    for model in models:
        if str_ch in model:
            return float(model.split("_")[0])

    raise ValueError(f"Didn't find a model for chromosome: {str_ch}")


def scatter_population(df: pd.DataFrame, *args, title="Title", **kwargs):
    fig = px.scatter(
        df,
        x="acc",
        y="cost",
        color="activation",
        symbol="compression",
        title=title,
        labels={
            "compression": "Compression",
            "activation": "Activation",
            "acc": "Accuracy (%)",
            "cost": "Model Complexity",
        },
        template="plotly_white",
        **kwargs,
    )
    fig.update_xaxes(autorange="reversed")
    fig.update_layout(margin=dict(autoexpand=True))

    return fig
