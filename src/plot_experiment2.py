import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.express as px

from src.datasets.dataset import CnnDataset, MlpDataset
from src.models.eval import NNArchitectureComplexityEvaluator
from src.nas.chromosome import ChromosomeConfig
from src.nas.cnn_chromosome import CNNChromosome
from src.nas.cnn_nas_problem import CnnNasProblem
from src.nas.mlp_chromosome import MLPChromosome
from src.nas.mlp_nas_problem import MlpNasProblem
from src.nas.nas_params import NasParams


def plot_mlp_report(
    folder: str, DatasetCls: type[MlpDataset], store: bool = True, **kwargs
):
    df = mlp_report_to_df(folder, DatasetCls)
    fig = scatter_population(df, **kwargs)

    if store:
        fig.write_image(os.path.join(folder, "population.pdf"), format="pdf")

    return fig


def plot_cnn_report(
    folder: str, DatasetCls: type[CnnDataset], store: bool = True, **kwargs
):
    df = cnn_report_to_df(folder, DatasetCls)
    fig = scatter_population(df, symbol="conv_compression", **kwargs)

    if store:
        fig.write_image(os.path.join(folder, "population.pdf"), format="pdf")

    return fig


def mlp_report_to_df(folder: str, DatasetCls: type[MlpDataset]):
    population = NasParams.load_population(os.path.join(folder, "population.csv"))
    if population is None:
        raise ValueError("Population is empty or not found.")

    cfg = ChromosomeConfig(MLPChromosome)
    nas_params = NasParams(batch_size=32)
    nas_problem = MlpNasProblem(nas_params, DatasetCls)

    data = []

    for raw_ch in population:
        ch = cfg.decode(raw_ch)
        acc = find_acc(raw_ch, folder)
        cost = NNArchitectureComplexityEvaluator(
            nas_problem.get_nn_params(ch)
        ).evaluate_complexity()

        ch_dict = asdict(ch)
        ch_dict["activation"] = ch.activation.name
        ch_dict["compression"] = ch.compression.name

        data.append(
            {
                "acc": acc,
                "cost": cost,
                **ch_dict,
            }
        )

    df = pd.DataFrame(data)
    return df


def cnn_report_to_df(folder: str, DatasetCls: type[CnnDataset]):
    population = NasParams.load_population(os.path.join(folder, "population.csv"))
    if population is None:
        raise ValueError("Population is empty or not found.")

    cfg = ChromosomeConfig(CNNChromosome)
    nas_params = NasParams(batch_size=32)
    nas_problem = CnnNasProblem(nas_params, DatasetCls)

    data = []

    for raw_ch in population:
        ch = cfg.decode(raw_ch)
        acc = find_acc(raw_ch, folder)
        cost = NNArchitectureComplexityEvaluator(
            nas_problem.get_nn_params(ch)
        ).evaluate_complexity()

        ch_dict = asdict(ch)
        ch_dict["activation"] = ch.activation.name
        ch_dict["conv_compression"] = ch.conv_compression.name
        ch_dict["fc_compression"] = ch.fc_compression.name

        data.append({"acc": acc, "cost": cost, **ch_dict, "chromosome": ch})

    df = pd.DataFrame(data)
    return df


def find_acc(chromosome: np.ndarray, report_folder: str) -> float:
    models = os.listdir(os.path.join(report_folder, "models"))
    str_ch = "-".join([str(x) for x in chromosome])

    for model in models:
        if str_ch in model:
            return float(model.split("_")[0])

    raise ValueError(f"Didn't find a model for chromosome: {str_ch}")


def scatter_population(
    df: pd.DataFrame,
    *args,
    color="activation",
    symbol="compression",
    **kwargs,
):
    fig = px.scatter(
        df,
        x="acc",
        y="cost",
        color=color,
        symbol=symbol,
        labels={
            "compression": "Compression",
            "conv_compression": "Conv layers compression",
            "fc_compression": "FC layers compression",
            "activation": "Activation",
            "acc": "Accuracy (%)",
            "cost": "Complexity",
        },
        template="plotly_white",
        **kwargs,
    )
    fig.update_xaxes(autorange="reversed")
    fig.update_layout(
        legend={
            "x": 0.99,
            "y": 0.99,
            "xanchor": "right",
            "yanchor": "top",
            "bgcolor": "rgba(240, 240, 240, 1)",
            "bordercolor": "rgba(240, 240, 240, 1)",
            "borderwidth": 3,
        },
    )
    # fig.update(**style_dict)

    return fig
