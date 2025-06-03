import os

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


def get_pareto_indicies(df: pd.DataFrame, acc_col: str, cost_col: str) -> list[int]:
    sorted_df = df.sort_values([acc_col, cost_col], ascending=[False, True])
    pareto_indices = []

    for i, this in sorted_df.iterrows():
        if len(pareto_indices) == 0:
            pareto_indices.append(i)
            continue

        last_idx = pareto_indices[-1]
        last = df.iloc[last_idx]

        if this[cost_col] < last[cost_col]:
            pareto_indices.append(i)
            continue

    return pareto_indices


def scatter_population(df: pd.DataFrame, *args, title="Title", x="best", **kwargs):
    fig = px.scatter(
        df,
        x=x,
        y="cost",
        color="conv_activation",
        symbol="conv_compression",
        title=title,
        labels={
            "conv_compression": "Compression",
            "conv_activation": "Activation",
            "best": "Accuracy (%)",
            "cost": "Complexity",
        },
        template="plotly_white",
        **kwargs,
    )
    fig.update_xaxes(autorange="reversed")
    fig.update_layout(margin=dict(autoexpand=True))

    return fig


def make_plots(
    df: pd.DataFrame, title: str, x: str = "best", **kwargs
) -> tuple[go.Figure, go.Figure]:
    models = scatter_population(
        df, title="Models for " + title, height=520, x=x, **kwargs
    )

    pareto_indices = get_pareto_indicies(df, x, "cost")
    pareto = scatter_population(
        df.loc[pareto_indices],
        title="Pareto front of models for " + title,
        x=x,
        **kwargs,
    )

    return models, pareto


def make_plots_for_results(
    title: str, folder: str, csv_file: str, store: bool = True, *args, **kwargs
) -> tuple[go.Figure, go.Figure]:
    population_file = os.path.join(folder, csv_file)
    df = pd.read_csv(population_file)

    models, pareto = make_plots(df, title=title, **kwargs)

    if store:
        models.write_image(os.path.join(folder, "population.pdf"), format="pdf")
        pareto.write_image(os.path.join(folder, "pareto.pdf"), format="pdf")

    return models, pareto
