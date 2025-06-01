import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def get_pareto_points(x, y):
    indexes = np.arange(0, len(x))
    pareto_front = []

    # Sort by higest accuracy first. If points have same accuracy, put the higher cost first
    for point in sorted(
        np.column_stack(((x, y, indexes))), key=lambda v: (-v[0], -v[1])
    ):
        if len(pareto_front) == 0:
            pareto_front.append(point)
            continue

        last = pareto_front[-1]
        if point[1] < last[1]:
            pareto_front.append(point)

    return pareto_front


def plot_pareto_front(accuracy: pd.Series, cost: pd.Series):
    pareto_front = np.array(get_pareto_points(accuracy, cost))

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(accuracy, cost, facecolors="none", edgecolors="blue")
    ax.plot(
        pareto_front[:, 0],
        pareto_front[:, 1],
        color="red",
        label="Pareto Frontier",
        linewidth=2,
    )

    # Beautify the plot
    max_cost = max(cost)
    ax.set_ylim(-(max_cost * 0.1), max_cost * 1.1)
    ax.ticklabel_format(style="plain", axis="y")

    accuracy_range = max(accuracy) - min(accuracy)
    ax.set_xlim(
        max(accuracy) + (accuracy_range * 0.1), min(accuracy) - (accuracy_range * 0.1)
    )

    ax.set_title("Pareto Front")
    ax.set_ylabel("Cost")
    ax.set_xlabel("Accuracy")
    ax.grid(True)
    return fig


def hist_accuracies(accuracies: pd.Series, bins=20) -> Figure:
    fig, ax = plt.subplots()
    ax.hist(accuracies, bins=bins, range=(0, 100), edgecolor="black")
    ax.set_title("Histogram of Accuracies")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Frequency")
    return fig
