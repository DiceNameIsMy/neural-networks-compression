import matplotlib.pyplot as plt
import numpy as np


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
        if point[1] <= last[1]:
            pareto_front.append(point)

    return pareto_front


def plot_pareto_front(accuracy, cost):
    pareto_front = np.array(get_pareto_points(accuracy, cost))

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(accuracy, cost, facecolors="none", edgecolors="blue")
    plt.plot(
        pareto_front[:, 0],
        pareto_front[:, 1],
        color="red",
        label="Pareto Frontier",
        linewidth=2,
    )

    # Beautify the plot
    max_cost = max(cost)
    plt.ylim(max_cost * 1.1, -(max_cost * 0.1))
    plt.ticklabel_format(style="plain", axis="y")

    accuracy_range = max(accuracy) - min(accuracy)
    plt.xlim(
        min(accuracy) - (accuracy_range * 0.1), max(accuracy) + (accuracy_range * 0.1)
    )

    plt.title("Pareto Front")
    plt.ylabel("Cost")
    plt.xlabel("Accuracy")
    plt.grid(True)
    plt.show()


def hist_accuracies(accuracies: list[float], bins=20) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.hist(accuracies, bins=bins, range=(0, 100), edgecolor="black")
    ax.set_title("Histogram of Accuracies")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Frequency")
    return fig
