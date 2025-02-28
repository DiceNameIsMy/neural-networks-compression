import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.result import Result
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from constants import SEED
from datasets.dataset import Dataset
from models.mlp import ModelParams, evaluate_model
from models.quantization import ActivationFunc
from nas.mlp_nas_problem import NASParams, NASProblem, get_cost_approximation


def get_hardware_cost_approximation(df: pd.DataFrame, dataset):
    data = list()
    for _, row in df.iterrows():
        data.append(
            get_cost_approximation(
                dataset.input_size,
                dataset.output_size,
                row["layers_amount"],
                row["hidden_height"],
                row["input_bitwidth"],
                row["hidden_bitwidth"],
            )
        )

    return data


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


def plot_pareto_front(accuracy, cost, pareto_front):
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

    accuracy_range = max(accuracy) - min(accuracy)
    plt.xlim(
        min(accuracy) - (accuracy_range * 0.1), max(accuracy) + (accuracy_range * 0.1)
    )

    plt.title("Pareto Front")
    plt.ylabel("Cost")
    plt.xlabel("Accuracy")
    plt.grid(True)
    plt.show()


def get_population_df(problem, res: Result, plot=True):
    # Create a dataframe of final population
    population_conf = problem._expand_X(res.X)
    df = pd.DataFrame(population_conf)

    accuracy = 100 - res.F[:, 0]
    df["accuracy"] = accuracy

    cost = get_hardware_cost_approximation(df, problem.dataset)
    df["cost"] = cost

    # Compute & Plot pareto front

    pareto_front = np.array(get_pareto_points(df["accuracy"], df["cost"]))

    if plot:
        plot_pareto_front(accuracy, cost, pareto_front)

    pareto_idxs = pareto_front[:, 2]
    pf = df.iloc[pareto_idxs.astype(int)]
    return {"all": df, "pareto": pf}


def train_pf(pf, DatasetClass, epochs):
    train_loader, test_loader = DatasetClass.get_dataloaders()

    def _train_conf(conf):
        params = ModelParams(
            input_size=DatasetClass.input_size,
            input_bitwidth=conf["input_bitwidth"],
            output_size=DatasetClass.output_size,
            hidden_size=conf["hidden_height"],
            hidden_bitwidth=conf["hidden_bitwidth"],
            model_layers=conf["layers_amount"],
            learning_rate=conf["learning_rate"],
            activation=ActivationFunc.BINARIZE,
            epochs=epochs,
            quantization_mode=conf["quantization_mode"],
        )
        perf = evaluate_model(params, train_loader, test_loader, times=5)
        conf["best"] = perf["max"]
        conf["mean"] = perf["mean"]
        conf["std"] = perf["std"]
        return conf

    return pf.apply(_train_conf, axis=1)


def run_NAS_pipeline(
    DatasetClass: Dataset, params: NASParams, n_gen=1, train_pf_epochs=10
):
    problem = NASProblem(DatasetClass, params)

    algorithm = NSGA2(
        pop_size=params.population_size,
        n_offsprings=params.population_offspring_count,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
        mutation=PM(eta=20, repair=RoundingRepair()),
    )

    termination = get_termination("n_gen", n_gen)
    res = minimize(
        problem, algorithm, termination, seed=SEED, save_history=False, verbose=True
    )

    problem.show_metadata()

    population = get_population_df(problem, res)

    population["pareto_trained"] = train_pf(
        population["pareto"], DatasetClass, train_pf_epochs
    )
    return population


def run_simple_grid_search(DatasetClass: Dataset):
    # A model trained without NAS
    # TODO: How to properly evaluate my baseline?

    no_NAS_datapoints = []
    for layers in range(2, 4 + 1):
        for layer_size in range(4, 17, 4):
            p = ModelParams(
                input_size=DatasetClass.input_size,
                input_bitwidth=8,
                output_size=DatasetClass.output_size,
                hidden_size=layer_size,
                hidden_bitwidth=8,
                model_layers=layers,
                learning_rate=0.01,
                activation=ActivationFunc.BINARIZE,
                epochs=20,
            )
            train_loader, test_loader = DatasetClass.get_dataloaders()

            accuracy = evaluate_model(p, train_loader, test_loader, times=3)
            cost = get_cost_approximation(
                p.input_size,
                p.output_size,
                p.model_layers,
                p.hidden_size,
                p.input_bitwidth,
                p.hidden_bitwidth,
            )
            no_NAS_datapoints.append([layers, layer_size, accuracy, cost])

    best_accuracy = max(no_NAS_datapoints, key=lambda x: x[2])

    print(
        f"Simple grid search results: Layers: {best_accuracy[0]}, Hidden layers size: {best_accuracy[1]} ",
        f"Accuracy: {best_accuracy[2]}, Cost: {best_accuracy[3]}.",
    )
