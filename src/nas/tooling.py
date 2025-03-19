import os
from dataclasses import dataclass

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
from models.quant.enums import ActivationModule
from nas.mlp_nas_problem import NASParams, NASProblem, get_cost_approximation


def get_hardware_cost_approximation(df: pd.DataFrame, dataset: Dataset):
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


def get_population_df(problem: NASProblem, res: Result, plot=True):
    # Create a dataframe of final population
    population_conf = problem._expand_X(res.X)
    all_offs = pd.DataFrame(population_conf)

    accuracy = 100 - res.F[:, 0]
    all_offs["accuracy"] = accuracy

    cost = get_hardware_cost_approximation(all_offs, problem.dataset)
    all_offs["cost"] = cost

    # Compute & Plot pareto front
    pareto_front = np.array(get_pareto_points(all_offs["accuracy"], all_offs["cost"]))

    if plot:
        plot_pareto_front(accuracy, cost, pareto_front)

    pareto_idxs = pareto_front[:, 2]
    pareto_front = all_offs.iloc[pareto_idxs.astype(int)]
    return (all_offs, pareto_front)


def train_pf(pf, DatasetClass, epochs):
    train_loader, test_loader = DatasetClass.get_dataloaders()

    def _train_conf(conf):
        params = ModelParams(
            in_layer_height=DatasetClass.input_size,
            in_bitwidth=conf["input_bitwidth"],
            out_height=DatasetClass.output_size,
            hidden_height=conf["hidden_height"],
            hidden_bitwidth=conf["hidden_bitwidth"],
            model_layers=conf["layers_amount"],
            learning_rate=conf["learning_rate"],
            activation=ActivationModule.BINARIZE,
            epochs=epochs,
            quantization_mode=conf["quantization_mode"],
        )
        perf = evaluate_model(params, train_loader, test_loader, times=5)
        conf["best"] = perf["max"]
        conf["mean"] = perf["mean"]
        conf["std"] = perf["std"]
        return conf

    return pf.apply(_train_conf, axis=1)


@dataclass
class NASResult:
    res: Result
    all: pd.DataFrame
    pareto: pd.DataFrame
    pareto_trained: pd.DataFrame

    def get_pf_population(self):
        return np.array([e.X for e in self.res.history[-1].off])

    def store_pf_population(self, path: str):
        pop = self.get_pf_population()
        pd.DataFrame(pop).to_csv(path, index=False)

    @classmethod
    def load_population(cls, path: str | None):
        if path is None:
            return None
        if not os.path.exists(path):
            return None

        return pd.read_csv(path).values


def run_NAS_pipeline(
    DatasetClass: Dataset,
    params: NASParams,
    n_gen=1,
    train_pf_epochs=10,
    population_cache_file: str | None = None,
) -> NASResult:

    # Run NAS
    problem = NASProblem(DatasetClass, params)

    sampling = NASResult.load_population(population_cache_file)
    if sampling is None:
        sampling = IntegerRandomSampling()

    algorithm = NSGA2(
        pop_size=params.population_size,
        n_offsprings=params.population_offspring_count,
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
        mutation=PM(eta=20, repair=RoundingRepair()),
    )
    termination = get_termination("n_gen", n_gen)
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=SEED,
        save_history=True,
        verbose=True,
    )

    problem.show_metadata()

    # Prepare df's for interpretation
    all_offs, pareto_offs = get_population_df(problem, res)

    trained_pf = train_pf(pareto_offs, DatasetClass, train_pf_epochs)

    return NASResult(res, all_offs, pareto_offs, trained_pf)


def run_simple_grid_search(DatasetClass: Dataset):
    # A model trained without NAS
    # TODO: How to properly evaluate my baseline?

    no_NAS_datapoints = []
    for layers in range(2, 4 + 1):
        for layer_size in range(4, 17, 4):
            p = ModelParams(
                in_layer_height=DatasetClass.input_size,
                in_bitwidth=8,
                out_height=DatasetClass.output_size,
                hidden_height=layer_size,
                hidden_bitwidth=8,
                model_layers=layers,
                learning_rate=0.01,
                activation=ActivationModule.BINARIZE,
                epochs=20,
            )
            train_loader, test_loader = DatasetClass.get_dataloaders()

            accuracy = evaluate_model(p, train_loader, test_loader, times=3)
            cost = get_cost_approximation(
                p.in_layer_height,
                p.out_height,
                p.model_layers,
                p.hidden_height,
                p.in_bitwidth,
                p.hidden_bitwidth,
            )
            no_NAS_datapoints.append([layers, layer_size, accuracy, cost])

    best_accuracy = max(no_NAS_datapoints, key=lambda x: x[2])

    print(
        f"Simple grid search results: Layers: {best_accuracy[0]}, Hidden layers size: {best_accuracy[1]} ",
        f"Accuracy: {best_accuracy[2]}, Cost: {best_accuracy[3]}.",
    )
