from dataclasses import dataclass

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from constants import EPOCHS
from datasets.vertebral_dataset import VertebralDataset
from models.mlp import ModelParams, evaluate_model
from models.quantization import ActivationFunc, QMode
from src.datasets.dataset import Dataset

BITWIDTHS_MAPPING = (2, 3, 4, 5, 7, 10, 14, 32)
LEARNING_RATES_MAPPING = (
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
)


@dataclass
class NASParams:
    hidden_height_bounds: tuple = (4, 16)
    input_bitwidth_bounds: tuple = (0, len(BITWIDTHS_MAPPING) - 1)
    hidden_bitwidth_bounds: tuple = (0, len(BITWIDTHS_MAPPING) - 1)
    layers_amount_bounds: tuple = (2, 8)
    learning_rate_bounds: tuple = (0, len(LEARNING_RATES_MAPPING) - 1)
    quantization_mode_bounds: tuple = (0, 1)

    epochs: int = EPOCHS
    min_accuracy: float = 50.0

    amount_of_evaluations: int = 1
    population_size: int = 30
    population_offspring_count: int = 10

    def get_n_var(self):
        return 6

    def get_xl(self):
        return np.array(np.stack(self._as_array(), 1)[0])

    def get_xu(self):
        return np.array(np.stack(self._as_array(), 1)[1])

    def _as_array(self):
        return np.array(
            [
                self.layers_amount_bounds,
                self.hidden_height_bounds,
                self.input_bitwidth_bounds,
                self.hidden_bitwidth_bounds,
                self.learning_rate_bounds,
                self.quantization_mode_bounds,
            ]
        )


def get_mult_approximation(
    input_size, output_size, layers_amount, hidden_height
) -> tuple[int, int, int]:
    hidden_layers = max(layers_amount - 2, 0)

    if hidden_layers == 0:
        return input_size * output_size, 0, 0

    elif hidden_layers == 1:
        return input_size * hidden_height, 0, hidden_height * output_size

    else:
        hidden_layers_mul = hidden_height * hidden_height * (hidden_layers - 1)
        return (
            input_size * hidden_height,
            hidden_layers_mul,
            hidden_height * output_size,
        )


def get_cost_approximation(
    input_size,
    output_size,
    layers_amount,
    hidden_height,
    input_bitwidth,
    hidden_bitwidth,
):
    first_second, hidden_hidden, before_last_last = get_mult_approximation(
        input_size, output_size, layers_amount, hidden_height
    )

    return (
        (input_bitwidth * first_second)
        + (hidden_hidden * hidden_bitwidth)
        + (before_last_last * hidden_bitwidth)
    )


class NASProblem(ElementwiseProblem):
    params: NASParams
    dataset: Dataset
    train_loader = None
    test_loader = None

    best_accuracy = 0

    def __init__(self, dataset, params: NASParams):
        self.params = params
        self.dataset = dataset
        self.train_loader, self.test_loader = self.dataset.get_dataloaders()

        super().__init__(
            n_var=params.get_n_var(),
            n_obj=5,
            n_ieq_constr=1,
            xl=params.get_xl(),
            xu=params.get_xu(),
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        conf = self._expand_x(x)

        model_params = self.conf_to_model_params(conf)
        model_perf = evaluate_model(
            model_params,
            self.train_loader,
            self.test_loader,
            times=self.params.amount_of_evaluations,
        )
        accuracy = model_perf["mean"]

        self.best_accuracy = max(self.best_accuracy, accuracy)

        # TODO: We could choose a better metric than just mean accuracy.
        # For example compute whether model A is statistically better than model B.
        # NSGA-2 uses a tournament selection, where this metric can be used.

        # Objectives are optimized to get closer to zero
        # Objective: Higher model accuracy
        f1 = 100 - accuracy

        # Objective: Less hidden layers
        f2 = self.val_to_f(conf["layers_amount"], self.params.layers_amount_bounds)

        # Objective: Less perceptrons per layer
        f3 = self.val_to_f(conf["hidden_height"], self.params.hidden_height_bounds)

        # Objective: Higher hidden layer quantization
        f4 = self.val_to_f(
            conf["hidden_bitwidth"], (min(BITWIDTHS_MAPPING), max(BITWIDTHS_MAPPING))
        )

        # Objective: Higher inputs quantization
        f5 = self.val_to_f(
            conf["input_bitwidth"], (min(BITWIDTHS_MAPPING), max(BITWIDTHS_MAPPING))
        )
        out["F"] = [f1, f2, f3, f4, f5]

        # Constraints are optimized to be less than zero
        # Constraint: Accuracy must be higher than .min_accuracy
        g1 = -(accuracy - self.params.min_accuracy)
        out["G"] = [g1]

    def _expand_x(self, x):
        return {
            "layers_amount": x[0].astype(int),
            "hidden_height": x[1].astype(int),
            "input_bitwidth": BITWIDTHS_MAPPING[x[2].astype(int)],
            "hidden_bitwidth": BITWIDTHS_MAPPING[x[3].astype(int)],
            "learning_rate": LEARNING_RATES_MAPPING[x[4].astype(int)],
            "quantization_mode": QMode.DET if x[5].astype(int) == 0 else QMode.STOCH,
        }

    def _expand_X(self, X: np.ndarray):
        return {
            "layers_amount": X[:, 0].astype(int),
            "hidden_height": X[:, 1].astype(int),
            "input_bitwidth": np.array(
                [BITWIDTHS_MAPPING[x] for x in X[:, 2].astype(int)]
            ),
            "hidden_bitwidth": np.array(
                [BITWIDTHS_MAPPING[x] for x in X[:, 3].astype(int)]
            ),
            "learning_rate": np.array(
                [LEARNING_RATES_MAPPING[x] for x in X[:, 4].astype(int)]
            ),
            "quantization_mode": np.where(
                X[:, 5].astype(int) == 1, QMode.DET, QMode.STOCH
            ),
        }

    def val_to_f(self, val, bounds):
        return (float(val) - bounds[0]) * 100 / (bounds[1] - bounds[0])

    def conf_to_model_params(self, conf):
        return ModelParams(
            in_layer_height=self.dataset.input_size,
            in_bitwidth=conf["input_bitwidth"],
            out_height=self.dataset.output_size,
            hidden_height=conf["hidden_height"],
            hidden_bitwidth=conf["hidden_bitwidth"],
            model_layers=conf["layers_amount"],
            quantization_mode=conf["quantization_mode"],
            learning_rate=conf["learning_rate"],
            activation=ActivationFunc.BINARIZE,
            epochs=self.params.epochs,
        )

    def show_metadata(self):
        print(f"Best accuracy: {self.best_accuracy}")


def test_objectives():
    nas_params = NASParams()
    p = NASProblem(VertebralDataset, nas_params)
    print(p.conf_to_model_params(p._expand_x(nas_params.get_xl())))
    print(p.conf_to_model_params(p._expand_x(nas_params.get_xu())))

    min_out = dict()
    p_min = nas_params.get_xl()
    print(p_min)
    p._evaluate(p_min, min_out)
    print("Objective values on min bounds: ", min_out["F"])
    print("NAS params for min bounds: ", p._expand_x(p_min))

    max_out = dict()
    p_max = nas_params.get_xu()
    print(p_max)
    p._evaluate(p_max, max_out)
    print("Objective values on max bounds: ", max_out["F"])
    print("NAS params for max bounds: ", p._expand_x(p_max))


test_objectives()
