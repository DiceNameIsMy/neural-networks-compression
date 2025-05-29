import logging
import math
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result
from torch.utils import data

from src.datasets.dataset import MlpDataset
from src.models.compression.enums import Activation, NNParamsCompMode
from src.models.eval import KFoldNNArchitectureEvaluator
from src.models.mlp import MLP, FCLayerParams, FCParams, MLPParams
from src.models.nn import ActivationParams, NNTrainParams
from src.nas.mlp_chromosome import MLPChromosome, RawMLPChromosome
from src.nas.nas_params import NasParams

logger = logging.getLogger(__name__)


class MlpNasProblem(ElementwiseProblem):
    # TODO: I'm not getting many diverse solutions because
    #       I only have 2 objectives -> there isn't space for
    #       many distinct solutions. It that okay?

    # TODO: It it better to use best accuracy instead of mean
    #       accuracy as an optimization goal? Or perhaps I should
    #       keep using both, but then train the final population
    #       again & show best accuracy?

    # NOTE: On every evaluation a new set of dataloaders is created.
    #       It's reduntant, although only a small % of NAS is spent on that.

    p: NasParams
    DatasetCls: type[MlpDataset]
    train_loader: data.DataLoader
    test_loader: data.DataLoader

    best_models: dict[tuple[int], tuple[float, MLP]]

    def __init__(self, params: NasParams, DatasetCls: type[MlpDataset]):
        x_low, x_high = RawMLPChromosome.get_bounds()
        super().__init__(
            n_var=RawMLPChromosome.get_size(), n_obj=2, xl=x_low, xu=x_high + 0.99
        )  # Part of a workaround to the rounding problem

        self.p = params
        self.DatasetCls = DatasetCls
        self.train_loader, self.test_loader = self.DatasetCls.get_dataloaders(
            self.p.batch_size
        )
        self.best_models = {}

    def _evaluate(self, x, out, *args, **kwargs):
        ch = RawMLPChromosome(x).parse()
        params = self.get_nn_params(ch)
        logger.debug(f"Evaluating {params}")

        stats = KFoldNNArchitectureEvaluator(params).evaluate(
            times=self.p.amount_of_evaluations
        )
        self.try_store_model(x, stats["max"], stats["best_model"])

        # Maximize accuracy
        f1 = -self.normalize(stats["max"], 0, 100)

        # Minimize NN complexity
        complexity = self.compute_nn_complexity(params)
        f2 = self.normalize(
            complexity, self.get_min_complexity(), self.get_max_complexity()
        )

        out["F"] = [f1, f2]

    def try_store_model(self, x: np.ndarray, accuracy: float, model: MLP):
        if len(self.best_models) < self.p.population_size:
            # Store the best model for this chromosome
            self.best_models[tuple(x)] = (accuracy, model)
        else:
            # Replace the worst model if a new one is better
            worst_key = min(self.best_models, key=lambda k: self.best_models[k][0])
            worst_key_accuracy = self.best_models[worst_key][0]

            if accuracy > worst_key_accuracy:
                self.best_models.pop(worst_key)
                self.best_models[tuple(x)] = (accuracy, model)

    def get_nn_params(self, ch: MLPChromosome) -> MLPParams:
        layers = []
        layers.append(
            FCLayerParams(
                self.DatasetCls.input_size, NNParamsCompMode.NBITS, ch.in_bitwidth
            )
        )
        if ch.hidden_layers >= 1:
            layers.append(
                FCLayerParams(
                    ch.hidden_height1, NNParamsCompMode.NBITS, ch.hidden_bitwidth1
                )
            )
        if ch.hidden_layers >= 2:
            layers.append(
                FCLayerParams(
                    ch.hidden_height2, NNParamsCompMode.NBITS, ch.hidden_bitwidth2
                )
            )
        if ch.hidden_layers >= 3:
            layers.append(
                FCLayerParams(
                    ch.hidden_height3, NNParamsCompMode.NBITS, ch.hidden_bitwidth3
                )
            )
        layers.append(
            FCLayerParams(self.DatasetCls.output_size, NNParamsCompMode.NONE, 32)
        )

        fc_params = FCParams(
            layers=layers,
            activation=ActivationParams(
                activation=ch.activation,
                reste_o=ch.reste_o,
                binary_qmode=ch.quatization_mode,
            ),
            qmode=ch.quatization_mode,
            dropout_rate=ch.dropout,
        )
        train_params = NNTrainParams(
            DatasetCls=self.DatasetCls,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            epochs=self.p.epochs,
            learning_rate=ch.learning_rate,
            weight_decay=ch.weight_decay,
            early_stop_patience=self.p.patience,
        )
        return MLPParams(fc=fc_params, train=train_params)

    def compute_nn_complexity(self, p: MLPParams) -> float:
        complexity = 0

        prev_layer = p.fc.layers[0]
        for layer in p.fc.layers[1:]:
            mults = prev_layer.height * layer.height
            bitwidth = prev_layer.bitwidth
            complexity += mults * (math.log2(max(2, bitwidth)) * 3)

            prev_layer = layer

        activation_coef = 3 if p.fc.activation.activation == Activation.RELU else 1.2
        complexity *= activation_coef

        return complexity

    @lru_cache(maxsize=1)
    def get_min_complexity(self) -> float:
        x = RawMLPChromosome.get_bounds()[0]
        ch = RawMLPChromosome(x).parse()
        params = self.get_nn_params(ch)
        complexity = self.compute_nn_complexity(params)
        return complexity

    @lru_cache(maxsize=1)
    def get_max_complexity(self) -> float:
        x = RawMLPChromosome.get_bounds()[1]
        ch = RawMLPChromosome(x).parse()
        params = self.get_nn_params(ch)
        complexity = self.compute_nn_complexity(params)
        return complexity

    @staticmethod
    def normalize(x: float, min: float, max: float) -> float:
        if x < min:
            return 0.0
        elif x > max:
            return 1.0
        else:
            return (x - min) / (max - min)

    @staticmethod
    def denormalize(x: float, min: float, max: float) -> float:
        if x < 0.0:
            return min
        elif x > 1.0:
            return max
        else:
            return x * (max - min) + min

    def result_as_df(self, res: Result):
        data = []
        for i in range(len(res.X)):
            x = res.X[i]
            f = res.F[i]
            accuracy = self.denormalize(-f[0], 0, 100)
            complexity = self.denormalize(
                f[1], self.get_min_complexity(), self.get_max_complexity()
            )

            ch = RawMLPChromosome(x).parse()
            params = self.get_nn_params(ch)

            data.append(
                {
                    "Accuracy": accuracy,
                    "Complexity": complexity,
                    **asdict(params),
                    "Chromosome": x,
                }
            )

        return pd.DataFrame(data)
