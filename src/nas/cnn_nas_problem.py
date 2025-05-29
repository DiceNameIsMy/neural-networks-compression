import logging
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result
from torch.utils import data

from src.datasets.dataset import CnnDataset
from src.models.cnn import CNN, CNNParams, ConvLayerParams, ConvParams
from src.models.compression.enums import NNParamsCompMode, QMode
from src.models.eval import KFoldNNArchitectureEvaluator
from src.models.mlp import FCLayerParams, FCParams
from src.models.nn import ActivationParams, NNTrainParams
from src.nas.cnn_chromosome import CNNChromosome
from src.nas.nas_params import NasParams

logger = logging.getLogger(__name__)


class CnnNasProblem(ElementwiseProblem):
    p: NasParams
    DatasetCls: type[CnnDataset]
    train_loader: data.DataLoader
    test_loader: data.DataLoader

    # Used to store best models that appear during NAS.
    # Key is a tuple of chromosome genes, value is a tuple of (accuracy, CNN model).
    #
    # NOTE: A best accuracy is chosen, but due to the training running on varying dataset
    # subsets, there isn't a guarantee that it's actually the best model.
    # When comparing 2 models, they could perform good their own subset of data,
    # but have worse performance on another.
    #
    # NOTE: What "Best model" means needs to be clarified...
    best_models: dict[tuple[int], tuple[float, CNN]]

    def __init__(self, params: NasParams, DatasetCls: type[CnnDataset]):
        x_low, x_high = CNNChromosome.get_bounds()
        super().__init__(
            n_var=CNNChromosome.get_size(), n_obj=2, xl=x_low, xu=x_high + 0.99
        )  # Part of a workaround to the rounding problem

        self.p = params
        self.DatasetCls = DatasetCls
        self.train_loader, self.test_loader = self.DatasetCls.get_dataloaders(
            self.p.batch_size
        )
        self.best_models = {}

    def _evaluate(self, x, out, *args, **kwargs):
        ch = CNNChromosome.parse(x)
        params = self.get_nn_params(ch)
        logger.debug(f"Evaluating {params}")

        try:
            stats = KFoldNNArchitectureEvaluator(params).evaluate_accuracy(
                times=self.p.amount_of_evaluations
            )
            self.try_store_model(x, stats["max"], stats["best_model"])

            accuracy = stats["max"]

        except Exception as e:
            # Sometimes CNN model training can fail due to incompatible genes.
            logger.error(
                f"Error during evaluation: {e}",
                exc_info=True,
                extra={"chromosome": x, "parsed_chromosome": ch, "nn_params": params},
            )
            accuracy = 0.0

        # Maximize accuracy
        f1 = -self.normalize(accuracy, 0, 100)

        # Minimize NN complexity
        complexity = params.get_complexity()
        f2 = self.normalize(
            complexity, self.get_min_complexity(), self.get_max_complexity()
        )

        out["F"] = [f1, f2]

    def try_store_model(self, x: np.ndarray, accuracy: float, model: CNN):
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

    def get_nn_params(self, ch: CNNChromosome) -> CNNParams:
        activation = ActivationParams(
            activation=ch.activation,
            reste_o=ch.activation_reste_o,
            reste_threshold=ch.activation_reste_threshold,
            binary_qmode=ch.activation_qmode,
        )

        conv_layers = self._get_conv_layers(ch)
        conv_params = ConvParams(
            in_channels=self.DatasetCls.input_channels,
            in_dimensions=self.DatasetCls.input_dimensions,
            in_bitwidth=ch.in_bitwidth,
            out_height=self.DatasetCls.output_size,
            layers=conv_layers,
            compression=ch.compression,
            reste_threshold=ch.activation_reste_threshold,
            reste_o=ch.activation_reste_o,
            activation=activation,
            dropout_rate=ch.dropout,
        )

        fc_layers = self._get_fc_layers(ch)
        fc_params = FCParams(
            layers=fc_layers,
            activation=activation,
            qmode=QMode.DET,
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
        return CNNParams(
            conv=conv_params,
            fc=fc_params,
            train=train_params,
            in_bitwidth=ch.in_bitwidth,
        )

    def _get_fc_layers(self, ch: CNNChromosome) -> list[FCLayerParams]:
        layers = []
        if ch.fc_layers >= 1:
            layers.append(
                FCLayerParams(ch.fc_height1, NNParamsCompMode.NBITS, ch.fc_bitwidth1)
            )
        if ch.fc_layers >= 2:
            layers.append(
                FCLayerParams(ch.fc_height2, NNParamsCompMode.NBITS, ch.fc_bitwidth2)
            )
        if ch.fc_layers >= 3:
            layers.append(
                FCLayerParams(ch.fc_height3, NNParamsCompMode.NBITS, ch.fc_bitwidth3)
            )
        layers.append(
            FCLayerParams(self.DatasetCls.output_size, NNParamsCompMode.NONE, 32)
        )

        return layers

    def _get_conv_layers(self, ch: CNNChromosome) -> list[ConvLayerParams]:
        layers = []
        if ch.conv_layers >= 1:
            layers.append(
                ConvLayerParams(
                    channels=ch.conv_channels1,
                    kernel_size=3,
                    pooling_kernel_size=ch.conv_pooling_size1,
                    stride=ch.conv_stride1,
                )
            )
        if ch.conv_layers >= 2:
            layers.append(
                ConvLayerParams(
                    channels=ch.conv_channels2,
                    kernel_size=3,
                    pooling_kernel_size=ch.conv_pooling_size2,
                    stride=ch.conv_stride2,
                )
            )
        if ch.conv_layers >= 3:
            layers.append(
                ConvLayerParams(
                    channels=ch.conv_channels3,
                    kernel_size=3,
                    pooling_kernel_size=ch.conv_pooling_size3,
                    stride=ch.conv_stride3,
                )
            )
        return layers

    @lru_cache(maxsize=1)
    def get_min_complexity(self) -> float:
        x = CNNChromosome.get_bounds()[0]
        ch = CNNChromosome.parse(x)
        params = self.get_nn_params(ch)
        complexity = params.get_complexity()
        return complexity

    @lru_cache(maxsize=1)
    def get_max_complexity(self) -> float:
        x = CNNChromosome.get_bounds()[1]
        ch = CNNChromosome.parse(x)
        params = self.get_nn_params(ch)
        complexity = params.get_complexity()
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

            ch = CNNChromosome.parse(x)
            params = self.get_nn_params(ch)

            data.append(
                {
                    "Accuracy": accuracy,
                    "Complexity": complexity,
                    **{
                        f"conv_{key}": value
                        for key, value in asdict(params.conv).items()
                    },
                    **{f"fc_{key}": value for key, value in asdict(params.fc).items()},
                    **{
                        f"train_{key}": value
                        for key, value in asdict(params.train).items()
                        if not key.endswith("_loader")
                    },
                    "Chromosome": x,
                }
            )

        return pd.DataFrame(data)
