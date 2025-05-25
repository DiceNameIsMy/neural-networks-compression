import logging
import math
from dataclasses import asdict
from functools import lru_cache

import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result
from torch.utils import data

from src.datasets.dataset import CnnDataset
from src.models.cnn import CNNEvaluator, CNNParams, ConvLayerParams, ConvParams
from src.models.mlp import FCLayerParams, FCParams
from src.models.nn import ActivationParams, NNTrainParams
from src.models.quant.enums import ActivationModule, WeightQuantMode
from src.nas.cnn_chromosome import CNNChromosome, RawCNNChromosome
from src.nas.nas_params import NasParams

logger = logging.getLogger(__name__)


class CnnNasProblem(ElementwiseProblem):
    p: NasParams
    dataset: type[CnnDataset]
    train_loader: data.DataLoader
    test_loader: data.DataLoader

    def __init__(self, params: NasParams, dataset: type[CnnDataset]):
        x_low, x_high = RawCNNChromosome.get_bounds()
        super().__init__(
            n_var=RawCNNChromosome.get_size(), n_obj=2, xl=x_low, xu=x_high + 0.99
        )  # Part of a workaround to the rounding problem

        self.p = params
        self.dataset = dataset
        self.train_loader, self.test_loader = self.dataset.get_dataloaders()

    def _evaluate(self, x, out, *args, **kwargs):
        ch = RawCNNChromosome(x).parse()
        params = self.get_nn_params(ch)
        logger.debug(f"Evaluating {params}")

        try:
            performance = CNNEvaluator(params).evaluate_model(
                times=self.p.amount_of_evaluations
            )
            accuracy = performance["max"]
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            accuracy = 0.0

        # Maximize accuracy
        f1 = -self.normalize(accuracy, 0, 100)

        # Minimize NN complexity
        complexity = self.compute_nn_complexity(params)
        f2 = self.normalize(
            complexity, self.get_min_complexity(), self.get_max_complexity()
        )

        out["F"] = [f1, f2]

    def get_nn_params(self, ch: CNNChromosome) -> CNNParams:
        conv_layers = self._get_conv_layers(ch)
        conv_params = ConvParams(
            in_channels=self.dataset.input_channels,
            in_dimensions=self.dataset.input_dimensions,
            in_bitwidth=ch.in_bitwidth,
            out_height=self.dataset.output_size,
            layers=conv_layers,
            activation=ch.activation,
            qmode=ch.quatization_mode,
            reste_threshold=ch.reste_threshold,
            reste_o=ch.reste_o,
            dropout_rate=ch.dropout,
        )

        fc_layers = self._get_fc_layers(ch)
        fc_params = FCParams(
            layers=fc_layers,
            activation=ActivationParams(
                activation=ch.activation,
                reste_o=ch.reste_o,
                binary_qmode=ch.quatization_mode,
            ),
            qmode=ch.quatization_mode,
            dropout_rate=ch.dropout,
        )
        train_params = NNTrainParams(
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
        layers.append(
            FCLayerParams(
                self.dataset.input_size, WeightQuantMode.NBITS, ch.in_bitwidth
            )
        )
        if ch.fc_layers >= 1:
            layers.append(
                FCLayerParams(ch.fc_height1, WeightQuantMode.NBITS, ch.fc_bitwidth1)
            )
        if ch.fc_layers >= 2:
            layers.append(
                FCLayerParams(ch.fc_height2, WeightQuantMode.NBITS, ch.fc_bitwidth2)
            )
        if ch.fc_layers >= 3:
            layers.append(
                FCLayerParams(ch.fc_height3, WeightQuantMode.NBITS, ch.fc_bitwidth3)
            )
        layers.append(FCLayerParams(self.dataset.output_size, WeightQuantMode.NONE, 32))

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

    def compute_nn_complexity(self, p: CNNParams) -> float:
        conv_complexity = 1000 * len(p.conv.layers)  # TODO: Make it more precise

        fc_complexity = 0
        prev_layer = p.fc.layers[0]
        for layer in p.fc.layers[1:]:
            mults = prev_layer.height * layer.height
            bitwidth = prev_layer.weight_bitwidth
            fc_complexity += mults * (math.log2(max(2, bitwidth)) * 3)

            prev_layer = layer

        activation_coef = (
            3 if p.fc.activation.activation == ActivationModule.RELU else 1.2
        )
        fc_complexity *= activation_coef

        complexity = 0
        complexity += conv_complexity
        complexity += fc_complexity
        return complexity

    @lru_cache(maxsize=1)
    def get_min_complexity(self) -> float:
        x = RawCNNChromosome.get_bounds()[0]
        ch = RawCNNChromosome(x).parse()
        params = self.get_nn_params(ch)
        complexity = self.compute_nn_complexity(params)
        return complexity

    @lru_cache(maxsize=1)
    def get_max_complexity(self) -> float:
        x = RawCNNChromosome.get_bounds()[1]
        ch = RawCNNChromosome(x).parse()
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

            ch = RawCNNChromosome(x).parse()
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
