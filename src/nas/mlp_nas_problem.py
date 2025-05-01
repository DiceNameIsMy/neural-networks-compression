import math
from functools import lru_cache

from pymoo.core.problem import ElementwiseProblem

from src.datasets.dataset import Dataset
from src.models.mlp import FCLayerParams, MLPParams, evaluate_model
from src.models.quant.enums import ActivationModule
from src.nas.mlp_chromosome import BITWIDTHS_MAPPING, MLPChromosome, RawChromosome
from src.nas.nas import MlpNasParams


class MlpNasProblem(ElementwiseProblem):
    # TODO: I'm not getting many diverse solutions because
    #       I only have 2 objectives -> there isn't space for
    #       many distinct solutions. It that okay?

    # TODO: It it better to use best accuracy instead of mean accuracy as an optimization goal?
    #       Or perhaps I should keep using both, but then train
    #       the final population again & show best accuracy?

    p: MlpNasParams
    dataset: Dataset
    train_loader = None
    test_loader = None

    def __init__(self, params: MlpNasParams, dataset: Dataset):
        x_low, x_high = RawChromosome.get_bounds()
        super().__init__(
            n_var=RawChromosome.get_size(), n_obj=3, xl=x_low, xu=x_high + 0.99
        )  # Part of a workaround to the rounding problem

        self.p = params
        self.dataset = dataset
        self.train_loader, self.test_loader = self.dataset.get_dataloaders()

    def _evaluate(self, x, out, *args, **kwargs):
        ch = RawChromosome(x).parse()
        params = self.get_nn_params(ch)
        perf = evaluate_model(
            params,
            self.train_loader,
            self.test_loader,
            times=self.p.amount_of_evaluations,
            patience=self.p.patience,
        )
        accuracy = perf["max"]

        # Maximize accuracy
        f1 = -self.normalize(accuracy, 0, 100)

        # Minimize NN complexity
        complexity = self.compute_nn_complexity(params)
        f2 = self.normalize(
            complexity, self.get_min_complexity(), self.get_max_complexity()
        )

        # Minimize bitwidth sum
        low, high = BITWIDTHS_MAPPING[0], BITWIDTHS_MAPPING[-1]
        bitwidth_sum = ch.in_bitwidth
        if ch.hidden_layers >= 1:
            bitwidth_sum += ch.hidden_bitwidth1
        if ch.hidden_layers >= 2:
            bitwidth_sum += ch.hidden_bitwidth2
        if ch.hidden_layers >= 3:
            bitwidth_sum += ch.hidden_bitwidth3

        f3 = self.normalize(bitwidth_sum, low * 4, high * 4)

        out["F"] = [f1, f2, f3]

    def get_nn_params(self, ch: MLPChromosome) -> MLPParams:
        layers = []
        layers.append(FCLayerParams(self.dataset.input_size, ch.in_bitwidth))
        if ch.hidden_layers >= 1:
            layers.append(FCLayerParams(ch.hidden_height1, ch.hidden_bitwidth1))
        if ch.hidden_layers >= 2:
            layers.append(FCLayerParams(ch.hidden_height2, ch.hidden_bitwidth2))
        if ch.hidden_layers >= 3:
            layers.append(FCLayerParams(ch.hidden_height3, ch.hidden_bitwidth3))
        layers.append(FCLayerParams(self.dataset.output_size, 32))

        return MLPParams(
            layers=layers,
            learning_rate=ch.learning_rate,
            weight_decay=ch.weight_decay,
            activation=ch.activation,
            epochs=self.p.epochs,
            dropout_rate=ch.dropout,
            quantization_mode=ch.quatization_mode,
        )

    def compute_nn_complexity(self, p: MLPParams) -> float:
        complexity = 0

        # Compute input layer complexity
        if p.hidden_layers > 0:
            layer_in_mults = p.in_height * p.hidden_layers_heights[0]
        else:
            layer_in_mults = p.in_height * p.out_height

        complexity += layer_in_mults * (math.log2(max(2, p.in_bitwidth)) * 3)

        # Compute hidden layers complexity
        heights = list(p.hidden_layers_heights) + [p.out_height]
        for i in range(p.hidden_layers):
            mults = heights[i] * heights[i + 1]
            bitwidth = p.hidden_layers_bitwidths[i]
            complexity += mults * (math.log2(max(2, bitwidth)) * 3)

        activation_coef = 3 if p.activation == ActivationModule.RELU else 1.2
        complexity *= activation_coef

        return complexity

    @lru_cache(maxsize=1)
    def get_min_complexity(self) -> float:
        x = RawChromosome.get_bounds()[0]
        ch = RawChromosome(x).parse()
        params = self.get_nn_params(ch)
        complexity = self.compute_nn_complexity(params)
        return complexity

    @lru_cache(maxsize=1)
    def get_max_complexity(self) -> float:
        x = RawChromosome.get_bounds()[1]
        ch = RawChromosome(x).parse()
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
