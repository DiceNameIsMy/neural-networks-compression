import math
from functools import lru_cache

from pymoo.core.problem import ElementwiseProblem

from src.datasets.dataset import Dataset
from src.models.mlp import MLPParams, evaluate_model
from src.models.quant.enums import ActivationModule
from src.nas.mlp_chromosome import Chromosome, RawChromosome
from src.nas.nas import MlpNasParams


class MlpNasProblem(ElementwiseProblem):
    # TODO: I'm not getting many diverse solutions because
    #       I only have 2 objectives -> there isn't space for
    #       many distinct solutions. It that okay?

    p: MlpNasParams
    dataset: Dataset
    train_loader = None
    test_loader = None

    def __init__(self, params: MlpNasParams, dataset: Dataset):
        x_low, x_high = RawChromosome.get_bounds()
        super().__init__(
            n_var=RawChromosome.get_size(), n_obj=2, xl=x_low, xu=x_high + 0.99
        )  # A workaround for the rounding problem

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
        accuracy = perf["mean"]

        f1 = -(self.normalize(accuracy, 0, 100))  # Maximize accuracy

        complexity = self.compute_nn_complexity(params)
        f2 = self.normalize(
            complexity, self._get_min_complexity(), self._get_max_complexity()
        )  # Minimize NN complexity

        out["F"] = [f1, f2]

    def get_nn_params(self, ch: Chromosome) -> MLPParams:
        return MLPParams(
            in_height=self.dataset.input_size,
            in_bitwidth=ch.in_bitwidth,
            out_height=self.dataset.output_size,
            hidden_height=ch.hidden_height,
            hidden_bitwidth=ch.hidden_bitwidth,
            model_layers=2 + ch.hidden_layers,
            learning_rate=ch.learning_rate,
            weight_decay=ch.weight_decay,
            activation=ch.activation,
            epochs=self.p.epochs,
            dropout_rate=ch.dropout,
            quantization_mode=ch.quatization_mode,
        )

    def compute_nn_complexity(self, p: MLPParams) -> float:
        has_hidden_layers = p.model_layers > 2

        layer1_ops = 0
        other_ops = 0
        if has_hidden_layers:
            layer1_ops += p.in_height * p.hidden_height
            other_ops += p.hidden_height * p.hidden_height * (p.model_layers - 2)
            other_ops += p.hidden_height * p.out_height
        else:
            layer1_ops += p.in_height * p.hidden_height
            other_ops += p.hidden_height * p.out_height

        complexity = 0
        complexity += layer1_ops * (math.log2(max(2, p.in_bitwidth)) * 3)
        complexity += other_ops * (math.log2(max(2, p.hidden_bitwidth)) * 3)

        activation_coef = 3 if p.activation == ActivationModule.RELU else 1.2
        complexity *= activation_coef

        return complexity

    @lru_cache(maxsize=1)
    def _get_min_complexity(self) -> float:
        x = RawChromosome.get_bounds()[0]
        ch = RawChromosome(x).parse()
        params = self.get_nn_params(ch)
        complexity = self.compute_nn_complexity(params)
        return complexity

    @lru_cache(maxsize=1)
    def _get_max_complexity(self) -> float:
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
