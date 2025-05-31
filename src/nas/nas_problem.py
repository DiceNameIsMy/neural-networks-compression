import logging
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result
from torch.utils import data

from src.datasets.dataset import CnnDataset, MlpDataset
from src.models.cnn import CNN, CNNParams
from src.models.eval import NNArchitectureEvaluator
from src.models.mlp import MLP, MLPParams
from src.nas.chromosome import Chromosome, ChromosomeConfig
from src.nas.nas_params import NasParams

logger = logging.getLogger(__name__)


class NasProblem(ElementwiseProblem):
    # TODO: It it better to use best accuracy instead of mean
    #       accuracy as an optimization goal? Or perhaps I should
    #       keep using both, but then train the final population
    #       again & show best accuracy?

    # NOTE: On every evaluation a new set of dataloaders is created.
    #       It's reduntant, although only a small % of NAS is spent on that.

    p: NasParams
    chromosome_cfg: ChromosomeConfig
    DatasetCls: type[MlpDataset | CnnDataset]

    train_loader: data.DataLoader
    test_loader: data.DataLoader

    # Used to store best models that appear during NAS.
    # Key is a tuple of chromosome genes, value is a tuple of (accuracy, CNN model).
    #
    # NOTE: A best accuracy is chosen, but due to the training running on varying dataset
    # subsets, there isn't a guarantee that it's actually the best model.
    # When comparing 2 models, they could perform good their own subset of data,
    # but have worse performance on another.
    best_architecture: dict[tuple[int], tuple[float, MLP | CNN]]

    def __init__(
        self,
        params: NasParams,
        DatasetCls: type[MlpDataset | CnnDataset],
        ChromosomeCls: type[Chromosome],
        chromosome_options_override: dict[str, tuple] | None = None,
    ):
        self.p = params
        self.DatasetCls = DatasetCls
        self.chromosome_cfg = ChromosomeConfig(
            ChromosomeCls, override=chromosome_options_override
        )
        self.train_loader, self.test_loader = self.DatasetCls.get_dataloaders(
            self.p.batch_size
        )
        self.best_architecture = {}

        x_low, x_high = self.chromosome_cfg.get_bounds()
        super().__init__(
            n_var=self.chromosome_cfg.get_size(),
            n_obj=2,
            n_ieq_constr=2,
            xl=x_low,
            xu=x_high + 0.99,  # Part of a workaround to the rounding problem
        )

    def _evaluate(self, x, out, *args, **kwargs):
        ch = self.chromosome_cfg.decode(x)
        params = self.get_nn_params(ch)
        logger.debug(f"Evaluating {params}")

        evaluator = NNArchitectureEvaluator(params.train)
        try:
            stats = evaluator.evaluate_accuracy(
                params, times=self.p.amount_of_evaluations
            )
            self.store_model_if_is_is_good(x, stats["max"], stats["best_model"])

            accuracy = stats["max"]

        except Exception as e:
            # Sometimes model training can fail due to incompatible genes.
            logger.error(
                f"Error during evaluation: {e}",
                exc_info=True,
                extra={"chromosome": x, "parsed_chromosome": ch, "nn_params": params},
            )
            accuracy = 0.0

        # Maximize accuracy
        norm_acc = self.normalize(accuracy, 0, 100)
        f1 = -norm_acc

        # Minimize NN complexity
        complexity = evaluator.evaluate_complexity(params)
        norm_complexity = self.normalize(
            complexity, self.get_min_complexity(), self.get_max_complexity()
        )
        f2 = norm_complexity
        out["F"] = [f1, f2]

        # Add constraints. They are optimized to be less than zero

        # Ensure accuracy is above the minimum
        g1 = self.p.min_accuracy - norm_acc
        # Ensure complexity is below the maximum
        g2 = norm_complexity - self.p.max_complexity
        out["G"] = [g1, g2]

    def store_model_if_is_is_good(self, x: np.ndarray, accuracy: float, model: MLP):
        if len(self.best_architecture) < self.p.population_size:
            # Store the best model for this chromosome
            self.best_architecture[tuple(x)] = (accuracy, model)
            return

        worst_key = min(
            self.best_architecture, key=lambda k: self.best_architecture[k][0]
        )
        worst_accuracy = self.best_architecture[worst_key][0]

        # Replace the worst model if a new one is better
        if accuracy > worst_accuracy:
            self.best_architecture.pop(worst_key)
            self.best_architecture[tuple(x)] = (accuracy, model)

    def get_nn_params(self, ch: Chromosome) -> MLPParams | CNNParams:
        raise NotImplementedError(
            "get_nn_params method must be implemented in the subclass"
        )

    @lru_cache(maxsize=1)
    def get_min_complexity(self) -> float:
        x = self.chromosome_cfg.get_bounds()[0]
        ch = self.chromosome_cfg.decode(x)
        params = self.get_nn_params(ch)
        evaluator = NNArchitectureEvaluator(params.train)
        complexity = evaluator.evaluate_complexity(params)
        return complexity

    @lru_cache(maxsize=1)
    def get_max_complexity(self) -> float:
        x = self.chromosome_cfg.get_bounds()[1]
        ch = self.chromosome_cfg.decode(x)
        params = self.get_nn_params(ch)
        evaluator = NNArchitectureEvaluator(params.train)
        complexity = evaluator.evaluate_complexity(params)
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

            ch = self.chromosome_cfg.decode(x)
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
