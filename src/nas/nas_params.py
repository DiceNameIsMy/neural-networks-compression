import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.repair import Repair
from pymoo.core.result import Result
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling

from src.constants import EPOCHS, POPULATION_FOLDER

logger = logging.getLogger(__name__)


class FloorRepair(Repair):
    """
    Used to properly round integer genes. RoundingRepair does not fit.

    Example: if gene's valid values are in (1,2,3), interval of values is from 1 to 3.
             If we round the gene instance, probability that it will be 2 is 0.5, but 0.25 for 1 and 3.
             This distribution has a bias towards 2, a non-edge value.

    A uniform distribution of values in the range of (1,3) is acheived when:
    1) expanding the valid range of values by 0.99 on the right
    2) using floor instead of round

    """

    def _do(self, problem, X, **kwargs):
        return X.astype(int)


class MyIntegerRandomSampling(IntegerRandomSampling):
    """
    Allows specifying initial population alongside randomly sampled population.
    """

    extend_by: np.ndarray | None

    def __init__(self, extend_by: np.ndarray | None = None):
        self.extend_by = extend_by
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        n_preexisting = len(self.extend_by) if self.extend_by is not None else 0
        sampled = self._do_sampling(problem, n_samples, problem.n_var - n_preexisting)

        if self.extend_by is None:
            return sampled

        if len(self.extend_by) == 0:
            self.extend_by = None
            return sampled

        result = np.concatenate((sampled, self.extend_by))
        self.extend_by = None
        return result

    def _do_sampling(self, problem, n_samples, n_var: int):
        n, (xl, xu) = n_var, problem.bounds()
        return np.column_stack(
            [np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)]
        )


@dataclass
class NasParams:
    # Architecture evaluation
    batch_size: int | None = None
    epochs: int = EPOCHS
    patience: int = 5
    amount_of_evaluations: int = 1

    # GA params
    population_size: int = 30
    population_offspring_count: int = 10
    algorithm_generations: int = 10

    # GA constraints
    min_accuracy: float = 0.0
    max_complexity: float = 1.0

    #
    population_store_file: str | None = None

    def get_algorithm(self) -> NSGA2:
        population = None

        if self.population_store_file is not None:
            population = self.load_population(self.population_store_file)

        if population is None or len(population) == 0:
            logger.info("Using random initial population")
        else:
            logger.info(
                f"Initial population loaded from `{self.population_store_file}` successfully"
            )

        sampling = MyIntegerRandomSampling(population)

        return NSGA2(
            pop_size=self.population_size,
            n_offsprings=self.population_offspring_count,
            sampling=sampling,  # type: ignore
            crossover=SBX(prob=0.9, eta=15, repair=FloorRepair()),
            mutation=PM(eta=20, repair=FloorRepair()),
            eliminate_duplicates=True,
        )

    def get_termination(self):
        return ("n_gen", self.algorithm_generations)

    @staticmethod
    def load_population(filename: str | None):
        if filename is None:
            return None

        path = os.path.join(POPULATION_FOLDER, filename)
        if not os.path.exists(path):
            return None

        try:
            df = pd.read_csv(path)
            return df.values
        except EmptyDataError as e:
            logger.error(f"Failed to load population from {path}: {e}")
            return np.array([])

    @staticmethod
    def store_population(res: Result, file: str | None):
        if file is None:
            file = os.path.join(POPULATION_FOLDER, "population.csv")

        # Create cache directory if it does not exist
        if not os.path.exists(file):
            os.makedirs(os.path.dirname(file), exist_ok=True)

        pd.DataFrame(res.X).to_csv(file, index=False)
