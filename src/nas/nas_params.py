import logging
import os
from dataclasses import dataclass

import pandas as pd
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


@dataclass
class NasParams:
    batch_size: int | None = None
    epochs: int = EPOCHS
    patience: int = 5
    amount_of_evaluations: int = 1

    population_size: int = 30
    population_offspring_count: int = 10

    algorithm_generations: int = 10

    population_store_file: str | None = None

    def get_algorithm(self) -> NSGA2:
        sampling = IntegerRandomSampling()

        if self.population_store_file is not None:
            population = self.load_population(self.population_store_file)
            if population is None:
                logger.info(
                    f"Population file `{self.population_store_file}` is empty. Using random sampling"
                )
            else:
                logger.info(
                    f"Population file loaded from `{self.population_store_file}` successfully"
                )
                sampling = population
        else:
            logger.info("No population file provided. Using random sampling")

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

        return pd.read_csv(path).values

    @staticmethod
    def store_population(res: Result, filename: str):

        # Create cache directory if it does not exist
        if not os.path.exists(POPULATION_FOLDER):
            os.makedirs(POPULATION_FOLDER)

        pd.DataFrame(res.X).to_csv(
            os.path.join(POPULATION_FOLDER, filename), index=False
        )
