from dataclasses import dataclass, fields

import numpy as np

from src.models.quant.enums import ActivationModule, QMode
from src.nas.chromosome import (
    ACTIVATION_MAPPING,
    BITWIDTHS_MAPPING,
    DROPOUT_MAPPING,
    LEARNING_RATES_MAPPING,
    QMODE_MAPPING,
    RESTE_O_MAPPING,
    WEIGHT_DECAY_MAPPING,
)

MLP_HIDDEN_LAYERS_MAPPING = (0, 1, 2, 3)


@dataclass
class MLPChromosome:
    in_bitwidth: int

    hidden_layers: int
    hidden_height1: int
    hidden_bitwidth1: int
    hidden_height2: int
    hidden_bitwidth2: int
    hidden_height3: int
    hidden_bitwidth3: int

    dropout: float

    activation: ActivationModule
    reste_o: float

    quatization_mode: QMode
    binarization_mode: QMode

    learning_rate: float
    weight_decay: float


@dataclass
class RawMLPChromosome:
    x: np.ndarray[int]

    def parse(self) -> MLPChromosome:
        x = list(self.x)
        ch = MLPChromosome(
            in_bitwidth=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_layers=MLP_HIDDEN_LAYERS_MAPPING[x.pop(0)],
            hidden_height1=x.pop(0),
            hidden_bitwidth1=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_height2=x.pop(0),
            hidden_bitwidth2=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_height3=x.pop(0),
            hidden_bitwidth3=BITWIDTHS_MAPPING[x.pop(0)],
            dropout=DROPOUT_MAPPING[x.pop(0)],
            activation=ACTIVATION_MAPPING[x.pop(0)],
            reste_o=RESTE_O_MAPPING[x.pop(0)],
            quatization_mode=QMODE_MAPPING[x.pop(0)],
            binarization_mode=QMODE_MAPPING[x.pop(0)],
            learning_rate=LEARNING_RATES_MAPPING[x.pop(0)],
            weight_decay=WEIGHT_DECAY_MAPPING[x.pop(0)],
        )
        return ch

    @staticmethod
    def get_bounds() -> tuple[np.ndarray, np.ndarray]:
        layer_height_bounds = (1, 8)
        layer_bitwidth_bounds = (0, len(BITWIDTHS_MAPPING) - 1)
        bounds = (
            layer_bitwidth_bounds,
            (0, len(MLP_HIDDEN_LAYERS_MAPPING) - 1),
            # Per-hidden layer config
            layer_height_bounds,
            layer_bitwidth_bounds,
            layer_height_bounds,
            layer_bitwidth_bounds,
            layer_height_bounds,
            layer_bitwidth_bounds,
            #
            (0, len(DROPOUT_MAPPING) - 1),
            (0, len(ACTIVATION_MAPPING) - 1),
            (0, len(RESTE_O_MAPPING) - 1),
            (0, len(QMODE_MAPPING) - 1),
            (0, len(QMODE_MAPPING) - 1),
            (0, len(LEARNING_RATES_MAPPING) - 1),
            (0, len(WEIGHT_DECAY_MAPPING) - 1),
        )
        low, high = np.column_stack(bounds)

        size = RawMLPChromosome.get_size()
        assert size == len(low)
        assert size == len(high)

        return np.array(low), np.array(high)

    @staticmethod
    def get_size() -> tuple[np.ndarray, np.ndarray]:
        return len(fields(MLPChromosome))
