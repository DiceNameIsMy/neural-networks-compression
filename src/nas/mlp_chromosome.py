from dataclasses import dataclass, fields

import numpy as np

from src.models.compression.enums import Activation, NNParamsCompMode, QMode
from src.nas.chromosome import (
    ACTIVATION_MAPPING,
    BITWIDTHS_MAPPING,
    DROPOUT_MAPPING,
    LEARNING_RATES_MAPPING,
    NN_PARAMS_COMP_MODE_MAPPING,
    QMODE_MAPPING,
    RESTE_O_MAPPING,
    WEIGHT_DECAY_MAPPING,
)

MLP_HIDDEN_LAYERS_MAPPING = (0, 1, 2, 3)
HIDDEN_LAYER_HEIGHTS_MAPPING = (6, 7, 8, 12, 16, 24, 32)


@dataclass
class MLPChromosome:
    # NN Topology
    in_bitwidth: int

    hidden_layers: int
    hidden_height1: int
    hidden_bitwidth1: int
    hidden_height2: int
    hidden_bitwidth2: int
    hidden_height3: int
    hidden_bitwidth3: int

    output_bitwidth: int

    # Compression
    compression: NNParamsCompMode
    activation: Activation
    reste_o: float
    parameters_quantization_mode: QMode
    activation_binarization_mode: QMode

    # Training
    dropout: float
    learning_rate: float
    weight_decay: float

    @staticmethod
    def parse(encoded: np.ndarray) -> "MLPChromosome":
        x = list(encoded)
        ch = MLPChromosome(
            in_bitwidth=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_layers=MLP_HIDDEN_LAYERS_MAPPING[x.pop(0)],
            hidden_height1=HIDDEN_LAYER_HEIGHTS_MAPPING[x.pop(0)],
            hidden_bitwidth1=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_height2=HIDDEN_LAYER_HEIGHTS_MAPPING[x.pop(0)],
            hidden_bitwidth2=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_height3=HIDDEN_LAYER_HEIGHTS_MAPPING[x.pop(0)],
            hidden_bitwidth3=BITWIDTHS_MAPPING[x.pop(0)],
            output_bitwidth=BITWIDTHS_MAPPING[x.pop(0)],
            # Compression
            compression=NN_PARAMS_COMP_MODE_MAPPING[x.pop(0)],
            activation=ACTIVATION_MAPPING[x.pop(0)],
            reste_o=RESTE_O_MAPPING[x.pop(0)],
            parameters_quantization_mode=QMODE_MAPPING[x.pop(0)],
            activation_binarization_mode=QMODE_MAPPING[x.pop(0)],
            #
            dropout=DROPOUT_MAPPING[x.pop(0)],
            learning_rate=LEARNING_RATES_MAPPING[x.pop(0)],
            weight_decay=WEIGHT_DECAY_MAPPING[x.pop(0)],
        )
        return ch

    @staticmethod
    def get_bounds() -> tuple[np.ndarray, np.ndarray]:
        layer_height_bounds = (1, len(HIDDEN_LAYER_HEIGHTS_MAPPING) - 1)
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
            layer_bitwidth_bounds,
            #
            (0, len(NN_PARAMS_COMP_MODE_MAPPING) - 1),
            (0, len(ACTIVATION_MAPPING) - 1),
            (0, len(RESTE_O_MAPPING) - 1),
            (0, len(QMODE_MAPPING) - 1),
            (0, len(QMODE_MAPPING) - 1),
            (0, len(DROPOUT_MAPPING) - 1),
            (0, len(LEARNING_RATES_MAPPING) - 1),
            (0, len(WEIGHT_DECAY_MAPPING) - 1),
        )
        low, high = np.column_stack(bounds)

        size = MLPChromosome.get_size()
        assert size == len(low)
        assert size == len(high)

        return np.array(low), np.array(high)

    @staticmethod
    def get_size() -> int:
        """Get the size of the MLP chromosome."""
        return len(fields(MLPChromosome))
