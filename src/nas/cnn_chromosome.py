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

CONV_LAYERS_MAPPING = (1, 2, 3)
CONV_CHANNELS_MAPPING = (16, 24, 32)
CONV_STRIDES_MAPPING = (1, 2)
CONV_POOLING_SIZES_MAPPING = (1, 2)
FC_LAYERS_MAPPING = (1, 2, 3)
FC_HEIGHTS_MAPPING = (16, 32, 64, 128)


@dataclass
class CNNChromosome:
    in_bitwidth: int

    conv_layers: int
    conv_channels1: int
    conv_stride1: int
    conv_pooling_size1: int
    conv_channels2: int
    conv_stride2: int
    conv_pooling_size2: int
    conv_channels3: int
    conv_stride3: int
    conv_pooling_size3: int

    fc_layers: int
    fc_height1: int
    fc_bitwidth1: int
    fc_height2: int
    fc_bitwidth2: int
    fc_height3: int
    fc_bitwidth3: int

    dropout: float

    activation: ActivationModule
    reste_o: float

    quatization_mode: QMode
    binarization_mode: QMode

    learning_rate: float
    weight_decay: float


@dataclass
class RawCNNChromosome:
    x: np.ndarray[int]

    def parse(self) -> CNNChromosome:
        x = list(self.x)
        ch = CNNChromosome(
            in_bitwidth=BITWIDTHS_MAPPING[x.pop(0)],
            # Conv layers
            conv_layers=CONV_LAYERS_MAPPING[x.pop(0)],
            conv_channels1=CONV_CHANNELS_MAPPING[x.pop(0)],
            conv_stride1=CONV_STRIDES_MAPPING[x.pop(0)],
            conv_pooling_size1=CONV_POOLING_SIZES_MAPPING[x.pop(0)],
            conv_channels2=CONV_CHANNELS_MAPPING[x.pop(0)],
            conv_stride2=CONV_STRIDES_MAPPING[x.pop(0)],
            conv_pooling_size2=CONV_POOLING_SIZES_MAPPING[x.pop(0)],
            conv_channels3=CONV_CHANNELS_MAPPING[x.pop(0)],
            conv_stride3=CONV_STRIDES_MAPPING[x.pop(0)],
            conv_pooling_size3=CONV_POOLING_SIZES_MAPPING[x.pop(0)],
            # FC layers
            fc_layers=FC_LAYERS_MAPPING[x.pop(0)],
            fc_height1=FC_HEIGHTS_MAPPING[x.pop(0)],
            fc_bitwidth1=BITWIDTHS_MAPPING[x.pop(0)],
            fc_height2=FC_HEIGHTS_MAPPING[x.pop(0)],
            fc_bitwidth2=BITWIDTHS_MAPPING[x.pop(0)],
            fc_height3=FC_HEIGHTS_MAPPING[x.pop(0)],
            fc_bitwidth3=BITWIDTHS_MAPPING[x.pop(0)],
            # Other
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
        bitwidth_bounds = (0, len(BITWIDTHS_MAPPING) - 1)
        conv_channels_bounds = (0, len(CONV_CHANNELS_MAPPING) - 1)
        conv_stride_bounds = (0, len(CONV_STRIDES_MAPPING) - 1)
        conv_pooling_bounds = (0, len(CONV_POOLING_SIZES_MAPPING) - 1)
        fc_height_bounds = (1, len(FC_HEIGHTS_MAPPING) - 1)
        bounds = (
            bitwidth_bounds,
            # Conv layers config
            (0, len(CONV_LAYERS_MAPPING) - 1),
            conv_channels_bounds,
            conv_stride_bounds,
            conv_pooling_bounds,
            conv_channels_bounds,
            conv_stride_bounds,
            conv_pooling_bounds,
            conv_channels_bounds,
            conv_stride_bounds,
            conv_pooling_bounds,
            # FC layers config
            (0, len(FC_LAYERS_MAPPING) - 1),
            fc_height_bounds,
            bitwidth_bounds,
            fc_height_bounds,
            bitwidth_bounds,
            fc_height_bounds,
            bitwidth_bounds,
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

        size = RawCNNChromosome.get_size()
        assert size == len(low)
        assert size == len(high)

        return np.array(low), np.array(high)

    def get_size() -> tuple[np.ndarray, np.ndarray]:
        return len(fields(CNNChromosome))
