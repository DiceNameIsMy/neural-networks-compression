from dataclasses import dataclass

from src.models.compression.enums import Activation, NNParamsCompMode, QMode
from src.nas.chromosome import (
    ACTIVATION_MAPPING,
    BITWIDTHS_MAPPING,
    DROPOUT_MAPPING,
    LEARNING_RATES_MAPPING,
    NN_PARAMS_COMP_MODE_MAPPING,
    QMODE_MAPPING,
    RESTE_O_MAPPING,
    RESTE_THRESHOLD_MAPPING,
    WEIGHT_DECAY_MAPPING,
    Chromosome,
    with_options,
)

CONV_LAYERS_MAPPING = (1, 2, 3, 4)
CONV_CHANNELS_MAPPING = (16, 24, 32, 64, 128)
CONV_STRIDES_MAPPING = (1, 2)
CONV_POOLING_SIZES_MAPPING = (1, 2)
FC_LAYERS_MAPPING = (1, 2, 3)
FC_HEIGHTS_MAPPING = (8, 16, 32, 64, 128)


@dataclass
class CNNChromosome(Chromosome):
    in_bitwidth: int = with_options(BITWIDTHS_MAPPING)

    conv_layers: int = with_options(CONV_LAYERS_MAPPING)

    conv_channels1: int = with_options(CONV_CHANNELS_MAPPING)
    conv_stride1: int = with_options(CONV_STRIDES_MAPPING)
    conv_pooling_size1: int = with_options(CONV_POOLING_SIZES_MAPPING)
    conv_compression_bitwidth1: int = with_options(BITWIDTHS_MAPPING)

    conv_channels2: int = with_options(CONV_CHANNELS_MAPPING)
    conv_stride2: int = with_options(CONV_STRIDES_MAPPING)
    conv_pooling_size2: int = with_options(CONV_POOLING_SIZES_MAPPING)
    conv_compression_bitwidth2: int = with_options(BITWIDTHS_MAPPING)

    conv_channels3: int = with_options(CONV_CHANNELS_MAPPING)
    conv_stride3: int = with_options(CONV_STRIDES_MAPPING)
    conv_pooling_size3: int = with_options(CONV_POOLING_SIZES_MAPPING)
    conv_compression_bitwidth3: int = with_options(BITWIDTHS_MAPPING)

    conv_channels4: int = with_options(CONV_CHANNELS_MAPPING)
    conv_stride4: int = with_options(CONV_STRIDES_MAPPING)
    conv_pooling_size4: int = with_options(CONV_POOLING_SIZES_MAPPING)
    conv_compression_bitwidth4: int = with_options(BITWIDTHS_MAPPING)

    fc_layers: int = with_options(FC_LAYERS_MAPPING)
    fc_height1: int = with_options(FC_HEIGHTS_MAPPING)
    fc_bitwidth1: int = with_options(BITWIDTHS_MAPPING)
    fc_height2: int = with_options(FC_HEIGHTS_MAPPING)
    fc_bitwidth2: int = with_options(BITWIDTHS_MAPPING)
    fc_height3: int = with_options(FC_HEIGHTS_MAPPING)
    fc_bitwidth3: int = with_options(BITWIDTHS_MAPPING)
    fc_output_bitwidth: int = with_options(BITWIDTHS_MAPPING)

    conv_compression: NNParamsCompMode = with_options(NN_PARAMS_COMP_MODE_MAPPING)
    fc_compression: NNParamsCompMode = with_options(NN_PARAMS_COMP_MODE_MAPPING)

    dropout: float = with_options(DROPOUT_MAPPING)

    activation: Activation = with_options(ACTIVATION_MAPPING)
    activation_qmode: QMode = with_options(QMODE_MAPPING)
    activation_reste_o: float = with_options(RESTE_O_MAPPING)
    activation_reste_threshold: float = with_options(RESTE_THRESHOLD_MAPPING)

    learning_rate: float = with_options(LEARNING_RATES_MAPPING)
    weight_decay: float = with_options(WEIGHT_DECAY_MAPPING)
