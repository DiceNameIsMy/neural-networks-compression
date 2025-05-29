from dataclasses import dataclass, field

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

CONV_LAYERS_MAPPING = (1, 2, 3)
CONV_CHANNELS_MAPPING = (16, 24, 32)
CONV_STRIDES_MAPPING = (1, 2)
CONV_POOLING_SIZES_MAPPING = (1, 2)
FC_LAYERS_MAPPING = (1, 2, 3)
FC_HEIGHTS_MAPPING = (16, 32, 64, 128)


@dataclass
class CNNChromosome(Chromosome):
    in_bitwidth: int = field(metadata=with_options(BITWIDTHS_MAPPING))

    conv_layers: int = field(metadata=with_options(CONV_LAYERS_MAPPING))
    conv_channels1: int = field(metadata=with_options(CONV_CHANNELS_MAPPING))
    conv_stride1: int = field(metadata=with_options(CONV_STRIDES_MAPPING))
    conv_pooling_size1: int = field(metadata=with_options(CONV_POOLING_SIZES_MAPPING))
    conv_channels2: int = field(metadata=with_options(CONV_CHANNELS_MAPPING))
    conv_stride2: int = field(metadata=with_options(CONV_STRIDES_MAPPING))
    conv_pooling_size2: int = field(metadata=with_options(CONV_POOLING_SIZES_MAPPING))
    conv_channels3: int = field(metadata=with_options(CONV_CHANNELS_MAPPING))
    conv_stride3: int = field(metadata=with_options(CONV_STRIDES_MAPPING))
    conv_pooling_size3: int = field(metadata=with_options(CONV_POOLING_SIZES_MAPPING))

    fc_layers: int = field(metadata=with_options(FC_LAYERS_MAPPING))
    fc_height1: int = field(metadata=with_options(FC_HEIGHTS_MAPPING))
    fc_bitwidth1: int = field(metadata=with_options(BITWIDTHS_MAPPING))
    fc_height2: int = field(metadata=with_options(FC_HEIGHTS_MAPPING))
    fc_bitwidth2: int = field(metadata=with_options(BITWIDTHS_MAPPING))
    fc_height3: int = field(metadata=with_options(FC_HEIGHTS_MAPPING))
    fc_bitwidth3: int = field(metadata=with_options(BITWIDTHS_MAPPING))

    dropout: float = field(metadata=with_options(DROPOUT_MAPPING))

    compression: NNParamsCompMode = field(
        metadata=with_options(NN_PARAMS_COMP_MODE_MAPPING)
    )

    activation: Activation = field(metadata=with_options(ACTIVATION_MAPPING))
    activation_qmode: QMode = field(metadata=with_options(QMODE_MAPPING))
    activation_reste_o: float = field(metadata=with_options(RESTE_O_MAPPING))
    activation_reste_threshold: float = field(
        metadata=with_options(RESTE_THRESHOLD_MAPPING)
    )

    learning_rate: float = field(metadata=with_options(LEARNING_RATES_MAPPING))
    weight_decay: float = field(metadata=with_options(WEIGHT_DECAY_MAPPING))
