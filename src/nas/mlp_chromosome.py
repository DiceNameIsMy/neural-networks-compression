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
    WEIGHT_DECAY_MAPPING,
    Chromosome,
    with_options,
)

MLP_HIDDEN_LAYERS_MAPPING = (0, 1, 2, 3)
HIDDEN_LAYER_HEIGHTS_MAPPING = (6, 7, 8, 12, 16, 24, 32)


@dataclass
class MLPChromosome(Chromosome):
    # NN Topology
    in_bitwidth: int = with_options(BITWIDTHS_MAPPING)

    hidden_layers: int = with_options(MLP_HIDDEN_LAYERS_MAPPING)
    hidden_height1: int = with_options(HIDDEN_LAYER_HEIGHTS_MAPPING)
    hidden_bitwidth1: int = with_options(BITWIDTHS_MAPPING)
    hidden_height2: int = with_options(HIDDEN_LAYER_HEIGHTS_MAPPING)
    hidden_bitwidth2: int = with_options(BITWIDTHS_MAPPING)
    hidden_height3: int = with_options(HIDDEN_LAYER_HEIGHTS_MAPPING)
    hidden_bitwidth3: int = with_options(BITWIDTHS_MAPPING)

    output_bitwidth: int = with_options(BITWIDTHS_MAPPING)

    # Compression
    compression: NNParamsCompMode = with_options(NN_PARAMS_COMP_MODE_MAPPING)
    activation: Activation = with_options(ACTIVATION_MAPPING)
    reste_o: float = with_options(RESTE_O_MAPPING)
    parameters_quantization_mode: QMode = with_options(QMODE_MAPPING)
    activation_binarization_mode: QMode = with_options(QMODE_MAPPING)

    # Training
    dropout: float = with_options(DROPOUT_MAPPING)
    learning_rate: float = with_options(LEARNING_RATES_MAPPING)
    weight_decay: float = with_options(WEIGHT_DECAY_MAPPING)
