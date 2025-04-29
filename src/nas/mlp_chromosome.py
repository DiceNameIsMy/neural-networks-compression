from dataclasses import dataclass, fields

import numpy as np

from src.models.quant.enums import ActivationModule, QMode


@dataclass
class Chromosome:
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


BITWIDTHS_MAPPING = (1, 2, 3, 4, 5, 6, 7, 8)
LEARNING_RATES_MAPPING = (
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
)
WEIGHT_DECAY_MAPPING = (
    0.01,
    0.005,
    0.002,
    0.001,
)
DROPOUT_MAPPING = (0.0, 0.1, 0.2)
RESTE_O_MAPPING = (1.5, 2.0, 3.0, 4.0)


@dataclass
class RawChromosome:
    x: np.ndarray[int]

    def parse(self) -> Chromosome:
        x = list(self.x)
        ch = Chromosome(
            in_bitwidth=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_layers=x.pop(0),
            hidden_height1=x.pop(0),
            hidden_bitwidth1=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_height2=x.pop(0),
            hidden_bitwidth2=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_height3=x.pop(0),
            hidden_bitwidth3=BITWIDTHS_MAPPING[x.pop(0)],
            dropout=DROPOUT_MAPPING[x.pop(0)],
            activation=self.get_activation(x.pop(0)),
            reste_o=x.pop(0),
            quatization_mode=self.get_qmode(x.pop(0)),
            binarization_mode=self.get_qmode(x.pop(0)),
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
            (0, 3),
            # Per-hidden layer config
            layer_height_bounds,
            layer_bitwidth_bounds,
            layer_height_bounds,
            layer_bitwidth_bounds,
            layer_height_bounds,
            layer_bitwidth_bounds,
            #
            (0, len(DROPOUT_MAPPING) - 1),
            (0, 3),
            (0, len(RESTE_O_MAPPING) - 1),
            (0, 1),
            (0, 1),
            (0, len(LEARNING_RATES_MAPPING) - 1),
            (0, len(WEIGHT_DECAY_MAPPING) - 1),
        )
        low, high = np.column_stack(bounds)

        size = RawChromosome.get_size()
        assert size == len(low)
        assert size == len(high)

        return np.array(low), np.array(high)

    @staticmethod
    def get_size() -> tuple[np.ndarray, np.ndarray]:
        return len(fields(Chromosome))

    @staticmethod
    def get_activation(a: int) -> ActivationModule:
        if a == 0:
            return ActivationModule.RELU
        elif a == 1:
            return ActivationModule.BINARIZE_RESTE
        elif a == 2:
            return ActivationModule.BINARIZE
        elif a == 3:
            return ActivationModule.TERNARIZE
        else:
            raise ValueError(f"Invalid activation function index: {a}")

    @staticmethod
    def get_qmode(q: int) -> QMode:
        if q == 0:
            return QMode.DET
        elif q == 1:
            return QMode.STOCH
        else:
            raise ValueError(f"Invalid quantization mode index: {q}")
