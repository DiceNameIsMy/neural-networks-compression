from dataclasses import dataclass, fields

import numpy as np

from src.models.quant.enums import ActivationModule, QMode


@dataclass
class Chromosome:
    in_bitwidth: int

    hidden_height: int
    hidden_layers: int
    hidden_bitwidth1: int
    hidden_bitwidth2: int
    hidden_bitwidth3: int

    dropout: float
    activation: ActivationModule
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


@dataclass
class RawChromosome:
    x: np.ndarray[int]

    def parse(self) -> Chromosome:
        x = list(self.x)
        ch = Chromosome(
            in_bitwidth=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_height=x.pop(0),
            hidden_layers=x.pop(0),
            hidden_bitwidth1=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_bitwidth2=BITWIDTHS_MAPPING[x.pop(0)],
            hidden_bitwidth3=BITWIDTHS_MAPPING[x.pop(0)],
            dropout=DROPOUT_MAPPING[x.pop(0)],
            activation=self.get_activation(x.pop(0)),
            quatization_mode=self.get_qmode(x.pop(0)),
            binarization_mode=self.get_qmode(x.pop(0)),
            learning_rate=LEARNING_RATES_MAPPING[x.pop(0)],
            weight_decay=WEIGHT_DECAY_MAPPING[x.pop(0)],
        )
        return ch

    @staticmethod
    def get_bounds() -> tuple[np.ndarray, np.ndarray]:
        bounds = (
            (0, len(BITWIDTHS_MAPPING) - 1),
            (1, 8),
            (0, 3),
            # Hidden layers bitwidths
            (0, len(BITWIDTHS_MAPPING) - 1),
            (0, len(BITWIDTHS_MAPPING) - 1),
            (0, len(BITWIDTHS_MAPPING) - 1),
            #
            (0, len(DROPOUT_MAPPING) - 1),
            (0, 3),
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
