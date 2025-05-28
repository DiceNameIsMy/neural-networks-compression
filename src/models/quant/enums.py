import enum


class QMode(enum.Enum):
    DET = "det"
    STOCH = "stoch"


class WeightQuantMode(enum.Enum):
    NONE = "none"
    NBITS = "nbits"
    BINARY = "binary"
    BINARY_RESTE = "binary_reste"
    TERNARY = "ternary"


class ActivationModule(enum.Enum):
    NONE = "none"
    RELU = "relu"
    BINARIZE = "binary"
    BINARIZE_RESTE = "binary_ReSTE"
    TERNARIZE = "ternary"
