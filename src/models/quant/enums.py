import enum


class QMode(enum.Enum):
    DET = "det"
    STOCH = "stoch"


class WeightQuantMode(enum.Enum):
    NONE = "none"
    NBITS = "nbits"
    BINARY = "binary"
    TERNARY = "ternary"


class ActivationModule(enum.Enum):
    RELU = "relu"
    BINARIZE = "binary"
    BINARIZE_RESTE = "binary_ReSTE"
    TERNARIZE = "ternary"
