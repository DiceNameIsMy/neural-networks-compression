import enum


class QMode(enum.Enum):
    DET = "det"
    STOCH = "stoch"


class NNParamsCompMode(enum.Enum):
    """
    Neural network parameters compression modes
    """

    NONE = "none"
    NBITS = "nbits"
    BINARY = "binary"
    TERNARY = "ternary"


class Activation(enum.Enum):
    NONE = "none"
    RELU = "relu"
    BINARIZE = "binary"
    BINARIZE_RESTE = "binary_ReSTE"
    TERNARIZE = "ternary"
