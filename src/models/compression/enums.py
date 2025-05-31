import enum


class QMode(enum.Enum):
    DET = "det"
    STOCH = "stoch"


class NNParamsCompMode(enum.Enum):
    """
    Neural network parameters compression modes
    """

    BINARY = "binary"
    TERNARY = "ternary"
    NBITS = "nbits"
    NONE = "none"


class Activation(enum.Enum):
    NONE = "none"
    BINARIZE = "binary"
    BINARIZE_RESTE = "binary_ReSTE"
    TERNARIZE = "ternary"
    RELU = "relu"
