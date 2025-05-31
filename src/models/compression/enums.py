import enum


class QMode(enum.Enum):
    DET = "det"
    STOCH = "stoch"


class NNParamsCompMode(enum.Enum):
    """
    Neural network parameters compression modes.
    Modes should be defined in order, from cheapest to most expensive.
    """

    BINARY = "binary"
    TERNARY = "ternary"
    NBITS = "nbits"
    NONE = "none"


class Activation(enum.Enum):
    """
    Activation functions used in the neural network.
    They should be defined in order, from cheapest to most expensive.
    """

    NONE = "none"
    BINARIZE = "binary"
    BINARIZE_RESTE = "binary_ReSTE"
    RELU = "relu"
