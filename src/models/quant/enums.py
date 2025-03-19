import enum


class QMode(enum.Enum):
    DET = "det"
    STOCH = "stoch"


class ActivationFunc(enum.Enum):
    RELU = "relu"
    BINARIZE = "binary"
    BINARIZE_RESTE = "binary_ReSTE"
    TERNARIZE = "ternary"
