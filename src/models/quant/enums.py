import enum


class QMode(enum.Enum):
    DET = "det"
    STOCH = "stoch"


class ActivationModule(enum.Enum):
    RELU = "relu"
    BINARIZE = "binary"
    BINARIZE_RESTE = "binary_ReSTE"
    TERNARIZE = "ternary"
