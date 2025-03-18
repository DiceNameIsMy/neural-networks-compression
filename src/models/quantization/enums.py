import enum


class QMode(enum.Enum):
    DET = "det"
    STOCH = "stoch"


class ActivationFunc(enum.Enum):
    RELU = "relu"
    BINARIZE = "binarize"
    BINARIZE_RESTE = "binarize_ReSTE"
    TERNARIZE = "ternarize"
