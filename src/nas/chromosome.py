from src.models.quant.enums import ActivationModule, QMode, WeightQuantMode

BITWIDTHS_MAPPING = (1, 2, 3, 4, 5, 6, 7, 8)

ACTIVATION_MAPPING = tuple(act for act in ActivationModule)
QMODE_MAPPING = tuple(q for q in QMode)
WEIGHT_QUANT_MODE_MAPPING = tuple(q for q in WeightQuantMode)
RESTE_O_MAPPING = (1.5, 2.0, 3.0, 4.0)
RESTE_THRESHOLD_MAPPING = (1.5,)

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
