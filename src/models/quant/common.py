from torch import nn

from models.quant import binary, binary_ReSTE, ternarize
from models.quant.enums import ActivationModule, QMode


def get_activation_module(activation: ActivationModule, qmode: QMode):
    match activation:
        case ActivationModule.RELU:
            return nn.ReLU()
        case ActivationModule.BINARIZE:
            return binary.Module_Binarize(qmode)
        case ActivationModule.BINARIZE_RESTE:
            return binary_ReSTE.Module_Binarize_ReSTE()
        case ActivationModule.TERNARIZE:
            return ternarize.Module_Ternarize()
        case _:
            raise Exception(
                f"Unknown activation function: {activation} of type {type(activation)}"
            )
