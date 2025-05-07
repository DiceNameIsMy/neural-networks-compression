from dataclasses import dataclass

from torch import nn
from torch.utils.data import DataLoader

from src.constants import EPOCHS, LEARNING_RATE
from src.models.quant import binary, binary_ReSTE, ternarize
from src.models.quant.conv import Conv2dWrapper
from src.models.quant.enums import ActivationModule, QMode


@dataclass
class ActivationParams:
    activation: ActivationModule
    binary_qmode: QMode = QMode.DET
    reste_o: float = 1
    reste_threshold: float = 1.5

    def get_fc_layer_activation(self):
        match self.activation:
            case ActivationModule.RELU:
                return nn.ReLU()
            case ActivationModule.BINARIZE:
                return binary.Module_Binarize(self.binary_qmode)
            case ActivationModule.BINARIZE_RESTE:
                return binary_ReSTE.Module_Binarize_ReSTE(
                    self.reste_threshold, self.reste_o
                )
            case ActivationModule.TERNARIZE:
                return ternarize.Module_Ternarize()
            case _:
                raise Exception(
                    "Unknown activation function: "
                    + f"{self.activation} of type {type(self.activation)}"
                )

    def get_conv_layer_class(self) -> type[Conv2dWrapper]:
        match self.activation:
            case ActivationModule.RELU:
                return Conv2dWrapper
            case ActivationModule.BINARIZE:
                return ternarize.BinaryConv2d
            case ActivationModule.BINARIZE_RESTE:
                raise Exception(
                    "Binarized ReSTE is not supported for convolution layers yet."
                )
            case ActivationModule.TERNARIZE:
                return ternarize.TernaryConv2d
            case _:
                raise Exception(
                    f"Unknown activation function: {self.activation} of type {type(self.activation)}"
                )


@dataclass
class NNTrainParams:
    train_loader: DataLoader
    test_loader: DataLoader

    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = 0.0
    early_stop_patience: int = 5
