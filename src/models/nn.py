import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.constants import EPOCHS, LEARNING_RATE, MODELS_FOLDER
from src.datasets.dataset import Dataset
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
    train_loader: DataLoader[Dataset]
    test_loader: DataLoader[Dataset]

    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = 0.0
    early_stop_patience: int = 5


def save_model(model: torch.nn.Module, filename: str, override: bool = False):
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path) and not override:
        raise Exception(f"File {path} already exists. Unable to store model here.")

    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, filename: str):
    path = os.path.join(MODELS_FOLDER, filename)
    if not os.path.exists(path):
        raise Exception(f"Path {path} does not exist")

    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model
