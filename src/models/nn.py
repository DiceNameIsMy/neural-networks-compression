import logging
import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.constants import EPOCHS, LEARNING_RATE
from src.datasets.dataset import Dataset
from src.models.compression import binary, binary_ReSTE
from src.models.compression.enums import Activation, QMode

logger = logging.getLogger(__name__)


@dataclass
class ActivationParams:
    activation: Activation
    binary_qmode: QMode = QMode.DET
    reste_o: float = 3
    reste_threshold: float = 1.5

    def get_activation_module(self):
        match self.activation:
            case Activation.NONE:
                return nn.Identity()
            case Activation.RELU:
                return nn.ReLU()
            case Activation.BINARIZE:
                return binary.Binarize(self.binary_qmode)
            case Activation.BINARIZE_RESTE:
                return binary_ReSTE.BinarizeWithReSTE(
                    self.reste_threshold, self.reste_o
                )
            case _:
                raise Exception(
                    "Unknown activation function: "
                    + f"{self.activation} of type {type(self.activation)}"
                )

    def get_activation_complexity_coefficient(self) -> float:
        match self.activation:
            case Activation.NONE:
                return 0.0
            case Activation.RELU:
                return 10.0
            case Activation.BINARIZE:
                return 1.0
            case Activation.BINARIZE_RESTE:
                return 1.0
            case _:
                raise Exception(
                    "Unknown activation function: "
                    + f"{self.activation} of type {type(self.activation)}"
                )


@dataclass
class NNTrainParams:
    DatasetCls: type[Dataset]
    train_loader: DataLoader[Dataset]
    test_loader: DataLoader[Dataset]

    batch_size: int = Dataset.batch_size
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = 0.0
    early_stop_patience: int = 5


def save_model(model: torch.nn.Module, file: str, override: bool = False):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file), exist_ok=True)

    if os.path.exists(file) and not override:
        raise Exception(f"File {file} already exists. Unable to store model here.")

    torch.save(model.state_dict(), file)

    logger.info(f"Model {os.path.basename(file)} was stored")


def load_model(model: torch.nn.Module, file: str):
    if not os.path.exists(file):
        raise Exception(f"Path {file} does not exist")

    model.load_state_dict(torch.load(file))
    model.eval()  # Set the model to evaluation mode
    return model
