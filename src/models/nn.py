import logging
import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.constants import EPOCHS, LEARNING_RATE, MODELS_FOLDER
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


def save_model(model: torch.nn.Module, filename: str, override: bool = False):
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path) and not override:
        raise Exception(f"File {path} already exists. Unable to store model here.")

    torch.save(model.state_dict(), path)

    logger.info(f"Model `{filename}` was stored at {path}")


def load_model(model: torch.nn.Module, filename: str):
    path = os.path.join(MODELS_FOLDER, filename)
    if not os.path.exists(path):
        raise Exception(f"Path {path} does not exist")

    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model
