import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from src.constants import DEVICE, EPOCHS, LEARNING_RATE
from src.models.nn import ActivationParams
from src.models.quant import binary, binary_ReSTE, ternarize
from src.models.quant.enums import ActivationModule, QMode
from src.models.quant.weight_quant import Module_Quantize

logger = logging.getLogger(__name__)


@dataclass
class FCLayerParams:
    height: int
    bitwidth: int


@dataclass
class MLPParams:
    layers: list[FCLayerParams]
    activation: ActivationParams
    quantization_mode: QMode = QMode.DET

    # Other
    dropout_rate: int = 0.0

    # NN Training params
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = 0.0  # TODO: Parametrize?

    def get_activation_module(self):
        match self.activation.activation:
            case ActivationModule.RELU:
                return nn.ReLU()
            case ActivationModule.BINARIZE:
                return binary.Module_Binarize(self.activation.binary_quantization_mode)
            case ActivationModule.BINARIZE_RESTE:
                return binary_ReSTE.Module_Binarize_ReSTE(
                    self.activation.reste_threshold, self.activation.reste_o
                )
            case ActivationModule.TERNARIZE:
                return ternarize.Module_Ternarize()
            case _:
                raise Exception(
                    "Unknown activation function: "
                    + f"{self.activation.activation} of type {type(self.activation.activation)}"
                )


class BMLP(nn.Module):
    p: MLPParams

    def __init__(self, params: MLPParams):
        super().__init__()
        self.p = params
        if len(self.p.layers) < 2:
            raise Exception("Model can't have less than 2 layers")

        layers = []

        in_layer = self.p.layers[0]
        if in_layer.bitwidth < 32:
            layers.append(Module_Quantize(self.p.quantization_mode, in_layer.bitwidth))

        last_layer_height = in_layer.height
        for hidden in self.p.layers[1:-1]:
            layers.append(binary.BinarizeLinear(last_layer_height, hidden.height))
            layers.append(nn.BatchNorm1d(hidden.height))

            # Add activation
            layers.append(self.p.get_activation_module())

            last_layer_height = hidden.height

        # Last layer
        last_layer = self.p.layers[-1]
        layers.append(binary.BinarizeLinear(last_layer_height, last_layer.height))
        layers.append(nn.BatchNorm1d(last_layer.height))

        # Combine all layers into Sequential model
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if self.training:
            x = nn.Softmax(dim=1)(x)

        return x

    def summarize_architecture(self):
        summary = []
        for idx, layer in enumerate(self.layers):
            layer_info = {
                "index": idx,
                "type": type(layer).__name__,
                "details": str(layer),
            }
            summary.append(layer_info)
        return summary


class MLP(nn.Module):
    p: MLPParams

    def __init__(self, params: MLPParams):
        super().__init__()
        self.p = params
        if len(self.p.layers) < 2:
            raise Exception("Model can't have less than 2 layers")

        layers = []

        in_layer = self.p.layers[0]
        if in_layer.bitwidth < 32:
            layers.append(Module_Quantize(self.p.quantization_mode, in_layer.bitwidth))

        last_layer_height = in_layer.height
        for hidden in self.p.layers[1:]:
            layers.append(nn.Linear(last_layer_height, hidden.height))
            layers.append(nn.BatchNorm1d(hidden.height))

            # Add quantization
            if hidden.bitwidth < 32:
                layers.append(
                    Module_Quantize(self.p.quantization_mode, hidden.bitwidth)
                )

            # Add dropout
            if self.p.dropout_rate > 0:
                layers.append(nn.Dropout(self.p.dropout_rate))

            # Add activation
            layers.append(self.p.get_activation_module())

            last_layer_height = hidden.height

        # Combine all layers into Sequential model
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if self.training:
            x = nn.Softmax(dim=1)(x)

        return x

    def summarize_architecture(self):
        summary = []
        for idx, layer in enumerate(self.layers):
            layer_info = {
                "index": idx,
                "type": type(layer).__name__,
                "details": str(layer),
            }
            summary.append(layer_info)
        return summary


class MLPEvaluator:
    train_loader: DataLoader
    test_loader: DataLoader
    early_stop_patience: int

    min_loss = float("inf")
    epochs_without_improvements = 0
    train_log: list[dict[str, float]] = []

    def __init__(
        self, train_loader: DataLoader, test_loader: DataLoader, early_stop_patience=10
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.early_stop_patience = early_stop_patience

    def train_model(self, params: MLPParams, Model=MLP) -> float:
        self.min_loss = float("inf")
        self.epochs_without_improvements = 0
        self.train_log = []

        model = Model(params).to(DEVICE)
        # best_model = copy.deepcopy(model)

        best_accuracy = 0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
        )

        for epoch in range(1, params.epochs + 1):
            loss = self.train_epoch(model, optimizer, criterion, epoch)
            accuracy = self.test_model(model, criterion)
            self.train_log.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})

            best_accuracy = max(best_accuracy, accuracy)

            if self.should_stop_early(loss):
                logger.debug(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return best_accuracy

    def should_stop_early(self, loss: float) -> bool:
        if loss < self.min_loss:
            self.min_loss = loss
            self.epochs_without_improvements = 0
        else:
            self.epochs_without_improvements += 1

        should_stop_early = self.epochs_without_improvements >= self.early_stop_patience
        return should_stop_early

    def train_epoch(
        self, model: MLP, optimizer: optim.Optimizer, criterion, epoch_no: int
    ) -> float:
        model.train()

        amount_of_batches = len(self.train_loader)
        amount_of_datapoints = len(self.train_loader.dataset)

        trained_on = 0
        loss_sum = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            loss_sum += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trained_on += self.test_loader.batch_size
            if batch_idx % 5 == 0:
                logger.debug(
                    f"Train Epoch: {epoch_no:>2} [{trained_on:>4}/{amount_of_datapoints}] Loss: {loss.item():.4f}"
                )

        avg_loss = loss_sum / amount_of_batches
        return avg_loss

    @torch.no_grad()
    def test_model(self, model: MLP, criterion) -> float:
        model.eval()

        loss_sum = 0
        correct = 0
        for data, target in self.test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            batch_loss = criterion(outputs, target)
            loss_sum += batch_loss

            pred = outputs.argmax(dim=1)  # get the index of the max log-probability
            correct += pred.eq(target.argmax(dim=1)).sum().item()

        amount_of_batches = len(self.test_loader)
        loss = loss_sum / amount_of_batches

        amount_of_datapoints = len(self.test_loader.dataset)
        accuracy = 100.0 * correct / amount_of_datapoints

        logger.debug(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                loss, correct, amount_of_datapoints, accuracy
            )
        )

        return accuracy


def evaluate_model(
    p: MLPParams, train_loader, test_loader, *, times=1, patience=10, verbose=0
):
    evaluator = MLPEvaluator(train_loader, test_loader, early_stop_patience=patience)
    accuracies = [evaluator.train_model(p) for _ in range(times)]
    return {
        "max": max(accuracies),
        "mean": np.mean(accuracies),
        "std": np.std(accuracies),
    }
