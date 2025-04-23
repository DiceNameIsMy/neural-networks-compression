import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from constants import DEVICE, EPOCHS, LEARNING_RATE
from models.quant.common import get_activation_module
from models.quant.enums import ActivationModule, QMode
from models.quant.weight_quant import Module_Quantize

logger = logging.getLogger(__name__)


@dataclass
class MLPParams:
    # Dataset specific params
    in_height: int
    in_bitwidth: int
    out_height: int

    # MLP Architecture params
    hidden_height: int
    hidden_bitwidth: int
    model_layers: int
    activation: ActivationModule

    # Training params
    dropout_rate: int = 0.0  # TODO: Parametrize?
    learning_rate: float = LEARNING_RATE

    epochs: int = EPOCHS
    quantization_mode: QMode = QMode.DET


class MLP(nn.Module):
    p: MLPParams

    def __init__(self, params: MLPParams):
        super(MLP, self).__init__()
        self.p = params
        if self.p.model_layers < 2:
            raise Exception("Model must have at least 2 layers")

        layers = []

        hidden_layers_count = max(0, self.p.model_layers - 2)
        layers_heights = (
            [self.p.in_height]
            + [self.p.hidden_height] * hidden_layers_count
            + [self.p.out_height]
        )

        if self.p.in_bitwidth < 32:
            layers.append(Module_Quantize(self.p.quantization_mode, self.p.in_bitwidth))

        # Add hidden layers.
        for _ in range(hidden_layers_count):

            # Add fully connected layer
            layers.append(nn.Linear(layers_heights.pop(0), layers_heights[0]))

            # Add quantization
            if self.p.hidden_bitwidth < 32:
                layers.append(
                    Module_Quantize(self.p.quantization_mode, self.p.hidden_bitwidth)
                )

            # Add dropout
            if self.p.dropout_rate > 0:
                layers.append(nn.Dropout(self.p.dropout_rate))

            # Add activation
            layers.append(
                get_activation_module(params.activation, params.quantization_mode)
            )

        # Add output layer
        layers.append(nn.Linear(layers_heights.pop(0), layers_heights[0]))
        layers.append(nn.Softmax(dim=1))

        # Combine all layers into Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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

    def train_model(self, params: MLPParams) -> float:
        self.min_loss = float("inf")
        self.epochs_without_improvements = 0
        self.train_log = []

        model = MLP(params).to(DEVICE)
        # best_model = copy.deepcopy(model)

        best_accuracy = 0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

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


def evaluate_model(p: MLPParams, train_loader, test_loader, *, times=1, verbose=0):
    evaluator = MLPEvaluator(train_loader, test_loader)
    accuracies = [evaluator.train_model(p) for _ in range(times)]
    return {
        "max": max(accuracies),
        "mean": np.mean(accuracies),
        "std": np.std(accuracies),
    }
