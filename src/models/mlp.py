import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from constants import DEVICE, EPOCHS, LEARNING_RATE
from models.quant.common import get_activation_module
from models.quant.enums import ActivationModule, QMode
from models.quant.weight_quant import Module_Quantize

logger = logging.getLogger(__name__)


@dataclass
class MLPParams:
    # Dataset specific params
    in_layer_height: int
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
        layer_heights = (
            [self.p.in_layer_height]
            + [self.p.hidden_height] * hidden_layers_count
            + [self.p.out_height]
        )

        if self.p.in_bitwidth < 32:
            layers.append(Module_Quantize(self.p.quantization_mode, self.p.in_bitwidth))

        # Setup hidden layers
        for _ in range(2, self.p.model_layers):
            layers.append(nn.Linear(layer_heights.pop(0), layer_heights[0]))
            if self.p.hidden_bitwidth < 32:
                layers.append(
                    Module_Quantize(self.p.quantization_mode, self.p.hidden_bitwidth)
                )
            if self.p.dropout_rate > 0:
                layers.append(nn.Dropout(self.p.dropout_rate))

            layers.append(
                get_activation_module(params.activation, params.quantization_mode)
            )

        # Output layer
        layers.append(nn.Linear(layer_heights.pop(0), layer_heights[0]))

        # Combine all layers into Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLPEvaluator:
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    early_stop_patience: int

    def __init__(self, train_loader, test_loader, early_stop_patience=3):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.early_stop_patience = early_stop_patience

    def train_model(self, params: MLPParams) -> float:
        model = MLP(params).to(DEVICE)
        # best_model = copy.deepcopy(model)

        min_loss = float("inf")
        best_test_acc = 0
        without_consecutive_improvements = 0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

        for epoch in range(1, params.epochs + 1):

            loss = self.train_epoch(model, optimizer, criterion, epoch)

            # Retain the best accuracy
            test_acc = self.test_model(model, criterion)
            if test_acc > best_test_acc:
                best_test_acc = test_acc

            # Early stopping
            if loss < min_loss:
                min_loss = loss
                without_consecutive_improvements = 0
            else:
                without_consecutive_improvements += 1

            if without_consecutive_improvements >= self.early_stop_patience:
                logger.debug(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return best_test_acc

    def train_epoch(self, model: MLP, optimizer, criterion, epoch_no) -> float:
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
                    f"Train Epoch: {epoch_no} [{trained_on}/{amount_of_datapoints}] Loss: {loss.item():.4f}"
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
        amount_of_datapoints = len(self.test_loader.dataset)
        loss = loss_sum / amount_of_batches

        accuracy = 100.0 * correct / amount_of_datapoints

        logger.debug(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
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
