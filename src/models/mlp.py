from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from constants import DEVICE, EPOCHS, LEARNING_RATE
from models.quantization import (
    ActivationFunc,
    BinarizeLayer_ReSTE,
    BinaryActivation,
    QMode,
    QuantizeLayer,
)
from src.models.ternary_activation import TernaryActivation


@dataclass
class ModelParams:
    # Dataset specific params
    in_layer_height: int
    in_bitwidth: int
    out_height: int

    # MLP Artchitecture params
    hidden_height: int
    hidden_bitwidth: int
    model_layers: int
    activation: ActivationFunc  # `binarize` or `relu`

    # Training params
    dropout_rate: int = 0.0  # TODO: Parametrize?
    learning_rate: float = LEARNING_RATE

    epochs: int = EPOCHS
    quantization_mode: QMode = QMode.DET


class MLP(nn.Module):
    p: ModelParams

    def __init__(self, params: ModelParams):
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
            layers.append(QuantizeLayer(self.p.quantization_mode, self.p.in_bitwidth))

        # Setup hidden layers
        for _ in range(2, self.p.model_layers):
            layers.append(nn.Linear(layer_heights.pop(0), layer_heights[0]))
            if self.p.hidden_bitwidth < 32:
                layers.append(
                    QuantizeLayer(self.p.quantization_mode, self.p.hidden_bitwidth)
                )
            if self.p.dropout_rate > 0:
                layers.append(nn.Dropout(self.p.dropout_rate))

            layers.append(self._get_activation_func(params))

        # Output layer
        layers.append(nn.Linear(layer_heights.pop(0), layer_heights[0]))

        # Combine all layers into Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _get_activation_func(p: ModelParams):
        if p.activation.value == ActivationFunc.RELU.value:
            return nn.ReLU()
        elif p.activation.value == ActivationFunc.BINARIZE.value:
            return BinaryActivation(p.quantization_mode)
        elif p.activation.value == ActivationFunc.BINARIZE_RESTE.value:
            return BinarizeLayer_ReSTE()
        elif p.activation.value == ActivationFunc.TERNARIZE.value:
            return TernaryActivation()
        else:
            raise Exception(
                f"Unknown activation function: {p.activation} of type {type(p.activation)}"
            )


def train_epoch(
    model: MLP, optimizer, criterion, train_loader, epoch_no, *, print_state=True
):
    model.train()

    loss_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        loss_sum += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if print_state and batch_idx % 5 == 0:
            trained_on = batch_idx * len(data)
            print(
                f"Train Epoch: {epoch_no} [{trained_on}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}"
            )

    amount_of_batches = len(train_loader)
    avg_loss = loss_sum / amount_of_batches
    return avg_loss


@torch.no_grad()
def test(model: MLP, criterion, test_loader, *, print_state=True):
    model.eval()

    loss_sum = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        outputs = model(data)
        batch_loss = criterion(outputs, target)
        loss_sum += batch_loss
        if print_state:
            print("Test batch loss: " + str(batch_loss))

        pred = outputs.argmax(dim=1)  # get the index of the max log-probability
        correct += pred.eq(target.argmax(dim=1)).sum().item()

    amount_of_batches = len(test_loader)
    amount_of_datapoints = len(test_loader.dataset)
    loss = loss_sum / amount_of_batches

    accuracy = 100.0 * correct / amount_of_datapoints
    if print_state:
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                loss, correct, amount_of_datapoints, accuracy
            )
        )

    return accuracy


def test_model(p: ModelParams, train_loader, test_loader, *, patience=3, verbose=0):
    model = MLP(p).to(DEVICE)
    # best_model = copy.deepcopy(model)

    min_loss = float("inf")
    best_test_acc = 0
    without_consecutive_improvements = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=p.learning_rate)

    for epoch in range(1, p.epochs + 1):

        loss = train_epoch(
            model, optimizer, criterion, train_loader, epoch, print_state=(verbose >= 2)
        )

        # Retain the best accuracy
        test_acc = test(model, criterion, test_loader, print_state=(verbose >= 1))
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # Early stopping
        if loss < min_loss:
            min_loss = loss
            without_consecutive_improvements = 0
        else:
            without_consecutive_improvements += 1

        if without_consecutive_improvements >= patience:
            if verbose >= 1:
                print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return best_test_acc


def evaluate_model(p: ModelParams, train_loader, test_loader, *, times=1, verbose=0):
    accuracies = [
        test_model(p, train_loader, test_loader, verbose=verbose) for _ in range(times)
    ]
    return {
        "max": max(accuracies),
        "mean": np.mean(accuracies),
        "std": np.std(accuracies),
    }
