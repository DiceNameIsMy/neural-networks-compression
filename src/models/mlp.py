import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils import data

from src.constants import DEVICE, SEED
from src.models.nn import ActivationParams, NNTrainParams
from src.models.quant import binary, ternarize, weight_quant
from src.models.quant.enums import QMode, WeightQuantMode
from src.models.quant.weight_quant import Module_Quantize

logger = logging.getLogger(__name__)


@dataclass
class FCLayerParams:
    height: int
    weight_qmode: WeightQuantMode
    weight_bitwidth: int = 32

    def get_fc_layer(self, in_height: int) -> nn.Linear:
        match self.weight_qmode:
            case WeightQuantMode.NONE:
                return nn.Linear(in_height, self.height)
            case WeightQuantMode.NBITS:
                assert self.weight_bitwidth > 0, "Bitwidth must be greater than 0"
                assert (
                    self.weight_bitwidth < 32
                ), "For NBITS, bitwidth must be less than 32"
                return weight_quant.QuantizedWeightLinear(
                    self.weight_bitwidth, in_height, self.height
                )
            case WeightQuantMode.BINARY:
                return binary.BinarizeLinear(in_height, self.height)
            case WeightQuantMode.TERNARY:
                return ternarize.TernarizeLinear(in_height, self.height)
            case _:
                raise Exception(
                    "Unknown weight quantization mode: "
                    + f"{self.weight_qmode} of type {type(self.weight_qmode)}"
                )


@dataclass
class FCParams:
    layers: list[FCLayerParams]
    activation: ActivationParams
    qmode: QMode = QMode.DET

    # Other
    dropout_rate: float = 0.0


@dataclass
class MLPParams:
    fc: FCParams
    train: NNTrainParams


class MLP(nn.Module):
    def __init__(self, p: MLPParams):
        super().__init__()

        if len(p.fc.layers) < 2:
            raise Exception("Model can't have less than 2 layers")

        layers = []

        in_layer = p.fc.layers[0]
        if in_layer.weight_bitwidth < 32:
            layers.append(Module_Quantize(p.fc.qmode, in_layer.weight_bitwidth))

        last_layer_height = in_layer.height
        for hidden in p.fc.layers[1:]:
            layers.append(hidden.get_fc_layer(last_layer_height))
            layers.append(nn.BatchNorm1d(hidden.height))

            # Add quantization
            if hidden.weight_bitwidth < 32:
                layers.append(Module_Quantize(p.fc.qmode, hidden.weight_bitwidth))

            # Add dropout
            if p.fc.dropout_rate > 0:
                layers.append(nn.Dropout(p.fc.dropout_rate))

            # Add activation
            layers.append(p.fc.activation.get_fc_layer_activation())

            last_layer_height = hidden.height

        # Combine all layers into Sequential model
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)

        # Apply softmax for better interpretability during training
        if self.training:
            x = nn.Softmax(dim=1)(x)

        return x


class MLPEvaluator:
    p: MLPParams

    criterion: nn.CrossEntropyLoss

    min_loss = float("inf")
    epochs_without_improvements = 0

    def __init__(self, params: MLPParams):
        self.p = params
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, model: MLP):
        training_log = []

        # Reset early stopping parameters
        self.min_loss = float("inf")
        self.epochs_without_improvements = 0

        # TODO: Can be made customizable
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.p.train.learning_rate,
            weight_decay=self.p.train.weight_decay,
        )

        # best_model = copy.deepcopy(model)
        for epoch in range(1, self.p.train.epochs + 1):
            loss = self.train_epoch(model, optimizer, epoch)
            accuracy = self.test_model(model)

            training_log.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "accuracy": accuracy,
                }
            )

            if self.should_stop_early(loss):
                logger.debug(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return training_log

    def should_stop_early(self, loss: float) -> bool:
        if loss < self.min_loss:
            self.min_loss = loss
            self.epochs_without_improvements = 0
        else:
            self.epochs_without_improvements += 1

        should_stop_early = (
            self.epochs_without_improvements >= self.p.train.early_stop_patience
        )
        return should_stop_early

    def train_epoch(
        self,
        model: MLP,
        optimizer: optim.Optimizer,
        epoch_no: int,
    ) -> float:
        model.train()

        amount_of_batches = len(self.p.train.train_loader)
        amount_of_datapoints = len(self.p.train.train_loader.dataset)

        trained_on = 0
        loss_sum = 0
        for batch_idx, (data, target) in enumerate(self.p.train.train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Forward pass
            outputs = model(data)
            loss = self.criterion(outputs, target)
            loss_sum += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trained_on += self.p.train.test_loader.batch_size
            if batch_idx % 5 == 0:
                logger.debug(
                    f"Train Epoch: {epoch_no:>2} [{trained_on:>4}/{amount_of_datapoints}] Loss: {loss.item():.4f}"
                )

        avg_loss = loss_sum / amount_of_batches
        return avg_loss

    @torch.no_grad()
    def test_model(self, model: MLP) -> float:
        model.eval()

        loss_sum = 0
        correct = 0
        for data, target in self.p.train.test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs: torch.Tensor = model(data)

            loss = self.criterion(outputs, target)
            loss_sum += loss

            dim = len(outputs.size()) - 1
            pred = outputs.argmax(dim=dim)  # get the index of the max log-probability
            correct += (pred == target).sum().item()

        amount_of_batches = len(self.p.train.test_loader)
        average_loss = loss_sum / amount_of_batches

        amount_of_datapoints = len(self.p.train.test_loader.dataset)
        accuracy = 100.0 * correct / amount_of_datapoints

        logger.debug(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                average_loss, correct, amount_of_datapoints, accuracy
            )
        )

        return accuracy

    def evaluate_model(self, times=1):
        accuracies = []
        for _ in range(times):
            model = MLP(self.p).to(DEVICE)
            self.train_model(model)
            accuracies.append(self.test_model(model))

        return {
            "max": max(accuracies),
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "accuracies": accuracies,
        }


class KFoldMLPEvaluator:
    """
    Evaluates MLP model using K-Fold cross-validation.

    Given the same dataset, splits will be the same on each run, since the SEED is fixed.
    """

    p: MLPParams

    def __init__(self, params: MLPParams):
        self.p = params
        self.kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def evaluate_model(self, times=1):
        X, y = self.p.train.DatasetCls.get_xy()

        accuracies = []

        for train_indexes, test_indexes in self.kfold.split(X, y):
            X_train, y_train = X[train_indexes], y[train_indexes]
            X_test, y_test = X[test_indexes], y[test_indexes]

            self.p.train.train_loader = data.DataLoader(
                self.p.train.DatasetCls(X_train, y_train),
                batch_size=self.p.train.DatasetCls.batch_size,
                shuffle=True,
            )
            self.p.train.test_loader = data.DataLoader(
                self.p.train.DatasetCls(X_test, y_test),
                batch_size=self.p.train.DatasetCls.batch_size,
                shuffle=False,
            )
            evaluator = MLPEvaluator(self.p)
            stats = evaluator.evaluate_model(times)
            accuracies += stats["accuracies"]

        return {
            "max": max(accuracies),
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "accuracies": accuracies,
        }
