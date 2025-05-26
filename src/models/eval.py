import logging

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils import data

from src.constants import DEVICE, SEED
from src.models.cnn import CNNParams
from src.models.mlp import MLPParams

logger = logging.getLogger(__name__)


class NNEvaluator:
    p: MLPParams | CNNParams

    criterion: nn.CrossEntropyLoss

    min_loss = float("inf")
    epochs_without_improvements = 0

    def __init__(self, params: MLPParams | CNNParams):
        self.p = params
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, model: nn.Module):
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
        model: nn.Module,
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
    def test_model(self, model: nn.Module) -> float:
        model.eval()

        loss_sum = 0
        correct = 0
        for data, target in self.p.train.test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)

            loss = self.criterion(outputs, target)
            loss_sum += loss

            dim = len(outputs.size()) - 1
            pred = outputs.argmax(dim=dim)
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
            model = self.p.get_model().to(DEVICE)
            self.train_model(model)
            accuracies.append(self.test_model(model))

        return {
            "max": max(accuracies),
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "accuracies": accuracies,
        }


class KFoldNNEvaluator:
    p: MLPParams | CNNParams

    def __init__(self, params: MLPParams | CNNParams):
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
                batch_size=self.p.train.batch_size,
                shuffle=True,
            )
            self.p.train.test_loader = data.DataLoader(
                self.p.train.DatasetCls(X_test, y_test),
                batch_size=self.p.train.batch_size,
                shuffle=False,
            )
            evaluator = NNEvaluator(self.p)
            stats = evaluator.evaluate_model(times)
            accuracies += stats["accuracies"]

        return {
            "max": max(accuracies),
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "accuracies": accuracies,
        }
