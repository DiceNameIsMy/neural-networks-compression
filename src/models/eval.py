import logging
from functools import lru_cache

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils import data

from src.constants import DATALOADERS_NUM_WORKERS, DEVICE, SEED
from src.datasets.dataset import Dataset
from src.models.cnn import CNNParams
from src.models.mlp import MLPParams
from src.models.nn import NNTrainParams

logger = logging.getLogger(__name__)


class NNArchitectureComplexityEvaluator:
    p: MLPParams | CNNParams

    def __init__(self, params: MLPParams | CNNParams):
        self.p = params

    def evaluate_complexity(self):
        if isinstance(self.p, MLPParams):
            return self.get_mlp_complexity(self.p)
        elif isinstance(self.p, CNNParams):
            return self.get_cnn_complexity(self.p)
        else:
            raise ValueError("Unsupported architecture type for complexity evaluation")

    def get_mlp_complexity(self, params: MLPParams) -> float:
        return params.fc.get_complexity()

    def get_cnn_complexity(self, params: CNNParams) -> float:
        conv_complexity = params.conv.get_conv_complexity()
        fc_complexity = params.fc.get_complexity()

        complexity = conv_complexity + fc_complexity

        return complexity


class NNArchitectureAccuracyEvaluator:
    p: MLPParams | CNNParams

    criterion: nn.CrossEntropyLoss

    def __init__(self, params: MLPParams | CNNParams):
        self.p = params
        self.criterion = nn.CrossEntropyLoss()

    def evaluate_accuracy(self, times=1):
        best_model = None
        accuracies = []
        for _ in range(times):
            model = self.p.get_model().to(DEVICE)
            model = self.train_model(model)
            acc = self.test_model(model)
            if best_model is None or acc > max(accuracies):
                best_model = model

            accuracies.append(acc)

        return {
            "max": max(accuracies),
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "accuracies": accuracies,
            "best_model": best_model,
        }

    def train_model(self, model: nn.Module):
        min_loss = float("inf")
        without_improvements = 0

        # TODO: Can be made customizable
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.p.train.learning_rate,
            weight_decay=self.p.train.weight_decay,
        )

        best_acc = 0.0
        best_model = None
        for epoch in range(1, self.p.train.epochs + 1):
            # Train for one epoch
            loss = self.train_epoch(model, optimizer, epoch)
            acc = self.test_model(model)

            # Remember best model
            if best_model is None or acc > best_acc:
                best_model = model

            # Stop early if needed
            if loss < min_loss:
                min_loss = loss
                without_improvements = 0
            else:
                without_improvements += 1

            should_stop_early = without_improvements >= self.p.train.early_stop_patience
            if should_stop_early:
                logger.debug(f"Early stopping triggered after {epoch + 1} epochs")
                break

        assert best_model is not None
        return best_model

    def train_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch_no: int,
    ) -> float:
        model.train()

        amount_of_batches = len(self.p.train.train_loader)
        amount_of_datapoints = len(self.p.train.train_loader.dataset)

        # Log progress every 25% of the batches
        log_progress_after = amount_of_batches // 4
        log_progress_counter = 0

        trained_on = 0
        loss_sum = 0
        for batch_idx, (_data, _target) in enumerate(self.p.train.train_loader):
            data, target = _data.to(DEVICE), _target.to(DEVICE)

            # Forward pass
            outputs = model(data)
            loss = self.criterion(outputs, target)
            loss_sum += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trained_on += self.p.train.test_loader.batch_size

            # Log progress
            log_progress_counter += 1
            if log_progress_counter == log_progress_after:
                log_progress_counter = 0

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
        for _data, _target in self.p.train.test_loader:
            data, target = _data.to(DEVICE), _target.to(DEVICE)
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
            f"Test set: Average loss: {average_loss:.4f}, Accuracy: {correct}/{amount_of_datapoints} ({accuracy:.2f}%)"
        )

        return accuracy


class NNArchitectureEvaluator:
    train: NNTrainParams
    kfold: StratifiedKFold

    def __init__(self, train_params: NNTrainParams):
        self.train = train_params
        self.kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def evaluate_accuracy(self, params: MLPParams | CNNParams, times=1):
        """
        Accuracy is represented by an average accuracy across all folds on a single dataset split.

        For example, given 5 folds(n_splits) and parameter times=3, 15 models will be trained and tested.
        5 models will be trained for each fold, average accuraacy computed. It will happend 3 times.
        mean and max accuracies will be computed using these 3 average accuracies.
        """
        best_model = None
        all_accuracies = []
        accuracies_per_fold: list[list[float]] = []

        dataloaders = self.get_dataloaders(
            self.kfold, self.train.DatasetCls, self.train.batch_size
        )
        for train_loader, test_loader in dataloaders:
            params.train.train_loader = train_loader
            params.train.test_loader = test_loader

            evaluator = NNArchitectureAccuracyEvaluator(params)
            stats = evaluator.evaluate_accuracy(times)

            if best_model is None or stats["max"] > max(all_accuracies):
                best_model = stats["best_model"]

            all_accuracies.extend(stats["accuracies"])
            accuracies_per_fold.append(stats["accuracies"])

        accuracies_per_dataset_run = np.column_stack(accuracies_per_fold)
        accuracies = np.mean(accuracies_per_dataset_run, axis=1)

        return {
            "max": max(accuracies),
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "accuracies": accuracies,
            "best_model": best_model,
        }

    def evaluate_complexity(self, params: MLPParams | CNNParams):
        evaluator = NNArchitectureComplexityEvaluator(params)
        complexity = evaluator.evaluate_complexity()
        return complexity

    @staticmethod
    @lru_cache(maxsize=1)
    def get_dataloaders(
        kfold: StratifiedKFold, DatasetCls: type[Dataset], batch_size: int
    ) -> list[tuple[data.DataLoader, data.DataLoader]]:
        X, y = DatasetCls.get_xy()

        dataloaders = []
        for train_indexes, test_indexes in kfold.split(X, y):
            X_train, y_train = X[train_indexes], y[train_indexes]
            X_test, y_test = X[test_indexes], y[test_indexes]

            train_loader = data.DataLoader(
                DatasetCls(X_train, y_train),
                batch_size=batch_size,
                shuffle=True,
                num_workers=DATALOADERS_NUM_WORKERS,
            )
            test_loader = data.DataLoader(
                DatasetCls(X_test, y_test),
                batch_size=batch_size,
                shuffle=False,
                num_workers=DATALOADERS_NUM_WORKERS,
            )
            dataloaders.append((train_loader, test_loader))

        return dataloaders
