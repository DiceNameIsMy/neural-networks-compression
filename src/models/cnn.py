import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim

from src.constants import DEVICE
from src.models.mlp import FCParams
from src.models.nn import NNTrainParams
from src.models.quant import ternarize
from src.models.quant.common import get_activation_module
from src.models.quant.conv import WrapperConv2d
from src.models.quant.enums import ActivationModule, QMode
from src.models.quant.weight_quant import Module_Quantize

logger = logging.getLogger(__name__)


@dataclass
class ConvLayerParams:
    channels: int
    kernel_size: int

    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True

    pooling_kernel_size: int = 1

    def add_pooling(self):
        return self.pooling_kernel_size > 1


@dataclass
class ConvParams:
    # Dataset specific params
    in_channels: int
    in_dimensions: int
    in_bitwidth: int
    out_height: int

    layers: list[ConvLayerParams]
    activation: ActivationModule
    qmode: QMode = QMode.DET

    # Other
    dropout_rate: int = 0.0

    def get_conv_layer(self) -> type[WrapperConv2d]:
        match self.activation:
            case ActivationModule.RELU:
                return WrapperConv2d
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
class CNNParams:
    conv: ConvParams
    fc: FCParams
    train: NNTrainParams
    in_bitwidth: int = 32


class CNN(nn.Module):
    p: CNNParams

    def __init__(self, p: CNNParams):
        super(CNN, self).__init__()
        self.p = p

        # Inputs quantization
        self.in_quantize_layer = (
            Module_Quantize(QMode.DET, p.in_bitwidth)
            if p.in_bitwidth < 32
            else nn.Identity()
        )

        self.conv_layers = self.build_conv_layers(p)

        fc_in_height = self._get_fc_in_height(p, self.conv_layers)

        self.fc_layers = self.build_fc_layers(p, fc_in_height)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_quantize_layer(x)

        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten for fc layers
        x = x.reshape(x.shape[0], -1)

        # Fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        return x

    @torch.no_grad()
    def inspect_conv_layers(self):
        logger.info("Inspecting convolutional layers...")

        # Forward pass dummy input through convolutional layers
        dummy_input = torch.zeros(
            1,
            self.p.conv.in_channels,
            self.p.conv.in_dimensions,
            self.p.conv.in_dimensions,
        )
        x = dummy_input
        for layer in self.conv_layers:
            x = layer(x)
            flattened_size = x.reshape(x.shape[0], -1).size(1)
            logger.info(
                f"Next layer shape: {x.shape}, equating to {flattened_size} inputs"
            )

        # Flatten the conv output
        flattened = x.reshape(x.shape[0], -1)
        logger.info(f"FC input size is {flattened.size(1)}")

    @staticmethod
    @torch.no_grad()
    def _get_fc_in_height(p: CNNParams, conv_layers: nn.ModuleList) -> int:
        # Forward pass dummy input through convolutional layers
        dummy_input = torch.zeros(
            1,
            p.conv.in_channels,
            p.conv.in_dimensions,
            p.conv.in_dimensions,
        )
        x = dummy_input
        for layer in conv_layers:
            x = layer(x)

        # Flatten the conv output
        flattened = x.reshape(x.shape[0], -1)
        return flattened.size(1)

    @classmethod
    def build_conv_layers(cls, p: CNNParams) -> nn.ModuleList:
        ConvModule = p.conv.get_conv_layer()
        conv_layers = nn.ModuleList()

        in_channels = p.conv.in_channels
        for layer_params in p.conv.layers:
            layers = []
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=layer_params.channels,
                    kernel_size=layer_params.kernel_size,
                    stride=layer_params.stride,
                )
            )
            layers.append(nn.BatchNorm2d(layer_params.channels))
            layers.append(get_activation_module(p.conv.activation, p.conv.qmode))

            if layer_params.add_pooling:
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=layer_params.pooling_kernel_size,
                        stride=layer_params.pooling_kernel_size,
                    )
                )

            conv_layers.append(nn.Sequential(*layers))

            in_channels = layer_params.channels

        return conv_layers

    @classmethod
    def build_fc_layers(cls, p: CNNParams, fc_in_height: int) -> nn.ModuleList:
        if len(p.fc.layers) < 2:
            raise Exception("Model can't have negative less than 2 layers")

        layers = []

        in_layer = p.fc.layers[0]
        in_layer.height = (
            fc_in_height  # IMPORTANT: in_height is dependant on conv layers.
        )

        last_layer_height = in_layer.height
        for hidden in p.fc.layers[1:-1]:
            layers.append(hidden.get_fc_layer(last_layer_height))
            layers.append(nn.BatchNorm1d(hidden.height))

            # Add dropout
            if p.fc.dropout_rate > 0:
                layers.append(nn.Dropout(p.fc.dropout_rate))

            # Add activation
            layers.append(p.fc.activation.get_fc_layer_activation())

            last_layer_height = hidden.height

        out_layer = p.fc.layers[-1]
        layers.append(nn.Linear(last_layer_height, out_layer.height))

        return nn.Sequential(*layers)


class CNNEvaluator:
    p: CNNParams

    criterion: nn.CrossEntropyLoss

    min_loss = float("inf")
    epochs_without_improvements = 0

    def __init__(self, params: CNNParams):
        self.p = params
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, model: CNN):
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
        model: CNN,
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
    def test_model(self, model: CNN) -> float:
        model.eval()

        loss_sum = 0
        correct = 0
        for data, target in self.p.train.test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)

            loss = self.criterion(outputs, target)
            loss_sum += loss

            dim = len(outputs.size()) - 1
            _, predicted = torch.max(outputs.data, dim=dim)
            correct += (predicted == target).sum().item()

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
            model = CNN(self.p).to(DEVICE)
            self.train_model(model)
            accuracies.append(self.test_model(model))

        return {
            "max": max(accuracies),
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
        }
