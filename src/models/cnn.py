import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from src.constants import DEVICE, EPOCHS, LEARNING_RATE
from src.models.mlp import MLPParams
from src.models.quant import ternarize
from src.models.quant.common import get_activation_module
from src.models.quant.conv import WrapperConv2d
from src.models.quant.enums import ActivationModule, QMode
from src.models.quant.weight_quant import Module_Quantize

logger = logging.getLogger(__name__)


@dataclass
class CNNLayerParams:
    out_channels: int
    kernel_size: int
    stride: int
    add_pooling: bool
    pooling_kernel_size: int = 2


@dataclass
class CNNParams:
    # Dataset specific params
    in_channels: int
    in_dimensions: int
    in_bitwidth: int
    out_height: int

    conv_layers: list[CNNLayerParams]
    fc: MLPParams

    activation: ActivationModule = ActivationModule.BINARIZE

    # Activation specific params
    reste_o: float = 1.5
    reste_threshold: float = 1
    quantization_mode: QMode = QMode.DET

    dropout_rate: int = 0.0

    # NN Training params
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = 0.0  # TODO: Parametrize?

    def __post_init__(self):
        assert 0 < self.in_channels
        assert 0 < self.in_bitwidth

        assert len(self.conv_layers) >= 1, "CNN must have at least 1 convolution layer"

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


class CNN(nn.Module):
    p: CNNParams

    def __init__(self, p: CNNParams):
        super(CNN, self).__init__()
        self.p = p

        # Inputs quantization
        self.in_quantize_layer = (
            Module_Quantize(p.quantization_mode, p.in_bitwidth)
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
        # Forward pass dummy input through convolutional layers
        dummy_input = torch.zeros(
            1, self.p.in_channels, self.p.in_dimensions, self.p.in_dimensions
        )
        x = dummy_input
        for layer in self.conv_layers:
            x = layer(x)
            logger.info(
                f"Next layer shape: {x.shape}, which equates to {x.reshape(x.shape[0], -1).size(1)} inputs"
            )

        # Flatten the conv output
        flattened = x.reshape(x.shape[0], -1)
        logger.info(f"FC input size is {flattened.size(1)}")

    @staticmethod
    @torch.no_grad()
    def _get_fc_in_height(p: CNNParams, conv_layers: nn.ModuleList) -> int:
        # Forward pass dummy input through convolutional layers
        dummy_input = torch.zeros(1, p.in_channels, p.in_dimensions, p.in_dimensions)
        x = dummy_input
        for layer in conv_layers:
            x = layer(x)

        # Flatten the conv output
        flattened = x.reshape(x.shape[0], -1)
        return flattened.size(1)

    @classmethod
    def build_conv_layers(cls, p: CNNParams) -> nn.ModuleList:
        ConvModule = p.get_conv_layer()
        conv_layers = nn.ModuleList()

        in_channels = p.in_channels
        for layer_params in p.conv_layers:
            layers = []
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=layer_params.out_channels,
                    kernel_size=layer_params.kernel_size,
                    stride=layer_params.stride,
                )
            )
            layers.append(nn.BatchNorm2d(layer_params.out_channels))
            layers.append(get_activation_module(p.activation, p.quantization_mode))

            if layer_params.add_pooling:
                layers.append(
                    nn.MaxPool2d(kernel_size=layer_params.pooling_kernel_size)
                )

            conv_layers.append(nn.Sequential(*layers))

            in_channels = layer_params.out_channels

        return conv_layers

    @classmethod
    def build_fc_layers(cls, p: CNNParams, fc_in_height: int) -> nn.ModuleList:
        if p.fc.hidden_layers < 0:
            raise Exception("Model can't have negative amount of hidden layers")

        if p.fc.hidden_layers > len(p.fc.hidden_layers_bitwidths):
            raise Exception(
                "Not enough qunatization information for hidden layers. "
                + f"Expected {p.fc.hidden_layers} but got {len(p.fc.hidden_layers_bitwidths)}"
            )

        if p.fc.hidden_layers > len(p.fc.hidden_layers_heights):
            raise Exception(
                "Not enough height information for hidden layers. "
                + f"Expected {p.fc.hidden_layers} but got {len(p.fc.hidden_layers_heights)}"
            )

        layers = nn.ModuleList()

        if p.fc.in_bitwidth < 32:
            layers.append(Module_Quantize(p.fc.quantization_mode, p.fc.in_bitwidth))

        # Add hidden layers.
        quant_levels = list(p.fc.hidden_layers_bitwidths)
        layers_heights = [fc_in_height] + [
            p.fc.hidden_layers_heights[i] for i in range(p.fc.hidden_layers)
        ]
        for i in range(p.fc.hidden_layers):
            perceptrons_in = layers_heights[i]
            perceptrons_out = layers_heights[i + 1]

            # Add fully connected layer
            layers.append(nn.Linear(perceptrons_in, perceptrons_out))
            layers.append(nn.BatchNorm1d(perceptrons_out))

            # Add quantization
            quantize_to = quant_levels[i]
            if quantize_to < 32:
                layers.append(Module_Quantize(p.fc.quantization_mode, quantize_to))

            # Add dropout
            if p.fc.dropout_rate > 0:
                layers.append(nn.Dropout(p.fc.dropout_rate))

            # Add activation
            layers.append(p.fc.get_activation_module())

        # Add output layer
        perceptrons_in = layers_heights[-1]
        perceptrons_out = p.fc.out_height
        layers.append(nn.Linear(perceptrons_in, perceptrons_out))
        layers.append(nn.Softmax(dim=1))

        # Combine all layers into Sequential model
        return layers


@torch.no_grad()
def test_cnn(model, criterion, test_loader, *, print_state=True):
    model.eval()

    loss_sum = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            batch_loss = criterion(outputs, target)
            loss_sum += batch_loss
            if print_state:
                logger.info("Test batch loss: " + str(batch_loss))

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()

    amount_of_batches = len(test_loader)
    loss = loss_sum / amount_of_batches

    amount_of_datapoints = len(test_loader.dataset)
    accuracy = 100.0 * correct / amount_of_datapoints
    if print_state:
        logger.info(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                loss, correct, amount_of_datapoints, accuracy
            )
        )

    return accuracy


def train_cnn_epoch(
    model, optimizer, criterion, train_loader, epoch_no, *, print_state=True
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
            logger.info(
                f"Train Epoch: {epoch_no} [{trained_on}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}"
            )

    amount_of_batches = len(train_loader)
    avg_loss = loss_sum / amount_of_batches
    return avg_loss


def test_cnn_model(p: CNNParams, train_loader, test_loader, *, patience=3, verbose=0):
    model = CNN(p).to(DEVICE)
    # best_model = copy.deepcopy(model)

    min_loss = float("inf")
    best_test_acc = 0
    without_consecutive_improvements = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=p.learning_rate)

    for epoch in range(1, p.epochs + 1):

        loss = train_cnn_epoch(
            model, optimizer, criterion, train_loader, epoch, print_state=(verbose >= 2)
        )

        # Retain the best accuracy
        test_acc = test_cnn(model, criterion, test_loader, print_state=(verbose >= 1))
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
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return best_test_acc


def evaluate_cnn_model(p: CNNParams, train_loader, test_loader, *, times=1, verbose=0):
    try:
        accuracies = [
            test_cnn_model(p, train_loader, test_loader, verbose=verbose)
            for _ in range(times)
        ]
        return {
            "max": max(accuracies),
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
        }
    except Exception as e:
        # Can be useful when conv layer parameters are incompatible
        logger.info(f"CNN model evaluation with params: {p}")
        logger.info(f"Failed with error: {e}")
        return {
            "max": 0,
            "mean": 0,
            "std": 0,
        }
