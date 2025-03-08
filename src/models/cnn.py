from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from constants import DEVICE, EPOCHS, LEARNING_RATE
from models.quantization import ActivationFunc, BinaryActivation, QMode, QuantizeLayer


@dataclass
class CNNLayerParams:
    out_channels: int
    kernel_size: int
    stride: int
    add_pooling: bool


@dataclass
class CNNParams:
    # Dataset specific params
    input_channels: int
    input_dimensions: int
    input_bitwidth: int
    output_height: int

    # NN Architecture params
    fc_height: int
    fc_layers: int
    fc_bitwidth: int
    conv_layers: list[CNNLayerParams]

    activation: ActivationFunc = ActivationFunc.BINARIZE  # `binarize` or `relu`

    # Training params
    dropout_rate: int = 0.0
    learning_rate: float = LEARNING_RATE
    epochs: int = EPOCHS
    quantization_mode: QMode = QMode.DET

    def __post_init__(self):
        assert 0 < self.input_channels
        assert 0 < self.input_bitwidth

        assert len(self.conv_layers) >= 1, "CNN must have at least 1 convolution layer"

        assert 0 < self.fc_layers, "CNN must have at least 1 fully connected layer"
        assert 0 < self.fc_bitwidth
        assert self.fc_bitwidth <= 32


class CNN(nn.Module):
    p: CNNParams

    def __init__(self, p: CNNParams):
        super(CNN, self).__init__()
        self.p = p

        # Inputs quantization
        self.input_quantize_layer = (
            QuantizeLayer(p.quantization_mode, p.input_bitwidth)
            if p.input_bitwidth < 32
            else nn.Identity()
        )

        # Convolutional layers
        self.conv_layers = nn.ModuleList()

        in_channels = p.input_channels
        for layer in p.conv_layers:
            modules = [
                nn.Conv2d(
                    in_channels,
                    layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                ),
                nn.BatchNorm2d(layer.out_channels),
                self._get_activation(p),
            ]
            if layer.add_pooling:
                modules.append(nn.MaxPool2d(kernel_size=2))

            self.conv_layers.append(nn.Sequential(*modules))

            in_channels = layer.out_channels

        # Fully connected layers
        self.fc_layers = nn.ModuleList()

        first_fc_input_size = self._get_first_fc_layer_input_size()
        prev_size = first_fc_input_size
        for _ in range(p.fc_layers):
            fc_layer = nn.Sequential(
                QuantizeLayer(self.p.quantization_mode, self.p.fc_bitwidth),
                nn.Linear(prev_size, p.fc_height),
                self._get_activation(p),
                nn.Dropout(
                    p.dropout_rate
                ),  # TODO: Exclude dropout if redundant (rate is 0.0)
            )
            self.fc_layers.append(fc_layer)
            prev_size = p.fc_height

        # Final classification layer
        self.classifier = nn.Linear(prev_size, p.output_height)

    def forward(self, x):
        x = self.input_quantize_layer(x)

        x = self._forward_conv_layers(x)

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # Fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        x = self.classifier(x)
        return x

    def _forward_conv_layers(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def inspect_conv_layers(self):
        # Forward pass dummy input through convolutional layers
        dummy_input = torch.zeros(
            1, self.p.input_channels, self.p.input_dimensions, self.p.input_dimensions
        )
        x = dummy_input
        for layer in self.conv_layers:
            x = layer(x)
            print(
                f"Next layer shape: {x.shape}, which equates to {x.reshape(x.shape[0], -1).size(1)} inputs"
            )

        # Flatten the conv output
        flattened = x.reshape(x.shape[0], -1)
        print(f"FC input size is {flattened.size(1)}")

    @torch.no_grad()
    def _get_first_fc_layer_input_size(self):
        # Forward pass dummy input through convolutional layers
        dummy_input = torch.zeros(
            1, self.p.input_channels, self.p.input_dimensions, self.p.input_dimensions
        )
        x = self._forward_conv_layers(dummy_input)

        # Flatten the conv output
        flattened = x.reshape(x.shape[0], -1)
        return flattened.size(1)

    @staticmethod
    def _get_activation(p: CNNParams):
        if p.activation == ActivationFunc.RELU:
            return nn.ReLU(inplace=True)
        elif p.activation == ActivationFunc.BINARIZE:
            return BinaryActivation(p.quantization_mode)
        else:
            raise Exception(f"Unknown activation function: {p.activation}")


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
                print("Test batch loss: " + str(batch_loss))

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()

    amount_of_batches = len(test_loader)
    loss = loss_sum / amount_of_batches

    amount_of_datapoints = len(test_loader.dataset)
    accuracy = 100.0 * correct / amount_of_datapoints
    if print_state:
        print(
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
            print(
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

    for epoch in range(p.epochs + 1):

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
                print(f"Early stopping triggered after {epoch + 1} epochs")
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
        print(f"CNN model evaluation with params: {p}")
        print(f"Failed with error: {e}")
        return {
            "max": 0,
            "mean": 0,
            "std": 0,
        }
