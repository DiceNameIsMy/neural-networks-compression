import logging
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn

from src.models.mlp import FCParams
from src.models.nn import ActivationParams, NNTrainParams
from src.models.quant import binary_ReSTE, ternarize
from src.models.quant.conv import Conv2dWrapper
from src.models.quant.enums import QMode, WeightQuantMode
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

    weight_qmode: WeightQuantMode
    reste_threshold: float
    reste_o: float

    activation: ActivationParams

    # Other
    dropout_rate: float = 0.0

    def get_conv_module_cls(self) -> type[Conv2dWrapper]:
        match self.weight_qmode:
            case WeightQuantMode.NONE:
                return Conv2dWrapper
            case WeightQuantMode.NBITS:
                return Conv2dWrapper
            case WeightQuantMode.BINARY:
                return ternarize.BinaryConv2d
            case WeightQuantMode.BINARY_RESTE:
                return partial(
                    binary_ReSTE.Binary_ReSTE_Conv2d,
                    threshold=self.reste_threshold,
                    o=self.reste_o,
                )  # type: ignore
            case WeightQuantMode.TERNARY:
                return ternarize.TernaryConv2d
            case _:
                raise Exception(
                    f"Unknown weight quantization function: {self.weight_qmode} of type {type(self.weight_qmode)}"
                )


@dataclass
class CNNParams:
    conv: ConvParams
    fc: FCParams
    train: NNTrainParams
    in_bitwidth: int = 32

    def get_model(self) -> "CNN":
        return CNN(self)


class CNN(nn.Module):
    p: CNNParams
    conv_layers: nn.ModuleList
    fc_layers: nn.Sequential

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
        ConvModule = p.conv.get_conv_module_cls()
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
                    padding=layer_params.padding,
                    dilation=layer_params.dilation,
                    groups=layer_params.groups,
                    bias=layer_params.bias,
                )
            )
            layers.append(nn.BatchNorm2d(layer_params.channels))
            layers.append(p.conv.activation.get_activation_module())

            if layer_params.add_pooling():
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
    def build_fc_layers(cls, p: CNNParams, fc_in_height: int) -> nn.Sequential:
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
            layers.append(p.fc.activation.get_activation_module())

            last_layer_height = hidden.height

        out_layer = p.fc.layers[-1]
        layers.append(nn.Linear(last_layer_height, out_layer.height))

        return nn.Sequential(*layers)
