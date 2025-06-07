import logging
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn

from src.models.compression import ternarize, weight_quant
from src.models.compression.conv import Conv2dWrapper
from src.models.compression.enums import NNParamsCompMode, QMode
from src.models.compression.weight_quant import Quantize
from src.models.mlp import FCParams
from src.models.nn import ActivationParams

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

    compression: NNParamsCompMode = NNParamsCompMode.NONE
    bitwidth: int = 32

    pooling_kernel_size: int = 1

    def add_max_pooling(self):
        return self.pooling_kernel_size > 1

    def get_conv_module_cls(self) -> type[Conv2dWrapper]:
        match self.compression:
            case NNParamsCompMode.NONE:
                return Conv2dWrapper
            case NNParamsCompMode.NBITS:
                return partial(weight_quant.Conv2DQuantized, nbits=self.bitwidth)  # type: ignore
            case NNParamsCompMode.BINARY:
                return ternarize.Conv2dBinary
            case NNParamsCompMode.TERNARY:
                return ternarize.Conv2dTernary
            case _:
                raise Exception(
                    f"Unknown compression value `{self.compression}` of type `{type(self.compression)}`"
                )

    def get_conv_complexity_coefficient(self) -> float:
        match self.compression:
            case NNParamsCompMode.NONE:
                return 32.0
            case NNParamsCompMode.NBITS:
                return self.bitwidth
            case NNParamsCompMode.BINARY:
                return 1.0
            case NNParamsCompMode.TERNARY:
                return 2.0
            case _:
                raise Exception(
                    f"Unknown compression value `{self.compression}` of type `{type(self.compression)}`"
                )


@dataclass
class ConvParams:
    # Dataset specific params
    in_channels: int
    in_dimensions: int
    out_height: int

    layers: list[ConvLayerParams]

    reste_threshold: float
    reste_o: float

    activation: ActivationParams

    # Other
    dropout_rate: float = 0.0

    def get_conv_complexity(self) -> float:
        complexity = 0

        in_dimensions = self.in_dimensions
        in_channels = self.in_channels
        for layer in self.layers:
            reduce_image_by = layer.kernel_size // 2 + layer.padding
            out_dimensions = (in_dimensions - reduce_image_by) // layer.stride
            out_channels = layer.channels

            # Convolution complexity
            mac_ops_per_out_channel = in_channels * out_dimensions**2
            mac_ops = mac_ops_per_out_channel * out_channels
            complexity += mac_ops * layer.get_conv_complexity_coefficient()

            # Activation complexity
            out_size = out_dimensions**2 * out_channels
            complexity += out_size * self.activation.get_activation_complexity_coef()

            # Pooling compexity
            if layer.add_max_pooling():
                # Accumulate
                acc_ops = (out_dimensions // 2) ** 2
                complexity += acc_ops

            in_dimensions = out_dimensions
            in_channels = out_channels

        return complexity


@dataclass
class CNNParams:
    conv: ConvParams
    fc: FCParams
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
        self.quantize_input = (
            Quantize(QMode.DET, p.in_bitwidth) if p.in_bitwidth < 32 else nn.Identity()
        )

        self.conv_layers = self.build_conv_layers(p)

        fc_in_height = self._get_fc_in_height(p, self.conv_layers)

        self.fc_layers = self.build_fc_layers(p, fc_in_height)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantize_input(x)

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
        flattened_size = x.reshape(x.shape[0], -1).size(1)
        logger.info(f"Input shape: {x.shape}, equating to {flattened_size} inputs")

        for layer in self.conv_layers:
            x = layer(x)
            flattened_size = x.reshape(x.shape[0], -1).size(1)
            logger.info(
                f"Next layer shape: {x.shape}, equating to {flattened_size} inputs"
            )

        # Flatten the conv output
        flattened = x.reshape(x.shape[0], -1)
        logger.info(f"FC input size is {flattened.size(1)}")

    @torch.no_grad()
    def get_conv_layers_sizes(self) -> list[int]:
        # Forward pass dummy input through convolutional layers
        dummy_input = torch.zeros(
            1,
            self.p.conv.in_channels,
            self.p.conv.in_dimensions,
            self.p.conv.in_dimensions,
        )

        # TODO: Remove influence of channels on these amounts. I only need the channel sizes
        x = dummy_input
        sizes = []
        for layer in self.conv_layers:
            x: torch.Tensor = layer(x)
            x_size = x.reshape(x.shape[0], -1).size(1)
            sizes.append(x_size)

        return sizes

    @classmethod
    def build_conv_layers(cls, p: CNNParams) -> nn.ModuleList:
        conv_layers = nn.ModuleList()

        in_channels = p.conv.in_channels
        for layer_params in p.conv.layers:
            layer = []

            ConvModule = layer_params.get_conv_module_cls()
            layer.append(
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
            layer.append(nn.BatchNorm2d(layer_params.channels))
            layer.append(p.conv.activation.get_activation_module())

            if layer_params.add_max_pooling():
                layer.append(
                    nn.MaxPool2d(
                        kernel_size=layer_params.pooling_kernel_size,
                        stride=layer_params.pooling_kernel_size,
                    )
                )

            conv_layers.append(nn.Sequential(*layer))

            in_channels = layer_params.channels

        return conv_layers

    @classmethod
    def build_fc_layers(cls, p: CNNParams, fc_in_height: int) -> nn.Sequential:
        if len(p.fc.layers) < 2:
            raise Exception("Model can't have negative less than 2 layers")

        layers = []

        last_layer_height = fc_in_height
        for hidden in p.fc.layers[:-1]:
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

    @staticmethod
    @torch.no_grad()
    def _get_fc_in_height(p: CNNParams, conv_layers: nn.ModuleList) -> int:
        # Forward pass dummy input through convolutional layers
        dummy_input = torch.zeros(
            8,
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
