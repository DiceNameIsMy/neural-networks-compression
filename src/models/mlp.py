import logging
from dataclasses import dataclass

from torch import nn

from src.models.compression import binary, ternarize, weight_quant
from src.models.compression.enums import NNParamsCompMode, QMode
from src.models.compression.weight_quant import Quantize
from src.models.nn import ActivationParams

logger = logging.getLogger(__name__)


@dataclass
class FCLayerParams:
    height: int
    compression: NNParamsCompMode
    bitwidth: int = 32

    def get_fc_layer(self, in_height: int) -> nn.Linear:
        match self.compression:
            case NNParamsCompMode.NONE:
                return nn.Linear(in_height, self.height, bias=False)
            case NNParamsCompMode.NBITS:
                assert self.bitwidth > 0, "Bitwidth must be greater than 0"
                assert self.bitwidth < 32, "Bitwidth must be less than 32"
                return weight_quant.LinearQunatized(
                    self.bitwidth, in_height, self.height, bias=False
                )
            case NNParamsCompMode.BINARY:
                return binary.LinearBinary(in_height, self.height, bias=False)
            case NNParamsCompMode.TERNARY:
                return ternarize.LinearTernary(in_height, self.height, bias=False)
            case _:
                raise Exception(
                    "Unknown weight quantization mode: "
                    + f"{self.compression} of type {type(self.compression)}"
                )

    def get_compression_complexity_coef(self) -> float:
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
class FCParams:
    layers: list[FCLayerParams]
    activation: ActivationParams
    qmode: QMode = QMode.DET

    # Other
    dropout_rate: float = 0.0

    def get_complexity(self) -> float:
        complexity = 0

        prev_layer = self.layers[0]

        without_last_layer = self.layers[1:]

        for layer in without_last_layer:
            mac_ops = prev_layer.height * layer.height

            complexity += mac_ops * layer.get_compression_complexity_coef()
            complexity += (
                layer.height * self.activation.get_activation_complexity_coef()
            )

            prev_layer = layer

        return complexity


@dataclass
class MLPParams:
    fc: FCParams

    def get_model(self) -> "MLP":
        return MLP(self)


class MLP(nn.Module):
    def __init__(self, p: MLPParams):
        super().__init__()

        if len(p.fc.layers) < 2:
            raise Exception("Model can't have less than 2 layers")

        layers = []

        in_layer = p.fc.layers[0]
        if in_layer.bitwidth < 32:
            layers.append(Quantize(p.fc.qmode, in_layer.bitwidth))

        last_layer_height = in_layer.height
        for layer in p.fc.layers[1:]:
            layers.append(layer.get_fc_layer(last_layer_height))
            layers.append(nn.BatchNorm1d(layer.height))

            # Add dropout
            if p.fc.dropout_rate > 0:
                layers.append(nn.Dropout(p.fc.dropout_rate))

            # Add activation
            layers.append(p.fc.activation.get_activation_module())

            last_layer_height = layer.height

        # Combine all layers into Sequential model
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)

        # Apply softmax for better interpretability during training
        if self.training:
            x = nn.Softmax(dim=1)(x)

        return x
