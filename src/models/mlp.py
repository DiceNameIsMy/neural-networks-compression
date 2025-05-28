import logging
from dataclasses import dataclass

from torch import nn

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

    def get_model(self) -> "MLP":
        return MLP(self)


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
