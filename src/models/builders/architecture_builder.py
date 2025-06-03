from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.datasets.dataset import CnnDataset
from src.models.cnn import CNNParams, ConvLayerParams, ConvParams
from src.models.compression.enums import NNParamsCompMode, QMode
from src.models.mlp import FCLayerParams, FCParams
from src.models.nn import ActivationParams, NNTrainParams


@dataclass
class BuilderParams:
    conv_compression: NNParamsCompMode
    conv_bitwidth: int
    conv_activation: ActivationParams

    fc_compression: NNParamsCompMode
    fc_bitwidth: int
    fc_activation: ActivationParams

    # Training params
    DatasetCls: type[CnnDataset]
    epochs: int = 1
    early_stop_patience: int = 1
    batch_size: int = 50
    evaluate_times: int = 1


class ArchitectureBuilder(ABC):
    name: str = "Unset"
    p: BuilderParams

    def __init__(self, params: BuilderParams):
        self.p = params

    @abstractmethod
    def get_params(self) -> CNNParams:
        pass

    def _build_conv_params(self, layers: list[ConvLayerParams]) -> ConvParams:
        return ConvParams(
            in_channels=self.p.DatasetCls.input_channels,
            in_dimensions=self.p.DatasetCls.input_dimensions,
            out_height=self.p.DatasetCls.output_size,
            layers=layers,
            reste_threshold=1.5,
            reste_o=3,
            activation=self.p.conv_activation,
            dropout_rate=0.0,
        )

    def _build_fc_params(self, layers: list[FCLayerParams]) -> FCParams:
        return FCParams(
            layers=layers,
            activation=self.p.fc_activation,
            qmode=QMode.DET,
            dropout_rate=0.0,
        )

    def _build_train_params(self) -> NNTrainParams:
        train_loader, test_loader = self.p.DatasetCls.get_dataloaders(self.p.batch_size)
        return NNTrainParams(
            self.p.DatasetCls,
            train_loader,
            test_loader,
            batch_size=self.p.batch_size,
            epochs=self.p.epochs,
            learning_rate=0.001,
            weight_decay=0.00001,
            early_stop_patience=self.p.early_stop_patience,
        )

    def _get_input_bitwidth(self) -> int:
        if self.p.conv_compression == NNParamsCompMode.NONE:
            return 32
        elif self.p.conv_compression == NNParamsCompMode.NBITS:
            return self.p.conv_bitwidth
        elif self.p.conv_compression in (
            NNParamsCompMode.BINARY,
            NNParamsCompMode.TERNARY,
        ):
            return 1
        else:
            raise ValueError(f"Unknown compression mode: {self.p.conv_compression}")
