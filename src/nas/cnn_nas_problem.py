import logging

from src.datasets.dataset import CnnDataset
from src.models.cnn import CNNParams, ConvLayerParams, ConvParams
from src.models.compression.enums import QMode
from src.models.mlp import FCLayerParams, FCParams
from src.models.nn import ActivationParams, NNTrainParams
from src.nas.cnn_chromosome import CNNChromosome
from src.nas.nas_params import NasParams
from src.nas.nas_problem import NasProblem

logger = logging.getLogger(__name__)


class CnnNasProblem(NasProblem):
    ChromosomeCls: type[CNNChromosome]
    DatasetCls: type[CnnDataset]

    def __init__(self, params: NasParams, DatasetCls: type[CnnDataset]):
        super().__init__(params, DatasetCls, CNNChromosome)

    def get_nn_params(self, ch: CNNChromosome) -> CNNParams:
        activation = ActivationParams(
            activation=ch.activation,
            reste_o=ch.activation_reste_o,
            reste_threshold=ch.activation_reste_threshold,
            binary_qmode=ch.activation_qmode,
        )

        conv_layers = self._get_conv_layers(ch)
        conv_params = ConvParams(
            in_channels=self.DatasetCls.input_channels,
            in_dimensions=self.DatasetCls.input_dimensions,
            out_height=self.DatasetCls.output_size,
            layers=conv_layers,
            reste_threshold=ch.activation_reste_threshold,
            reste_o=ch.activation_reste_o,
            activation=activation,
            dropout_rate=ch.dropout,
        )

        fc_layers = self._get_fc_layers(ch)
        fc_params = FCParams(
            layers=fc_layers,
            activation=activation,
            qmode=QMode.DET,
            dropout_rate=ch.dropout,
        )
        train_params = NNTrainParams(
            DatasetCls=self.DatasetCls,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            epochs=self.p.epochs,
            learning_rate=ch.learning_rate,
            weight_decay=ch.weight_decay,
            early_stop_patience=self.p.patience,
        )
        return CNNParams(
            conv=conv_params,
            fc=fc_params,
            train=train_params,
            in_bitwidth=ch.in_bitwidth,
        )

    def _get_fc_layers(self, ch: CNNChromosome) -> list[FCLayerParams]:
        layers = []
        if ch.fc_layers >= 1:
            layers.append(
                FCLayerParams(ch.fc_height1, ch.fc_compression, ch.fc_bitwidth1)
            )
        if ch.fc_layers >= 2:
            layers.append(
                FCLayerParams(ch.fc_height2, ch.fc_compression, ch.fc_bitwidth2)
            )
        if ch.fc_layers >= 3:
            layers.append(
                FCLayerParams(ch.fc_height3, ch.fc_compression, ch.fc_output_bitwidth)
            )

        layers.append(
            FCLayerParams(
                self.DatasetCls.output_size, ch.fc_compression, ch.fc_output_bitwidth
            )
        )

        return layers

    def _get_conv_layers(self, ch: CNNChromosome) -> list[ConvLayerParams]:
        layers = []
        for i in range(1, ch.conv_layers + 1):
            layers.append(
                ConvLayerParams(
                    channels=getattr(ch, f"conv_channels{i}"),
                    kernel_size=3,
                    pooling_kernel_size=getattr(ch, f"conv_pooling_size{i}"),
                    stride=getattr(ch, f"conv_stride{i}"),
                    compression=ch.conv_compression,
                    bitwidth=getattr(ch, f"conv_compression_bitwidth{i}"),
                    bias=False,
                )
            )

        return layers
