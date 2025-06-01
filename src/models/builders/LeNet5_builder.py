from src.models.builders.architecture_builder import ArchitectureBuilder
from src.models.cnn import CNNParams, ConvLayerParams
from src.models.mlp import FCLayerParams


class LeNet5Builder(ArchitectureBuilder):
    # Source: https://github.com/dvgodoy/dl-visuals/blob/main/Architectures/architecture_lenet.png

    def get_name(self) -> str:
        return "LeNet5"

    def get_params(self) -> CNNParams:
        train_loader, test_loader = self.p.DatasetCls.get_dataloaders(self.p.batch_size)

        # TODO: LeNet5 uses avg pool, but we use max pool here.
        conv_layers = [
            ConvLayerParams(
                channels=6,
                kernel_size=5,
                stride=1,
                padding=0,
                pooling_kernel_size=2,
                compression=self.p.conv_compression,
                bitwidth=self.p.conv_bitwidth,
            ),
            ConvLayerParams(
                channels=16,
                kernel_size=5,
                stride=1,
                pooling_kernel_size=2,
                compression=self.p.conv_compression,
                bitwidth=self.p.conv_bitwidth,
            ),
        ]
        fc_layers = [
            FCLayerParams(120, self.p.fc_compression, bitwidth=self.p.fc_bitwidth),
            FCLayerParams(84, self.p.fc_compression, bitwidth=self.p.fc_bitwidth),
            FCLayerParams(
                self.p.DatasetCls.output_size,
                self.p.fc_compression,
                bitwidth=self.p.fc_bitwidth,
            ),
        ]

        conv_params = self._build_conv_params(conv_layers)
        fc_params = self._build_fc_params(fc_layers)

        train_params = self._build_train_params()
        in_bitwidth = self._get_input_bitwidth()

        return CNNParams(
            in_bitwidth=in_bitwidth,
            conv=conv_params,
            fc=fc_params,
            train=train_params,
        )
