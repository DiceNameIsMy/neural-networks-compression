from src.models.builders.architecture_builder import ArchitectureBuilder
from src.models.cnn import CNNParams, ConvLayerParams
from src.models.mlp import FCLayerParams


class VGGNetBuilder(ArchitectureBuilder):
    # Source: TODO

    name = "VGGNet"

    def get_params(self) -> CNNParams:
        conv_layers = [
            ConvLayerParams(
                channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                pooling_kernel_size=1,
                compression=self.p.conv_compression,
                bitwidth=self.p.conv_bitwidth,
                bias=False,
            ),
            ConvLayerParams(
                channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                pooling_kernel_size=2,
                compression=self.p.conv_compression,
                bitwidth=self.p.conv_bitwidth,
                bias=False,
            ),
            ConvLayerParams(
                channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                pooling_kernel_size=1,
                compression=self.p.conv_compression,
                bitwidth=self.p.conv_bitwidth,
                bias=False,
            ),
            ConvLayerParams(
                channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                pooling_kernel_size=2,
                compression=self.p.conv_compression,
                bitwidth=self.p.conv_bitwidth,
                bias=False,
            ),
            ConvLayerParams(
                channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                pooling_kernel_size=1,
                compression=self.p.conv_compression,
                bitwidth=self.p.conv_bitwidth,
                bias=False,
            ),
            ConvLayerParams(
                channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                pooling_kernel_size=2,
                compression=self.p.conv_compression,
                bitwidth=self.p.conv_bitwidth,
                bias=False,
            ),
        ]
        fc_layers = [
            FCLayerParams(1024, self.p.fc_compression, bitwidth=self.p.fc_bitwidth),
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
