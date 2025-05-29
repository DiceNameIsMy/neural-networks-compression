import logging

from src.datasets.dataset import MlpDataset
from src.models.compression.enums import NNParamsCompMode
from src.models.mlp import FCLayerParams, FCParams, MLPParams
from src.models.nn import ActivationParams, NNTrainParams
from src.nas.mlp_chromosome import MLPChromosome
from src.nas.nas_params import NasParams
from src.nas.nas_problem import NasProblem

logger = logging.getLogger(__name__)


class MlpNasProblem(NasProblem):
    # TODO: It it better to use best accuracy instead of mean
    #       accuracy as an optimization goal? Or perhaps I should
    #       keep using both, but then train the final population
    #       again & show best accuracy?

    # NOTE: On every evaluation a new set of dataloaders is created.
    #       It's reduntant, although only a small % of NAS is spent on that.

    DatasetCls: type[MlpDataset]
    ChromosomeCls: type[MLPChromosome]

    def __init__(self, params: NasParams, DatasetCls: type[MlpDataset]):
        super().__init__(params, DatasetCls, MLPChromosome)

    def get_nn_params(self, ch: MLPChromosome) -> MLPParams:
        activation = ActivationParams(
            activation=ch.activation,
            reste_o=ch.reste_o,
            binary_qmode=ch.activation_binarization_mode,
        )
        layers = self._make_fc_layers(ch)
        fc_params = FCParams(
            layers=layers,
            activation=activation,
            qmode=ch.parameters_quantization_mode,
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
        return MLPParams(fc=fc_params, train=train_params)

    def _make_fc_layers(self, ch: MLPChromosome) -> list[FCLayerParams]:
        layers = []

        layers.append(
            FCLayerParams(
                self.DatasetCls.input_size, NNParamsCompMode.NBITS, ch.in_bitwidth
            )
        )
        if ch.hidden_layers >= 1:
            layers.append(
                FCLayerParams(ch.hidden_height1, ch.compression, ch.hidden_bitwidth1)
            )
        if ch.hidden_layers >= 2:
            layers.append(
                FCLayerParams(ch.hidden_height2, ch.compression, ch.hidden_bitwidth2)
            )
        if ch.hidden_layers >= 3:
            layers.append(
                FCLayerParams(ch.hidden_height3, ch.compression, ch.hidden_bitwidth3)
            )
        layers.append(
            FCLayerParams(
                self.DatasetCls.output_size, ch.compression, ch.output_bitwidth
            )
        )

        return layers
