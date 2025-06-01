import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import plotly.express as px

from src.constants import POPULATION_FOLDER, REPORTS_FOLDER
from src.datasets.dataset import CnnDataset
from src.datasets.mnist_dataset import MiniMNIST32x32Dataset, MiniMNISTDataset
from src.models.cnn import CNNParams, ConvLayerParams, ConvParams
from src.models.compression.enums import Activation, NNParamsCompMode, QMode
from src.models.eval import NNArchitectureEvaluator
from src.models.mlp import FCLayerParams, FCParams
from src.models.nn import ActivationParams, NNTrainParams

logger = logging.getLogger(__name__)


class ArchitectureBuilder(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_params(
        self,
        DatasetCls: type[CnnDataset],
        conv_compression: NNParamsCompMode,
        conv_bitwidth: int,
        conv_activation: ActivationParams,
        fc_compression: NNParamsCompMode,
        fc_bitwidth: int,
        fc_activation: ActivationParams,
    ) -> CNNParams:
        pass


class LeNet5Builder(ArchitectureBuilder):
    # Source: https://github.com/dvgodoy/dl-visuals/blob/main/Architectures/architecture_lenet.png

    def __init__(
        self, epochs: int = 1, early_stop_patience: int = 1, batch_size: int = 50
    ):
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.batch_size = batch_size

    def get_name(self) -> str:
        return "LeNet5"

    def get_params(
        self,
        DatasetCls: type[CnnDataset],
        conv_compression: NNParamsCompMode,
        conv_bitwidth: int,
        conv_activation: ActivationParams,
        fc_compression: NNParamsCompMode,
        fc_bitwidth: int,
        fc_activation: ActivationParams,
    ) -> CNNParams:
        cnn_train_loader, cnn_test_loader = DatasetCls.get_dataloaders(self.batch_size)

        # TODO: LeNet5 uses avg pool, but we use max pool here.
        conv_params = ConvParams(
            in_channels=DatasetCls.input_channels,
            in_dimensions=DatasetCls.input_dimensions,
            in_bitwidth=8,
            out_height=DatasetCls.output_size,
            layers=[
                ConvLayerParams(
                    channels=6,
                    kernel_size=5,
                    stride=1,
                    padding=0,
                    pooling_kernel_size=2,
                    compression=conv_compression,
                    bitwidth=conv_bitwidth,
                ),
                ConvLayerParams(
                    channels=16,
                    kernel_size=5,
                    stride=1,
                    pooling_kernel_size=2,
                    compression=conv_compression,
                    bitwidth=conv_bitwidth,
                ),
            ],
            reste_threshold=1.5,
            reste_o=3,
            activation=conv_activation,
            dropout_rate=0.0,
        )
        cnn_fc_params = FCParams(
            layers=[
                FCLayerParams(120, fc_compression, bitwidth=fc_bitwidth),
                FCLayerParams(84, fc_compression, bitwidth=fc_bitwidth),
                FCLayerParams(
                    DatasetCls.output_size, fc_compression, bitwidth=fc_bitwidth
                ),
            ],
            activation=fc_activation,
            qmode=QMode.DET,
            dropout_rate=0.0,
        )
        cnn_train_params = NNTrainParams(
            DatasetCls,
            cnn_train_loader,
            cnn_test_loader,
            epochs=self.epochs,
            learning_rate=0.001,
            weight_decay=0.00001,
            early_stop_patience=self.early_stop_patience,
        )

        in_bitwidth = None
        if conv_compression == NNParamsCompMode.NONE:
            in_bitwidth = 32
        elif conv_compression == NNParamsCompMode.NBITS:
            in_bitwidth = conv_bitwidth
        elif conv_compression in (NNParamsCompMode.BINARY, NNParamsCompMode.TERNARY):
            in_bitwidth = 1
        else:
            raise ValueError(f"Unknown compression mode: {conv_compression}")

        cnn_params = CNNParams(
            in_bitwidth=in_bitwidth,
            conv=conv_params,
            fc=cnn_fc_params,
            train=cnn_train_params,
        )
        return cnn_params


def get_prefix(output_folder: str | None = None) -> str:
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        return os.path.join(output_folder, "experiment1_")

    timestamp = datetime.now().replace(microsecond=0).isoformat()
    folder = os.path.join(REPORTS_FOLDER, timestamp)
    os.makedirs(folder, exist_ok=True)

    return os.path.join(folder, "experiment1_")


def run_experiment1(
    output_folder: str | None = None,
    evaluations: int = 1,
    epochs: int = 1,
    plot: bool = False,
):
    for DatasetCls in [MiniMNIST32x32Dataset]:
        df = run_on_dataset(DatasetCls, epochs, evaluations)

        output_prefix = get_prefix(output_folder)
        df.to_csv(output_prefix + "results.csv", index=False)

        if plot:
            plot_results(df)

        logger.info(f"Experiment 1 on dataset {DatasetCls.__name__} completed")

    logger.info("Experiment 1 completed")


def plot_results(df: pd.DataFrame):
    fig = px.parallel_categories(
        df.sort_values("mean", ascending=False),
        dimensions=[
            "conv_activation",
            "conv_compression",
            "fc_compression",
            "mean",
        ],
        color_continuous_scale=px.colors.sequential.Inferno,
    )
    fig.write_html(get_prefix() + "mean_accuracy.html")
    fig.show()

    fig = px.parallel_categories(
        df.sort_values("best", ascending=False),
        dimensions=[
            "conv_activation",
            "conv_compression",
            "fc_compression",
            "best",
        ],
        color_continuous_scale=px.colors.sequential.Inferno,
    )
    fig.write_html(get_prefix() + "best_accuracy.html")
    fig.show()


def run_on_dataset(
    DatasetCls: type[CnnDataset], epochs: int = 1, evaluations: int = 1
) -> pd.DataFrame:
    datapoints = []

    logger.info(f"Running experiment on {DatasetCls.__name__} dataset...")

    # Run training
    for architecture_builder in [LeNet5Builder(epochs, epochs, DatasetCls.batch_size)]:
        for activation in Activation:
            for conv_compression in NNParamsCompMode:
                if conv_compression == NNParamsCompMode.TERNARY:
                    continue

                full = evaluate_compression(
                    architecture_builder,
                    DatasetCls,
                    conv_compression,
                    8,
                    activation,
                    conv_compression,
                    8,
                    activation,
                    evaluations,
                )
                if full is not None:
                    datapoints.append(full)

    logger.info(
        f"Collected {len(datapoints)} datapoints for {DatasetCls.__name__} dataset"
    )

    return pd.DataFrame(datapoints)


def evaluate_compression(
    architecture_builder: ArchitectureBuilder,
    DatasetCls: type[CnnDataset],
    conv_compression: NNParamsCompMode,
    conv_bitwidth: int,
    conv_activation: Activation,
    fc_compression: NNParamsCompMode,
    fc_bitwidth: int,
    fc_activation: Activation,
    evaluate_times: int = 1,
) -> dict | None:
    logger.info(
        f"Evaluating {architecture_builder.get_name()} with "
        + f"{conv_compression=}, {conv_activation=}, {fc_compression=}, {fc_activation=}",
    )

    model_params = architecture_builder.get_params(
        DatasetCls=DatasetCls,
        conv_compression=conv_compression,
        conv_bitwidth=conv_bitwidth,
        conv_activation=ActivationParams(conv_activation),
        fc_compression=fc_compression,
        fc_bitwidth=fc_bitwidth,
        fc_activation=ActivationParams(fc_activation),
    )
    evaluator = NNArchitectureEvaluator(model_params.train)

    try:
        stats = evaluator.evaluate_accuracy(model_params, times=evaluate_times)
    except Exception:
        logger.info(
            f"Failed to evaluate {architecture_builder.get_name()} with"
            + f" {conv_compression=}, {conv_activation=}, {fc_compression=}, {fc_activation=}",
        )
        return None

    return {
        # Properties
        "architecture": "LeNet5",
        "dataset": MiniMNISTDataset.__name__,
        "conv_compression": conv_compression.name,
        "conv_activation": conv_activation.name,
        "fc_compression": fc_compression.name,
        "fc_activation": fc_activation.name,
        # Stats
        "best": stats["max"],
        "mean": stats["mean"],
        "accuracies": stats["accuracies"],
        "cost": evaluator.evaluate_complexity(model_params),
    }
