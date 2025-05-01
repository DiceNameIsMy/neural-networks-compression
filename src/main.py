import logging

import matplotlib.pyplot as plt

from src.models.mlp import FCLayerParams
from src.models.nn import ActivationParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    from src.datasets.cardio_dataset import CardioDataset
    from src.models.mlp import MLPEvaluator, MLPParams
    from src.models.quant.enums import ActivationModule, QMode

    DatasetClass = CardioDataset
    p = MLPParams(
        layers=[
            FCLayerParams(DatasetClass.input_size, bitwidth=8),
            FCLayerParams(64, bitwidth=16),
            FCLayerParams(DatasetClass.output_size, bitwidth=8),
        ],
        activation=ActivationParams(
            activation=ActivationModule.RELU,
            binary_quantization_mode=QMode.DET,
        ),
        learning_rate=0.01,
        epochs=100,
        dropout_rate=0.2,
        quantization_mode=QMode.DET,
    )
    train_loader, test_loader = CardioDataset.get_dataloaders(batch_size=64)

    evaluator = MLPEvaluator(train_loader, test_loader)
    # best_acc = evaluator.train_model(p)
    # train_log = evaluator.train_log.copy()

    # for log in train_log:
    #     print(
    #         f"Epoch: {log["epoch"]}, loss: {log["loss"]:.4f}, acc: {log["accuracy"]:.4f}"
    #     )

    # print(f"Best accuracy: {best_acc:.4f}")

    accuracies = []
    for _ in range(10):
        best_acc = evaluator.train_model(p)
        accuracies.append(best_acc)

    plt.hist(accuracies, bins=10, edgecolor="black")
    plt.title("Histogram of Accuracies")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.savefig("accuracy_histogram_cardio.png")


if __name__ == "__main__":
    main()
