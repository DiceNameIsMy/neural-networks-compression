import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    from datasets.cardio_dataset import CardioDataset
    from models.mlp import MLPEvaluator, MLPParams
    from models.quant.enums import ActivationModule, QMode

    p = MLPParams(
        in_height=CardioDataset.input_size,
        in_bitwidth=32,
        out_height=CardioDataset.output_size,
        hidden_height=32,
        hidden_bitwidth=32,
        model_layers=4,
        learning_rate=0.01,
        activation=ActivationModule.RELU,
        epochs=100,
        dropout_rate=0.2,
        quantization_mode=QMode.DET,
    )
    train_loader, test_loader = CardioDataset.get_dataloaders(batch_size=128)

    evaluator = MLPEvaluator(train_loader, test_loader)
    best_acc = evaluator.train_model(p)
    train_log = evaluator.train_log.copy()

    for log in train_log:
        print(
            f"Epoch: {log["epoch"]}, loss: {log["loss"]:.4f}, acc: {log["accuracy"]:.4f}"
        )

    print(f"Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
