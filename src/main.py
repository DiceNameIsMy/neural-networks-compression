from src.datasets.vertebral_dataset import VertebralDataset
from src.mlp import ModelParams, test_model
from src.quantization import ActivationFunc, QMode


p = ModelParams(
    input_size=VertebralDataset.input_size,
    input_bitwidth=32,
    output_size=VertebralDataset.output_size,
    hidden_size=32,
    hidden_bitwidth=32,
    model_layers=2,
    learning_rate=0.01,
    activation=ActivationFunc.RELU,
    epochs=10,
    dropout_rate=0.0,
    quantization_mode=QMode.DET,
)
train_loader, test_loader = VertebralDataset.get_dataloaders()

test_model(p, train_loader, test_loader, verbose=2)
