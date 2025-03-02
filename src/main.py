from datasets.vertebral_dataset import VertebralDataset
from models.mlp import ModelParams, test_model
from models.quantization import ActivationFunc, QMode

p = ModelParams(
    in_layer_height=VertebralDataset.input_size,
    in_bitwidth=32,
    out_height=VertebralDataset.output_size,
    hidden_height=32,
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
