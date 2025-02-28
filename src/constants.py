import torch

SEED = 1

# Datasets default values
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# MLP default values
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.01
EPOCHS = 20
