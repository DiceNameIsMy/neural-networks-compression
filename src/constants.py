import logging
import os

import torch

import src

LOG_LEVEL = logging.INFO
SEED = 1

# Datasets default values
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# MLP default values
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATALOADERS_NUM_WORKERS = 0 if DEVICE.type == "cpu" else 4
LEARNING_RATE = 0.01
EPOCHS = 20

# Caching
PROJECT_HOME = os.path.dirname(os.path.dirname(src.__file__))
DATASETS_FOLDER = os.path.join(PROJECT_HOME, "datasets_cache")
MODELS_FOLDER = os.path.join(PROJECT_HOME, "models")
POPULATION_FOLDER = os.path.join(PROJECT_HOME, "populations")
