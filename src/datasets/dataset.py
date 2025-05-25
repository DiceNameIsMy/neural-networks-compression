import logging
import os
import pickle
from functools import wraps
from typing import Any

import torch
from sklearn.model_selection import train_test_split
from torch.utils import data

from src.constants import BATCH_SIZE, DATASETS_FOLDER, SEED, VALIDATION_SPLIT

logger = logging.getLogger(__name__)


class Dataset(data.Dataset):
    batch_size = BATCH_SIZE

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @classmethod
    def get_xy(cls) -> tuple[Any, torch.Tensor]:
        raise NotImplementedError("get_xy is not implemented")

    @classmethod
    def get_dataloaders(
        cls, batch_size: int | None = None
    ) -> tuple[data.DataLoader, data.DataLoader]:
        raise NotImplementedError("get_dataloaders is not implemented")

    @classmethod
    def get_dataloaders_from_xy(
        cls, X, y, batch_size: int | None
    ) -> tuple[data.DataLoader, data.DataLoader]:
        if batch_size is None:
            batch_size = cls.batch_size

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, random_state=SEED
        )
        train_loader = data.DataLoader(
            cls(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        test_loader = data.DataLoader(
            cls(X_test, y_test), batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader


class MlpDataset(Dataset):
    input_size: int
    output_size: int


class CnnDataset(Dataset):
    input_channels: int
    input_dimensions: int
    input_size: int
    output_size: int


def cache_to_file(name: str, cache_dir=DATASETS_FOLDER):
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            # Create cache directory if it does not exist
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            cache_file = os.path.join(cache_dir, f"{name}_cache.pkl")

            # Load from cache if exists
            if os.path.exists(cache_file):
                logger.info(f"Loading cached {name} from {cache_file}")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)

            # Save to cache
            with open(cache_file, "wb") as f:
                logger.info(f"Caching {name} to {cache_file}")
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator
