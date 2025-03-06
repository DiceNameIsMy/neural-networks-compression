import os
import pickle
from functools import wraps

import torch
from sklearn.model_selection import train_test_split
from torch.utils import data

from constants import BATCH_SIZE, SEED, VALIDATION_SPLIT


class Dataset(data.Dataset):
    input_size: int = None
    output_size: int = None

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @classmethod
    def _get_dataloaders(
        cls, X, y, batch_size=BATCH_SIZE
    ) -> tuple[data.DataLoader, data.DataLoader]:
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

    @classmethod
    def get_dataloaders(
        cls, batch_size=BATCH_SIZE
    ) -> tuple[data.DataLoader, data.DataLoader]:
        raise NotImplementedError("get_dataloaders is not implemented")


def cache_to_file(name: str, cache_dir="datasets_cache"):
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_file = os.path.join(cache_dir, f"{name}_cache.pkl")

            # Create cache directory if it does not exist
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            # Load from cache if exists
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)

            # Save to cache
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator
