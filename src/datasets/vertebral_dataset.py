import numpy as np
from ucimlrepo import fetch_ucirepo

import os
import pickle
from functools import wraps

from src.datasets.dataset import Dataset


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


@cache_to_file(name="vertebral")
def fetch_vertebral_dataset():
    vertebral_column = fetch_ucirepo(id=212)
    vertebral_X = np.array(vertebral_column.data.features)

    vertebral_y_class = vertebral_column.data.targets["class"]
    vertebral_y = np.zeros(
        (len(vertebral_y_class), len(vertebral_y_class.unique())), dtype=int
    )
    vertebral_y[vertebral_y_class == "Hernia", 0] = 1
    vertebral_y[vertebral_y_class == "Spondylolisthesis", 1] = 1
    vertebral_y[vertebral_y_class == "Normal", 2] = 1

    return vertebral_X, vertebral_y


vertebral_X, vertebral_y = fetch_vertebral_dataset()


class VertebralDataset(Dataset):
    input_size: int = len(vertebral_X[0])
    output_size: int = len(vertebral_y[0])

    @classmethod
    def get_dataloaders(cls, batch_size=32):
        return cls._get_dataloaders(vertebral_X, vertebral_y, batch_size)
