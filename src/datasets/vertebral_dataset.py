import numpy as np
from ucimlrepo import fetch_ucirepo

from src.datasets.dataset import MlpDataset, cache_to_file


@cache_to_file(name="vertebral")
def fetch_vertebral_dataset():
    vertebral_column = fetch_ucirepo(id=212)
    vertebral_X = np.array(vertebral_column.data.features)

    vertebral_y_class = vertebral_column.data.targets["class"]
    vertebral_y = np.zeros(len(vertebral_y_class), dtype=int)
    vertebral_y[vertebral_y_class == "Hernia"] = 0
    vertebral_y[vertebral_y_class == "Spondylolisthesis"] = 1
    vertebral_y[vertebral_y_class == "Normal"] = 2

    return vertebral_X, vertebral_y


vertebral_X, vertebral_y = fetch_vertebral_dataset()


class VertebralDataset(MlpDataset):
    input_size: int = len(vertebral_X[0])
    output_size: int = len(np.unique(vertebral_y))

    @classmethod
    def get_xy(cls) -> tuple[np.ndarray, np.ndarray]:
        return vertebral_X, vertebral_y

    @classmethod
    def get_dataloaders(cls, batch_size: int | None = None):
        X, y = cls.get_xy()
        return cls.get_dataloaders_from_xy(X, y, batch_size)
