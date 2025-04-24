import numpy as np
from ucimlrepo import fetch_ucirepo

from src.datasets.dataset import Dataset, cache_to_file


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
