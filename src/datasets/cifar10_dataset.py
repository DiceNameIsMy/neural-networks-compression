import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from src.constants import DATASETS_FOLDER
from src.datasets.dataset import CnnDataset


def fetch_cifar10_dataset():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATASETS_FOLDER, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATASETS_FOLDER, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


class CIFAR10Dataset(CnnDataset):
    batch_size: int = 128
    input_channels: int = 3
    input_dimensions: int = 32
    input_size: int = 32 * 32 * 3
    output_size: int = 10

    @classmethod
    def get_xy(cls) -> tuple[torch.Tensor, torch.Tensor]:
        train, test = fetch_cifar10_dataset()

        X = np.concatenate((train.data, test.data))
        X = torch.tensor(X).permute(0, 3, 1, 2).float() / 255.0
        X = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(X)

        y = torch.tensor(np.concatenate((train.targets, test.targets))).int()

        return X, y

    @classmethod
    def get_dataloaders(cls, batch_size: int | None = None):
        if batch_size is None:
            batch_size = cls.batch_size

        X, y = cls.get_xy()
        return cls.get_dataloaders_from_xy(X, y, batch_size)


class MiniCIFAR10Dataset(CIFAR10Dataset):

    @classmethod
    def get_xy(cls) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = super().get_xy()
        X = X[:4000]
        y = y[:4000]
        return X, y
