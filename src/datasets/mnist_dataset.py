import torch
import torchvision
import torchvision.transforms as transforms

from src.constants import DATASETS_FOLDER
from src.datasets.dataset import CnnDataset


def fetch_mnist_dataset():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # transforms.Lambda(lambda x: torch.flatten(x, 1)), # Flatten
        ]
    )

    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root=DATASETS_FOLDER, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=DATASETS_FOLDER, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


class MNISTDataset(CnnDataset):
    batch_size: int = 128
    input_channels: int = 1
    input_dimensions: int = 28
    input_size: int = 28 * 28
    output_size: int = 10  # 10 digit classes

    @classmethod
    def get_xy(cls) -> tuple[torch.Tensor, torch.Tensor]:
        """Get MNIST dataset as numpy arrays"""
        train_dataset, test_dataset = fetch_mnist_dataset()

        X = torch.cat((train_dataset.data, test_dataset.data)).unsqueeze(1).float()
        X = transforms.Normalize((0.1307,), (0.3081,))(X)

        y = torch.cat((train_dataset.targets, test_dataset.targets)).int()
        return X, y

    @classmethod
    def get_dataloaders(cls, batch_size: int | None = None):
        """Load MNIST dataset and create dataloaders"""

        if batch_size is None:
            batch_size = cls.batch_size

        X, y = cls.get_xy()

        return cls.get_dataloaders_from_xy(X, y, batch_size)


class MiniMNISTDataset(MNISTDataset):

    @classmethod
    def get_xy(cls) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = super().get_xy()
        X = X[:4000]
        y = y[:4000]
        return X, y
