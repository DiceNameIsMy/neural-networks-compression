import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

from src.constants import CACHE_FOLDER
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
        root=CACHE_FOLDER, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=CACHE_FOLDER, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


class MNISTDataset(CnnDataset):
    input_channels: int = 1
    input_dimensions: int = 28
    input_size: int = 28 * 28
    output_size: int = 10  # 10 digit classes

    @classmethod
    def get_dataloaders(cls, batch_size=256):
        """Load MNIST dataset and create dataloaders"""

        train_dataset, test_dataset = fetch_mnist_dataset()

        # Create dataloaders
        train_loader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        return train_loader, test_loader


class MiniMNISTDataset(MNISTDataset):

    @classmethod
    def get_dataloaders(cls, batch_size=128):
        """Load MNIST dataset and create dataloaders"""

        full_train_dataset, full_test_dataset = fetch_mnist_dataset()

        # Subset the datasets to return only a small part
        train_dataset = torch.utils.data.Subset(full_train_dataset, range(4000))
        test_dataset = torch.utils.data.Subset(full_test_dataset, range(800))

        # Create dataloaders
        train_loader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        return train_loader, test_loader
