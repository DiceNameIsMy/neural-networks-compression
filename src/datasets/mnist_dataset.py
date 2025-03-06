import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

from src.constants import BATCH_SIZE


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
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


class MNISTDataset(data.Dataset):
    input_channels: int = 1
    input_dimensions: int = 28
    input_size: int = 28 * 28
    output_size: int = 10  # 10 digit classes

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @classmethod
    def get_dataloaders(cls, batch_size=BATCH_SIZE):
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
