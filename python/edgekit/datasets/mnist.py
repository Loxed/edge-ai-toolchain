from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="data/mnist",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="data/mnist",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader