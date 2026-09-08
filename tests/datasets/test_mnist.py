import torch

from edgekit.datasets.mnist import get_mnist_loaders


def test_mnist_batch_shape():
    train_loader, test_loader = get_mnist_loaders(batch_size=4)

    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    assert train_images.shape == (4, 1, 28, 28)
    assert train_labels.shape == (4,)

    assert test_images.shape == (4, 1, 28, 28)
    assert test_labels.shape == (4,)


def test_mnist_batch_types():
    train_loader, _ = get_mnist_loaders(batch_size=4)

    images, labels = next(iter(train_loader))

    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64