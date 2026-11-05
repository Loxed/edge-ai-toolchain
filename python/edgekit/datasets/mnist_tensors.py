# python/edgekit/datasets/mnist_tensors.py
#
# MNIST complet charge une fois en tenseurs (60 000 + 10 000 images, ~200 Mo en float32),
# puis decoupe en batchs sans DataLoader : plus rapide sur GPU pour un petit modele, et
# deterministe avec une graine.

import torch
from torchvision import datasets


def mnist_tensors(root: str = "data/mnist", digits: tuple[int, ...] | None = None,
                  device: torch.device | str = "cpu"):
    """Retourne x_train, y_train, x_test, y_test. Images (N, 1, 28, 28) dans [0, 1].

    `digits` restreint aux chiffres donnes, relabellises 0, 1, ... dans l'ordre (ex. (1, 8)).
    """
    out = []
    for train in (True, False):
        ds = datasets.MNIST(root=root, train=train, download=True)
        x = ds.data.unsqueeze(1).float() / 255.0
        y = ds.targets.clone()
        if digits is not None:
            mask = torch.zeros_like(y, dtype=torch.bool)
            new_y = torch.empty_like(y)
            for i, d in enumerate(digits):
                mask |= y == d
                new_y[y == d] = i
            x, y = x[mask], new_y[mask]
        out += [x.to(device), y.to(device)]
    return tuple(out)


class TensorLoader:
    """Iterable de batchs (x, y) sur des tenseurs deja en memoire."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 128,
                 shuffle: bool = False, seed: int = 0):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, generator=self.gen) if self.shuffle else torch.arange(n)
        idx = idx.to(self.x.device)
        for i in range(0, n, self.batch_size):
            j = idx[i:i + self.batch_size]
            yield self.x[j], self.y[j]
