# python/edgekit/datasets/imagefolder.py
#
# Loaders pour un dataset image au format ImageFolder (<root>/<split>/<classe>/*.png), avec la
# normalisation ImageNet attendue par les reseaux pre-entraines de torchvision.

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from edgekit.models.mobilenet import IMAGENET_MEAN, IMAGENET_STD


def make_transforms(size: int = 96, augment: bool = False):
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not augment:
        return transforms.Compose([transforms.Resize((size, size))] + norm)
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ] + norm)


def fixed_subset_loader(dataset, n: int, seed: int = 0, batch_size: int = 128) -> DataLoader:
    """n images tirees une fois pour toutes (graine `seed`), toujours dans le meme ordre.

    A utiliser pour la calibration (PTQ) et la mesure d'importance (pruning) : un loader melange
    donnerait un jeu d'images different a chaque parcours.
    """
    from torch.utils.data import Subset

    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    return DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=False)


def imagefolder_loaders(root: str | Path, size: int = 96, batch_size: int = 32, augment: bool = True,
                        seed: int = 0):
    """Loaders train / val / test et la liste des classes."""
    root = Path(root)
    train = datasets.ImageFolder(root / "train", transform=make_transforms(size, augment))
    val = datasets.ImageFolder(root / "val", transform=make_transforms(size))
    test = datasets.ImageFolder(root / "test", transform=make_transforms(size))
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(val, batch_size=128),
            DataLoader(test, batch_size=128),
            train.classes)
