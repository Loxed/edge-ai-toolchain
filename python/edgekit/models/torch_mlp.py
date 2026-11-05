# python/edgekit/models/torch_mlp.py
#
# MLP PyTorch configurable par une liste de tailles, notation de l'ancien livret :
# {100, 8, 1} = 100 entrees, une couche cachee de 8 neurones, 1 sortie.

import torch
from torch import nn

ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
}


def make_mlp(sizes: list[int], activation: str = "tanh", out_activation: str | None = "sigmoid",
             seed: int | None = None) -> nn.Sequential:
    if len(sizes) < 2:
        raise ValueError("sizes doit contenir au moins une entree et une sortie")
    if seed is not None:
        torch.manual_seed(seed)
    layers: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        last = i == len(sizes) - 2
        if not last:
            layers.append(ACTIVATIONS[activation]())
        elif out_activation is not None:
            layers.append(ACTIVATIONS[out_activation]())
    return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_bytes(model: nn.Module) -> int:
    """Taille des poids et biais en memoire (float32 : 4 octets par parametre)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())
