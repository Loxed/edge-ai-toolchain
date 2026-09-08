import torch
from torch import nn


class TorchMLP(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        seed: int | None = None,
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.network = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)