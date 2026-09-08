import torch
from torch import nn


class TorchCNN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        seed: int | None = None,
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.network = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(16 * 7 * 7, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)