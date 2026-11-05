# python/edgekit/training/torch_loop.py
#
# Boucles d'entrainement PyTorch partagees par les notebooks des TP 4 a 7.

import time
from dataclasses import dataclass, field

import torch
from torch import nn


def get_device() -> torch.device:
    """GPU s'il y en a un visible, sinon CPU. EDGEKIT_DEVICE=cpu force le CPU (test du cas élève sans GPU)."""
    import os

    if os.environ.get("EDGEKIT_DEVICE", "").lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class MSERun:
    epochs: int
    seconds: float
    reached: bool
    history: list[float] = field(default_factory=list)


def train_until_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, target_mse: float,
                    lr: float = 0.05, max_epochs: int = 20_000, momentum: float = 0.9) -> MSERun:
    """Descente de gradient sur tout le jeu (petits datasets) jusqu'a MSE < target_mse.

    Reprend le critere d'arret de l'ancienne bibliotheque NeuralNetwork (seuil sur la MSE),
    pour comparer des architectures par le temps et le nombre d'epochs necessaires.
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.MSELoss()
    y = y.reshape(-1, 1) if y.dim() == 1 else y
    history = []
    _sync(x.device)
    t0 = time.perf_counter()
    reached = False
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        history.append(loss.item())
        if history[-1] < target_mse:
            reached = True
            break
    _sync(x.device)
    return MSERun(epochs=epoch, seconds=time.perf_counter() - t0, reached=reached, history=history)


@dataclass
class ClassifierRun:
    seconds: float
    train_loss: list[float] = field(default_factory=list)
    test_accuracy: list[float] = field(default_factory=list)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(dim=1) == yb).sum().item()
        total += yb.numel()
    return correct / total


def _freeze_batchnorm(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_classifier(model: nn.Module, train_loader, test_loader, epochs: int, device: torch.device,
                     lr: float = 1e-3, log=print, freeze_bn: bool = False) -> ClassifierRun:
    """Entrainement classique : Adam + entropie croisee, une evaluation par epoch.

    freeze_bn=True garde les BatchNorm en mode evaluation (statistiques figees), recommande pour
    le fine-tuning d'un reseau pre-entraine sur un petit dataset.
    """
    model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    run = ClassifierRun(seconds=0.0)
    _sync(device)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_bn:
            _freeze_batchnorm(model)
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * yb.numel()
            n += yb.numel()
        run.train_loss.append(total / n)
        run.test_accuracy.append(evaluate_accuracy(model, test_loader, device))
        if log is not None:
            log(f"  epoch {epoch:2d} : loss = {run.train_loss[-1]:.4f}, précision test = {run.test_accuracy[-1]:.2%}")
    _sync(device)
    run.seconds = time.perf_counter() - t0
    return run


@torch.no_grad()
def measure_latency_ms(model: nn.Module, example: torch.Tensor, n: int = 200) -> float:
    """Latence moyenne d'une inference (ms), apres 10 appels de chauffe."""
    model.eval()
    for _ in range(10):
        model(example)
    _sync(example.device)
    t0 = time.perf_counter()
    for _ in range(n):
        model(example)
    _sync(example.device)
    return (time.perf_counter() - t0) / n * 1e3
