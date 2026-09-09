import random


def sample_noisy_points(
    dataset,
    n_per_class: int = 20,
    low: float = 0.4,
    high: float = 0.6,
    noise_std: float = 0.05,
    seed: int = 0,
) -> list[tuple[list[float], int]]:
    """
    Turns a discrete corner dataset (every feature is 0 or 1) into a
    continuous cloud dataset: each corner becomes `n_per_class` points,
    centered at `low` (for a 0 feature) or `high` (for a 1 feature), with
    independent Gaussian noise added on every axis. The label of each
    generated point is copied unchanged from the corner it came from.
    """
    rng = random.Random(seed)
    samples: list[tuple[list[float], int]] = []

    for corner, label in dataset:
        center = [low if bit == 0 else high for bit in corner]

        for _ in range(n_per_class):
            point = [rng.gauss(c, noise_std) for c in center]
            samples.append((point, label))

    return samples