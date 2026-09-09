# python/edgekit/training/perceptron.py

import random

from edgekit.models.perceptron import Perceptron


def _count_correct(model: Perceptron, samples: list[tuple[list[float], int]]) -> int:
    return sum(1 for x, target in samples if model.predict(x) == target)


def train_perceptron(
    model: Perceptron,
    dataset,
    epochs: int = 20,
    learning_rate: float = 0.1,
    seed: int = 0,
    keep_best: bool = True,
) -> None:
    # Shuffling matters once a dataset has more than a handful of samples:
    # without it, same-label examples processed back to back push the
    # weights hard in one direction, then just as hard the other way for
    # the next label's block, which can prevent convergence even on
    # linearly separable data.
    samples = list(dataset)
    rng = random.Random(seed)

    # "Pocket" algorithm (Gallant, 1990): once the data isn't perfectly
    # separable (which noisy data almost never is), the plain perceptron
    # update rule has no convergence guarantee, it keeps correcting
    # forever and can oscillate between good and bad solutions epoch to
    # epoch. Tracking the best weights seen so far and restoring them at
    # the end avoids reporting whatever the last epoch happened to land on.
    best_weights = model.weights.copy()
    best_bias = model.bias
    best_correct = _count_correct(model, samples)

    for _ in range(epochs):
        rng.shuffle(samples)

        for x, target in samples:
            prediction = model.predict(x)
            error = target - prediction

            for i in range(len(model.weights)):
                model.weights[i] += learning_rate * error * x[i]

            model.bias += learning_rate * error

        if keep_best:
            correct = _count_correct(model, samples)
            if correct > best_correct:
                best_correct = correct
                best_weights = model.weights.copy()
                best_bias = model.bias

    if keep_best:
        model.weights = best_weights
        model.bias = best_bias