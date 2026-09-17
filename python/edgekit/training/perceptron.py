# python/edgekit/training/perceptron.py

import random
from dataclasses import dataclass

from edgekit.models.perceptron import Perceptron


@dataclass
class PerceptronPocket:
    weights: list[float]
    bias: float
    correct: int
    epoch: int


def _count_correct(
    model: Perceptron,
    samples: list[tuple[list[float], int]],
) -> int:
    return sum(
        model.predict(x) == target
        for x, target in samples
    )


def train_perceptron(
    model: Perceptron,
    dataset,
    epochs: int = 20,
    learning_rate: float = 0.1,
    seed: int = 0,
    keep_best: bool = False,
) -> PerceptronPocket | None:
    samples = list(dataset)
    rng = random.Random(seed)

    pocket = None

    if keep_best:
        pocket = PerceptronPocket(
            weights=model.weights.copy(),
            bias=model.bias,
            correct=_count_correct(model, samples),
            epoch=0,
        )

    for epoch in range(1, epochs + 1):
        rng.shuffle(samples)

        for x, target in samples:
            prediction = model.predict(x)
            error = target - prediction

            for i in range(len(model.weights)):
                model.weights[i] += learning_rate * error * x[i]

            model.bias += learning_rate * error

        if keep_best:
            correct = _count_correct(model, samples)

            if pocket is None or correct > pocket.correct:
                pocket = PerceptronPocket(
                    weights=model.weights.copy(),
                    bias=model.bias,
                    correct=correct,
                    epoch=epoch,
                )

    return pocket