# python/edgekit/training/perceptron.py

from edgekit.models.perceptron import Perceptron


def train_perceptron(
    model: Perceptron,
    dataset,
    epochs: int = 20,
    learning_rate: float = 0.1,
) -> None:
    for _ in range(epochs):
        for x, target in dataset:
            prediction = model.predict(x)
            error = target - prediction

            for i in range(len(model.weights)):
                model.weights[i] += learning_rate * error * x[i]

            model.bias += learning_rate * error