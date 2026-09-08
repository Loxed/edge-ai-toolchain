import math
import random


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(y: float) -> float:
    # y is already sigmoid(x); the derivative wrt x simplifies to this
    return y * (1.0 - y)


class MLP:
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, seed: int | None = 0):
        rng = random.Random(seed)

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.weights_input_hidden = [
            [rng.uniform(-1.0, 1.0) for _ in range(n_inputs)]
            for _ in range(n_hidden)
        ]
        self.bias_hidden = [0.0] * n_hidden

        self.weights_hidden_output = [
            [rng.uniform(-1.0, 1.0) for _ in range(n_hidden)]
            for _ in range(n_outputs)
        ]
        self.bias_output = [0.0] * n_outputs

    def forward(self, x: list[float]) -> tuple[list[float], list[float]]:
        hidden = [
            sigmoid(
                sum(w * xi for w, xi in zip(self.weights_input_hidden[h], x))
                + self.bias_hidden[h]
            )
            for h in range(self.n_hidden)
        ]

        output = [
            sigmoid(
                sum(w * hi for w, hi in zip(self.weights_hidden_output[o], hidden))
                + self.bias_output[o]
            )
            for o in range(self.n_outputs)
        ]

        return hidden, output

    def predict(self, x: list[float]) -> int:
        _, output = self.forward(x)
        return int(output[0] > 0.5)