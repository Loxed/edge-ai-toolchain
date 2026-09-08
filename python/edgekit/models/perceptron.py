# python/edgekit/models/perceptron.py

class Perceptron:
    def __init__(self, n_inputs: int):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0

    def predict(self, x: list[int]) -> int:
        activation = sum(
            weight * value
            for weight, value in zip(self.weights, x)
        ) + self.bias

        return int(activation > 0)