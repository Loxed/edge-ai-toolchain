# tests/unit/test_or.py

from edgekit.datasets.logic_gates import LogicGatesDataset
from edgekit.models.perceptron import Perceptron
from edgekit.training.perceptron import train_perceptron


# Test the perceptron model on the OR dataset
def test_perceptron_learns_or():
    dataset = LogicGatesDataset("data/logic_gates/or.csv")
    model = Perceptron(n_inputs=2)

    train_perceptron(model, dataset, epochs=20, learning_rate=0.1)

    for x, expected in dataset:
        assert model.predict(x) == expected