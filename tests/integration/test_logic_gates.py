import pytest

from edgekit.datasets.logic_gates import LogicGatesDataset
from edgekit.models.perceptron import Perceptron
from edgekit.training.perceptron import train_perceptron


@pytest.mark.parametrize("gate", ["and", "or", "nand", "nor"])
def test_perceptron_learns_linear_gate(gate):
    dataset = LogicGatesDataset(f"data/logic_gates/{gate}.csv")
    model = Perceptron(n_inputs=2)

    train_perceptron(model, dataset, epochs=20, learning_rate=0.1)

    for x, expected in dataset:
        assert model.predict(x) == expected


def test_perceptron_cannot_learn_xor():
    dataset = LogicGatesDataset("data/logic_gates/xor.csv")
    model = Perceptron(n_inputs=2)

    train_perceptron(model, dataset, epochs=100, learning_rate=0.1)

    predictions = [model.predict(x) for x, _ in dataset]
    expected = [y for _, y in dataset]

    assert predictions != expected