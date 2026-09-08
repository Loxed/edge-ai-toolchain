# tests/unit/test_and.py

from edgekit.datasets.logic_gates import LogicGatesDataset
from edgekit.models.perceptron import Perceptron
from edgekit.training.perceptron import train_perceptron


# Test the AND dataset
def test_and_dataset():
    dataset = LogicGatesDataset("data/logic_gates/and.csv")

    assert len(dataset) == 4
    assert dataset[0] == ([0, 0], 0)
    assert dataset[1] == ([0, 1], 0)
    assert dataset[2] == ([1, 0], 0)
    assert dataset[3] == ([1, 1], 1)


# Test the Perceptron model on the AND dataset
def test_perceptron_learns_and():
    dataset = LogicGatesDataset("data/logic_gates/and.csv")
    model = Perceptron(n_inputs=2)

    train_perceptron(model, dataset, epochs=20, learning_rate=0.1)

    for x, expected in dataset:
        assert model.predict(x) == expected