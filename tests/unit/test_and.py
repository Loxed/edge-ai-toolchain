from edgekit.datasets.logic_gates import LogicGatesDataset
from edgekit.models.perceptron import Perceptron


def test_and_dataset():
    dataset = LogicGatesDataset("data/logic_gates/and.csv")

    assert len(dataset) == 4
    assert dataset[0] == ([0, 0], 0)
    assert dataset[1] == ([0, 1], 0)
    assert dataset[2] == ([1, 0], 0)
    assert dataset[3] == ([1, 1], 1)


def test_perceptron_learns_and():
    dataset = LogicGatesDataset("data/logic_gates/and.csv")
    model = Perceptron(n_inputs=2)

    model.fit(dataset, epochs=20)

    for x, expected in dataset:
        assert model.predict(x) == expected