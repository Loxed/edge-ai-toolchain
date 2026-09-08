# tests/unit/test_xor.py

from edgekit.datasets.logic_gates import LogicGatesDataset

from edgekit.models.perceptron import Perceptron
from edgekit.training.perceptron import train_perceptron

# from edgekit.models.mlp import MLP

# Test the perceptron model on the XOR dataset (it should not be able to learn it)
def test_perceptron_cannot_learn_xor():
    dataset = LogicGatesDataset("data/logic_gates/xor.csv")
    model = Perceptron(n_inputs=2)

    train_perceptron(model, dataset, epochs=100, learning_rate=0.1)

    predictions = [model.predict(x) for x, _ in dataset]
    expected = [y for _, y in dataset]

    assert predictions != expected

# Test the MLP model on the XOR dataset (it should be able to learn it)
# def test_mlp_learns_xor():
#     dataset = LogicGatesDataset("data/logic_gates/xor.csv")
#     model = MLP(n_inputs=2, n_hidden=2, n_outputs=1)

#     train_mlp(model, dataset, epochs=1000, learning_rate=0.1)

#     for x, expected in dataset:
#         assert model.predict(x) == expected