from edgekit.models.perceptron import Perceptron


def test_perceptron_predicts_from_known_weights():
    model = Perceptron(n_inputs=2)
    model.weights = [1.0, 1.0]
    model.bias = -1.5

    assert model.predict([0, 0]) == 0
    assert model.predict([1, 1]) == 1