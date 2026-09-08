from edgekit.models.perceptron import Perceptron
from edgekit.training.perceptron import train_perceptron

# A simple dataset for testing
class TinyDataset:
    def __iter__(self):
        return iter([
            ([0, 0], 0),
            ([1, 1], 1),
        ])


def test_training_changes_weights():
    model = Perceptron(n_inputs=2)

    before = model.weights.copy()


    train_perceptron(model, TinyDataset(), epochs=1, learning_rate=0.1)

    assert model.weights != before