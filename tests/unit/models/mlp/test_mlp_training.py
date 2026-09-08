from edgekit.models.mlp import MLP
from edgekit.training.mlp import train_mlp


def test_mlp_training_returns_history():
    dataset = [
        ([0, 0], 0),
        ([1, 1], 1),
    ]

    model = MLP(n_inputs=2, n_hidden=4, n_outputs=1, seed=42)

    history = train_mlp(
        model,
        dataset,
        epochs=10,
        learning_rate=0.5,
    )

    assert "loss" in history
    assert len(history["loss"]) == 10


def test_mlp_training_reduces_loss():
    dataset = [
        ([0, 0], 0),
        ([1, 1], 1),
    ]

    model = MLP(n_inputs=2, n_hidden=4, n_outputs=1, seed=42)

    history = train_mlp(
        model,
        dataset,
        epochs=100,
        learning_rate=0.5,
    )

    assert history["loss"][-1] < history["loss"][0]