from edgekit.datasets.logic_gates import LogicGatesDataset
from edgekit.datasets.synthetic import sample_noisy_points
from edgekit.models.mlp import MLP
from edgekit.training.mlp import train_mlp


def test_mlp_learns_noisy_xor_cloud():
    # Reglages du notebook 02. Sans melange des echantillons, la precision plafonnait a 85 %.
    corners = LogicGatesDataset("data/logic_gates/xor.csv")
    dataset = sample_noisy_points(corners, n_per_class=30, noise_std=0.05, seed=0)
    model = MLP(n_inputs=2, n_hidden=4, n_outputs=1, seed=42)
    train_mlp(model, dataset, epochs=1000, learning_rate=0.1)
    accuracy = sum(model.predict(x) == y for x, y in dataset) / len(dataset)
    assert accuracy > 0.9
