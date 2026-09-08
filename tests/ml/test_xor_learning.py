from edgekit.datasets.logic_gates import LogicGatesDataset
from edgekit.models.mlp import MLP
from edgekit.training.mlp import train_mlp


def test_mlp_learns_xor():
    dataset = LogicGatesDataset("data/logic_gates/xor.csv")

    model = MLP(n_inputs=2, n_hidden=4, n_outputs=1, seed=42)

    history = train_mlp(
        model,
        dataset,
        epochs=1000,
        learning_rate=0.5,
    )


    predictions = [model.predict(x) for x, _ in dataset]
    expected = [y for _, y in dataset]

    assert predictions == expected
    assert history["loss"][-1] < history["loss"][0]