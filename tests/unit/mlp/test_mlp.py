from edgekit.models.mlp import MLP


def test_mlp_forward_returns_expected_output_size():
    model = MLP(n_inputs=2, n_hidden=4, n_outputs=1, seed=42)

    output = model.forward([0, 1])

    assert len(output) == 1


def test_mlp_output_is_between_zero_and_one():
    model = MLP(n_inputs=2, n_hidden=4, n_outputs=1, seed=42)

    output = model.forward([0, 1])

    assert 0.0 <= output[0] <= 1.0


def test_mlp_initialization_is_reproducible_with_seed():
    model_a = MLP(n_inputs=2, n_hidden=4, n_outputs=1, seed=42)
    model_b = MLP(n_inputs=2, n_hidden=4, n_outputs=1, seed=42)

    assert model_a.forward([0, 1]) == model_b.forward([0, 1])