import torch

from edgekit.models.torch_mlp_xor import TorchMLP


def test_mlp_output_shape():
    model = TorchMLP(
        n_inputs=2,
        n_hidden=4,
        n_outputs=1,
        seed=42,
    )

    x = torch.zeros(4, 2)
    output = model(x)

    assert output.shape == (4, 1)


def test_mlp_parameter_count():
    """
        XOR

        Input 2
        Linear 2→4 = 2*4 + 4 = 12
        Linear 4→1 = 4*1 + 1 = 5

        Total = 17
    """

    model = TorchMLP(
        n_inputs=2,
        n_hidden=4,
        n_outputs=1,
        seed=42,
    )

    n_parameters = sum(p.numel() for p in model.parameters())

    assert n_parameters == 17


def test_mlp_seed_is_reproducible():
    model_a = TorchMLP(2, 4, 1, seed=42)
    model_b = TorchMLP(2, 4, 1, seed=42)

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(param_a, param_b)