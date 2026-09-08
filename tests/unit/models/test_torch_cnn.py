import torch

from edgekit.models.torch_cnn import TorchCNN


def test_cnn_output_shape():
    model = TorchCNN(n_classes=10, seed=42)
    x = torch.zeros(4, 1, 28, 28)

    logits = model(x)

    assert logits.shape == (4, 10)


def test_cnn_parameter_count():
    """
        MNIST
        
        Input 1*28*28
        Conv 1→8, 3*3 = 8*1*3*3 + 8 = 80
        MaxPool
        Conv 8→16, 3*3 = 16*8*3*3 + 16 = 1168
        MaxPool
        Flatten
        Linear 16*7*7 → 10 = 16*7*7*10 + 10 = 7850

        Total = 80 + 1168 + 7850 = 9098
    """

    model = TorchCNN(n_classes=10, seed=42)

    n_parameters = sum(p.numel() for p in model.parameters())

    assert n_parameters == 9098


def test_cnn_seed_is_reproducible():
    model_a = TorchCNN(n_classes=10, seed=42)
    model_b = TorchCNN(n_classes=10, seed=42)

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(param_a, param_b)