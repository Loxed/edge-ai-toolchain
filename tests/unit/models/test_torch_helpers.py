import numpy as np
import torch

from edgekit.datasets.mnist_tensors import TensorLoader, mnist_tensors
from edgekit.datasets.old_labs import load_mnist_1v8, load_shapes, parse_c_array, parse_c_numbers
from edgekit.models.torch_mlp import count_parameters, make_mlp, model_size_bytes
from edgekit.training.torch_loop import train_until_mse


def test_parse_c_array_shape_and_comments():
    src = "// entete\nconst float a[2][3] = {\n {1, 2.5, -3}, // ligne\n {4, 5, 6}\n};\nconst float b[1] = {7};"
    np.testing.assert_array_equal(parse_c_array(src, "a"), [[1, 2.5, -3], [4, 5, 6]])
    np.testing.assert_array_equal(parse_c_array(src, "b"), [7])
    np.testing.assert_array_equal(parse_c_numbers("{1, /* x */ 2} // 3"), [1, 2])


def test_old_lab_arrays():
    x, y, xt = load_shapes()
    assert x.shape == (50, 100) and y.shape == (50,) and xt.shape == (8, 100)
    xm, ym, xmt, ymt = load_mnist_1v8()
    assert xm.shape == (50, 28, 28) and xmt.shape == (30, 28, 28)
    assert ym.sum() == 25 and ymt.sum() == 15
    assert 0.0 <= xm.min() and xm.max() <= 1.0


def test_make_mlp_counts():
    m = make_mlp([784, 25, 7, 1], activation="tanh", seed=0)
    assert count_parameters(m) == 784 * 25 + 25 + 25 * 7 + 7 + 7 + 1
    assert model_size_bytes(m) == 4 * count_parameters(m)
    assert m(torch.zeros(3, 784)).shape == (3, 1)


def test_train_until_mse_reaches_target_on_and_gate():
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([0.0, 0.0, 0.0, 1.0])
    run = train_until_mse(make_mlp([2, 4, 1], "tanh", seed=0), x, y, target_mse=0.01, lr=0.5)
    assert run.reached and run.history[-1] < 0.01


def test_tensor_loader_covers_everything_once():
    x = torch.arange(10).float().unsqueeze(1)
    loader = TensorLoader(x, torch.arange(10), batch_size=4, shuffle=True, seed=1)
    seen = torch.cat([yb for _, yb in loader])
    assert len(loader) == 3
    assert sorted(seen.tolist()) == list(range(10))


def test_mnist_tensors_digit_filter():
    x, y, xt, yt = mnist_tensors(digits=(1, 8))
    assert x.shape[1:] == (1, 28, 28)
    assert set(y.unique().tolist()) == {0, 1}
    assert 0.0 <= x.min().item() and x.max().item() <= 1.0
