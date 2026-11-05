import torch

from edgekit.compression import pruning as P
from edgekit.compression.pruning import (
    global_magnitude_prune,
    make_permanent,
    prune_mobilenet_channels,
    sparsity,
)
from edgekit.compression.quantization import ptq
from edgekit.models.mobilenet import mobilenet_v2_transfer
from edgekit.models.torch_mlp import count_parameters


def _loader(n=16):
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(8, 3, 96, 96, generator=g), torch.zeros(8, dtype=torch.long)) for _ in range(n // 8)]


def test_global_magnitude_prune_reaches_target_sparsity():
    m = mobilenet_v2_transfer(4, pretrained=False)
    global_magnitude_prune(m, 0.6)
    make_permanent(m)
    assert abs(sparsity(m) - 0.6) < 0.01
    assert m(torch.zeros(1, 3, 96, 96)).shape == (1, 4)


def test_slicing_with_all_channels_kept_is_exact():
    # Verifie le decoupage des Conv et BatchNorm sans changer le reseau
    m = mobilenet_v2_transfer(4, pretrained=False).eval()
    x = torch.randn(2, 3, 96, 96)
    y0 = m(x)
    for b in P._expansion_blocks(m):
        e, d, p = b.conv[0], b.conv[1], b.conv[2]
        keep = torch.arange(e[0].out_channels)
        e[0] = P._slice_conv(e[0], out_keep=keep)
        e[1] = P._slice_bn(e[1], keep)
        d[0] = P._slice_conv(d[0], out_keep=keep, depthwise=True)
        d[1] = P._slice_bn(d[1], keep)
        b.conv[2] = P._slice_conv(p, in_keep=keep)
    assert torch.allclose(m(x), y0, atol=1e-6)


def test_structured_pruning_shrinks_and_keeps_io():
    m = mobilenet_v2_transfer(4, pretrained=False).eval()
    before = count_parameters(m)
    prune_mobilenet_channels(m, 0.5, _loader())
    assert count_parameters(m) < 0.75 * before
    assert m(torch.zeros(2, 3, 96, 96)).shape == (2, 4)
    assert not any(mod.training for mod in m.modules())


def test_pruned_model_can_still_be_quantized():
    m = mobilenet_v2_transfer(4, pretrained=False).eval()
    prune_mobilenet_channels(m, 0.5, _loader())
    q = ptq(m, _loader(8), n_images=8)
    assert q(torch.zeros(1, 3, 96, 96)).shape == (1, 4)
