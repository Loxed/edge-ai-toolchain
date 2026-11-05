import numpy as np
import torch

from edgekit.compression.quantization import (
    dequantize_numpy,
    ptq,
    qat_convert,
    qat_prepare,
    quantize_per_tensor_numpy,
    state_dict_bytes,
)
from edgekit.models.mobilenet import mobilenet_v2_transfer


def test_numpy_quantization_error_is_half_a_step():
    w = np.random.default_rng(0).normal(0, 1, size=1000).astype(np.float32)
    q, scale, zp = quantize_per_tensor_numpy(w)
    assert q.dtype == np.int8
    err = np.abs(dequantize_numpy(q, scale, zp) - w).max()
    assert err <= scale / 2 + 1e-6


def test_numpy_matches_torch_quantize_per_tensor():
    w = np.random.default_rng(1).normal(0, 0.3, size=(8, 4)).astype(np.float32)
    q, scale, zp = quantize_per_tensor_numpy(w)
    tq = torch.quantize_per_tensor(torch.tensor(w), scale, zp, torch.qint8).int_repr().numpy()
    assert np.array_equal(q, tq)


def _fake_loader(n=16):
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(8, 3, 96, 96, generator=g), torch.zeros(8, dtype=torch.long)) for _ in range(n // 8)]


def test_ptq_shrinks_mobilenet_and_keeps_shape():
    model = mobilenet_v2_transfer(4, pretrained=False).eval()
    q = ptq(model, _fake_loader(), n_images=16)
    assert q(torch.zeros(2, 3, 96, 96)).shape == (2, 4)
    assert state_dict_bytes(q) < 0.35 * state_dict_bytes(model)


def test_qat_prepare_and_convert():
    model = mobilenet_v2_transfer(4, pretrained=False)
    t = qat_prepare(model)
    t(torch.randn(2, 3, 96, 96))           # une passe en mode train pour les observateurs
    q = qat_convert(t)
    assert q(torch.zeros(1, 3, 96, 96)).shape == (1, 4)
