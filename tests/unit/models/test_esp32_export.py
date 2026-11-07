import numpy as np
import torch
from torch import nn

from edgekit.exports.esp32_cnn import int8_forward, quantize, quantize_input, write_header


def _random_cnn_state_dict(seed=0):
    torch.manual_seed(seed)
    m = nn.ModuleDict({
        "features": nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)),
        "classifier": nn.Sequential(nn.Flatten(), nn.Dropout(0.0), nn.Linear(32 * 7 * 7, 10)),
    })
    return {k: v for k, v in m.state_dict().items()}, m


def test_quantize_input_is_integer_rounding():
    p = np.array([0, 1, 2, 127, 128, 254, 255], np.uint8)
    assert quantize_input(p).tolist() == [(int(v) * 127 + 127) // 255 for v in p]
    assert quantize_input(np.array([255], np.uint8))[0] == 127


def test_int8_forward_close_to_float():
    sd, m = _random_cnn_state_dict()
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, size=(64, 28, 28), dtype=np.uint8)
    q = quantize(sd, imgs[:32])
    with torch.no_grad():
        x = torch.from_numpy(imgs[32:].astype(np.float32)[:, None] / 255.0)
        ref = m["classifier"](m["features"](x)).numpy()
    got = np.stack([int8_forward(q, im) for im in imgs[32:]])
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 0.1


def test_header_contains_all_arrays(tmp_path):
    sd, _ = _random_cnn_state_dict()
    imgs = np.zeros((3, 28, 28), np.uint8)
    q = quantize(sd, np.full((4, 28, 28), 200, np.uint8))
    out = tmp_path / "model_data.h"
    write_header(out, sd, q, imgs, np.zeros(3, np.uint8), np.zeros(3, np.uint8), np.zeros(3, np.uint8))
    text = out.read_text()
    for name in ("w1_f", "w2_q", "wd_q", "m1", "bq2", "md", "test_images", "ref_pred_int8"):
        assert f" {name}[" in text
    assert "#define N_TEST 3" in text
