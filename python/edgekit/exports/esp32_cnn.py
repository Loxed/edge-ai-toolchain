# python/edgekit/exports/esp32_cnn.py
#
# Export du CNN du TP 3 (FashionMNIST) vers un en-tete C pour l'ESP32-S3, en float32 et en int8,
# avec une reference numpy du calcul int8 qui reproduit exactement ce que fait la carte.
#
# Architecture : conv 3x3 (1 -> 16, pad 1) + ReLU + max pool 2, conv 3x3 (16 -> 32, pad 1) + ReLU
# + max pool 2, dense 1568 -> 10. Entree : pixels uint8 (0 a 255) d'une image 28x28.
#
# Schema int8 (le meme sur PC et sur la carte) :
#   - entree : q = (p * 127 + 127) / 255 en division entiere, echelle s_in = 1/127 (x = p / 255) ;
#   - poids : int8 symetriques par canal de sortie, s_w[c] = max|w_c| / 127 ;
#   - activations apres ReLU : int8 symetriques par tenseur, s_a = max(a) / 127 sur un jeu de calibration ;
#   - convolution : accumulation int32, puis q_out = clamp(floor(acc * m[c] + bq[c] + 0.5), 0, 127) en float32,
#     avec m[c] = s_w[c] * s_in / s_out et bq[c] = b[c] / s_out (ReLU incluse dans le clamp a 0) ;
#   - max pool sur les entiers ; dense : logits = acc * md[c] + bd[c], classe = argmax.

from dataclasses import dataclass

import numpy as np
import torch

N_IN = 28


def quantize_input(pixels: np.ndarray) -> np.ndarray:
    """Pixels uint8 -> int8, arrondi entier identique a celui de la carte."""
    p = pixels.astype(np.int32)
    return ((p * 127 + 127) // 255).astype(np.int32)


def _conv3x3_int(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x (Cin, H, W) int32, w (Cout, Cin, 3, 3) int32 -> acc (Cout, H, W) int32, padding 1 de zeros."""
    cin, h, wd = x.shape
    xp = np.zeros((cin, h + 2, wd + 2), np.int32)
    xp[:, 1:-1, 1:-1] = x
    cols = np.stack([xp[:, i:i + h, j:j + wd] for i in range(3) for j in range(3)], axis=1)  # Cin, 9, H, W
    return np.einsum("ckhw,ock->ohw", cols, w.reshape(w.shape[0], cin, 9).astype(np.int32)).astype(np.int32)


def _requant(acc: np.ndarray, m: np.ndarray, bq: np.ndarray) -> np.ndarray:
    y = acc.astype(np.float32) * m[:, None, None].astype(np.float32)
    y = y + bq[:, None, None].astype(np.float32)
    y = y + np.float32(0.5)
    return np.clip(np.floor(y), 0, 127).astype(np.int32)


def _maxpool2(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


@dataclass
class Int8Model:
    w1: np.ndarray
    m1: np.ndarray
    bq1: np.ndarray
    w2: np.ndarray
    m2: np.ndarray
    bq2: np.ndarray
    wd: np.ndarray
    md: np.ndarray
    bd: np.ndarray
    s_a1: float
    s_a2: float


def _weights(sd):
    return [sd[k].detach().cpu().numpy().astype(np.float32) for k in
            ("features.0.weight", "features.0.bias", "features.3.weight", "features.3.bias",
             "classifier.2.weight", "classifier.2.bias")]


def _float_forward_numpy(w1, b1, w2, b2, wd, bd, x):
    """Reference float32 (pour la calibration) : x (N, 1, 28, 28) dans [0, 1]."""
    t = torch.from_numpy
    import torch.nn.functional as F
    a1 = F.relu(F.conv2d(t(x), t(w1), t(b1), padding=1))
    a2 = F.relu(F.conv2d(F.max_pool2d(a1, 2), t(w2), t(b2), padding=1))
    return a1.numpy(), a2.numpy()


def quantize(state_dict, calib_pixels: np.ndarray) -> Int8Model:
    """calib_pixels : (N, 28, 28) uint8."""
    w1, b1, w2, b2, wd, bd = _weights(state_dict)
    x = calib_pixels.astype(np.float32)[:, None] / 255.0
    a1, a2 = _float_forward_numpy(w1, b1, w2, b2, wd, bd, x)
    s_in = np.float32(1.0 / 127.0)
    s_a1 = np.float32(a1.max() / 127.0)
    s_a2 = np.float32(a2.max() / 127.0)

    def per_channel(w):
        s = np.abs(w.reshape(w.shape[0], -1)).max(axis=1) / 127.0
        q = np.clip(np.round(w / s.reshape(-1, *([1] * (w.ndim - 1)))), -127, 127).astype(np.int8)
        return q, s.astype(np.float32)

    q1, s1 = per_channel(w1)
    q2, s2 = per_channel(w2)
    qd, sd_ = per_channel(wd)
    return Int8Model(
        w1=q1, m1=(s1 * s_in / s_a1).astype(np.float32), bq1=(b1 / s_a1).astype(np.float32),
        w2=q2, m2=(s2 * s_a1 / s_a2).astype(np.float32), bq2=(b2 / s_a2).astype(np.float32),
        wd=qd, md=(sd_ * s_a2).astype(np.float32), bd=bd.astype(np.float32),
        s_a1=float(s_a1), s_a2=float(s_a2),
    )


def int8_forward(m: Int8Model, pixels: np.ndarray) -> np.ndarray:
    """Une image (28, 28) uint8 -> logits float32, calcul identique a la carte."""
    x = quantize_input(pixels)[None]
    a1 = _maxpool2(_requant(_conv3x3_int(x, m.w1.astype(np.int32)), m.m1, m.bq1))
    a2 = _maxpool2(_requant(_conv3x3_int(a1, m.w2.astype(np.int32)), m.m2, m.bq2))
    acc = m.wd.astype(np.int32) @ a2.reshape(-1).astype(np.int32)
    return acc.astype(np.float32) * m.md + m.bd


def _c_array(ctype: str, name: str, values: np.ndarray, per_line: int = 16) -> str:
    flat = values.reshape(-1)
    if ctype == "float":
        items = [f"{v:.9g}f" for v in flat.astype(np.float32)]
    else:
        items = [str(int(v)) for v in flat]
    lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return f"const {ctype} {name}[{len(items)}] = {{\n    " + ",\n    ".join(lines) + "\n};\n"


def write_header(path, state_dict, q: Int8Model, images: np.ndarray, labels: np.ndarray,
                 ref_float: np.ndarray, ref_int8: np.ndarray) -> None:
    w1, b1, w2, b2, wd, bd = _weights(state_dict)
    parts = [
        "// Genere par edge-ai-toolchain/tps/tp7/export_esp32_cnn.py : ne pas modifier a la main.\n",
        "// CNN du TP 3 (FashionMNIST), poids float32 et int8, images de test et predictions de reference.\n",
        "#pragma once\n#include <stdint.h>\n\n",
        f"#define N_TEST {len(images)}\n",
        f"// echelles des activations int8 : s_a1 = {q.s_a1:.6g}, s_a2 = {q.s_a2:.6g}\n\n",
        _c_array("float", "w1_f", w1), _c_array("float", "b1_f", b1),
        _c_array("float", "w2_f", w2), _c_array("float", "b2_f", b2),
        _c_array("float", "wd_f", wd), _c_array("float", "bd_f", bd),
        _c_array("int8_t", "w1_q", q.w1), _c_array("float", "m1", q.m1), _c_array("float", "bq1", q.bq1),
        _c_array("int8_t", "w2_q", q.w2), _c_array("float", "m2", q.m2), _c_array("float", "bq2", q.bq2),
        _c_array("int8_t", "wd_q", q.wd), _c_array("float", "md", q.md), _c_array("float", "bd_q", q.bd),
        _c_array("uint8_t", "test_images", images, per_line=28),
        _c_array("uint8_t", "test_labels", labels),
        _c_array("uint8_t", "ref_pred_float", ref_float),
        _c_array("uint8_t", "ref_pred_int8", ref_int8),
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(parts))
