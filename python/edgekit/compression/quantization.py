# python/edgekit/compression/quantization.py
#
# Quantification int8 d'un modele torchvision "quantifiable" (QuantStub/DeQuantStub, fuse_model()),
# en mode eager de torch.ao.quantization, moteur onednn (le seul disponible dans les wheels Windows).
#
# Note : torch.ao.quantization est annonce comme deprecie au profit de torchao. Il fonctionne encore
# en torch 2.11. Pour la carte, la quantification finale passe par l'outil du framework embarque.

import copy
import io
import warnings

import numpy as np
import torch
import torch.ao.quantization as tq

ENGINE = "onednn"
warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")
warnings.filterwarnings("ignore", message=".*must run observer before calling calculate_qparams.*")


def state_dict_bytes(model: torch.nn.Module) -> int:
    """Taille du state_dict serialise, en octets (ce qui serait stocke en flash)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def quantize_per_tensor_numpy(w: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Quantification affine int8 par tenseur : q = clip(round(w / scale) + zero_point, -128, 127)."""
    w_min, w_max = min(float(w.min()), 0.0), max(float(w.max()), 0.0)
    scale = (w_max - w_min) / 255.0
    zero_point = int(round(-128 - w_min / scale))
    q = np.clip(np.round(w / scale) + zero_point, -128, 127).astype(np.int8)
    return q, scale, zero_point


def dequantize_numpy(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * scale


def ptq(model: torch.nn.Module, calib_loader, n_images: int = 128) -> torch.nn.Module:
    """Quantification apres entrainement. Le modele d'entree n'est pas modifie."""
    torch.backends.quantized.engine = ENGINE
    q = copy.deepcopy(model).cpu().eval()
    q.fuse_model(is_qat=False)
    q.qconfig = tq.get_default_qconfig(ENGINE)
    tq.prepare(q, inplace=True)
    seen = 0
    with torch.no_grad():
        for x, _ in calib_loader:
            q(x[: n_images - seen])
            seen += min(len(x), n_images - seen)
            if seen >= n_images:
                break
    tq.convert(q, inplace=True)
    return q


def qat_prepare(model: torch.nn.Module) -> torch.nn.Module:
    """Copie du modele avec simulation de la quantification (fake quant), a reentrainer."""
    torch.backends.quantized.engine = ENGINE
    t = copy.deepcopy(model).cpu().train()
    t.fuse_model(is_qat=True)
    t.qconfig = tq.get_default_qat_qconfig(ENGINE)
    tq.prepare_qat(t, inplace=True)
    return t


def qat_convert(model: torch.nn.Module) -> torch.nn.Module:
    m = copy.deepcopy(model).cpu().eval()
    tq.convert(m, inplace=True)
    return m
