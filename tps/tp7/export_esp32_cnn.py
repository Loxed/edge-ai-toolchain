"""Exporte le CNN du TP 3 (models/fashion_cnn.pt) vers un en-tete C pour l'ESP32-S3.

Usage (depuis la racine du toolchain, apres le notebook 03) :
    python tps/tp7/export_esp32_cnn.py --out <chemin>/model_data.h [--n-test 100]

Affiche les precisions de reference (PyTorch float32, simulation int8) sur les images exportees.
"""

import argparse

import numpy as np
import torch
from torchvision import datasets

from edgekit.exports.esp32_cnn import int8_forward, quantize, write_header


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fashion_cnn.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--n-calib", type=int, default=512)
    args = ap.parse_args()

    sd = torch.load(args.model, map_location="cpu")
    train = datasets.FashionMNIST("data/fashion", train=True, download=True)
    test = datasets.FashionMNIST("data/fashion", train=False, download=True)
    g = torch.Generator().manual_seed(0)
    calib_idx = torch.randperm(len(train), generator=g)[:args.n_calib]
    calib = train.data[calib_idx].numpy()
    images = test.data[:args.n_test].numpy()
    labels = test.targets[:args.n_test].numpy().astype(np.uint8)

    q = quantize(sd, calib)

    # Reference float32 : meme calcul que le modele PyTorch
    import torch.nn.functional as F
    x = torch.from_numpy(images.astype(np.float32)[:, None] / 255.0)
    a = F.max_pool2d(F.relu(F.conv2d(x, sd["features.0.weight"], sd["features.0.bias"], padding=1)), 2)
    a = F.max_pool2d(F.relu(F.conv2d(a, sd["features.3.weight"], sd["features.3.bias"], padding=1)), 2)
    logits = F.linear(a.flatten(1), sd["classifier.2.weight"], sd["classifier.2.bias"])
    ref_float = logits.argmax(1).numpy().astype(np.uint8)
    ref_int8 = np.array([int8_forward(q, im).argmax() for im in images], np.uint8)

    write_header(args.out, sd, q, images, labels, ref_float, ref_int8)
    print(f"en-tete : {args.out}")
    print(f"precision float32 (PyTorch)   : {(ref_float == labels).mean():.1%} sur {len(labels)} images")
    print(f"precision int8 (simulation)   : {(ref_int8 == labels).mean():.1%}")
    print(f"accord int8 / float32         : {(ref_int8 == ref_float).mean():.1%}")
    n_params = sum(v.numel() for v in sd.values())
    print(f"poids : {n_params} parametres, {4 * n_params} octets en float32, "
          f"{q.w1.size + q.w2.size + q.wd.size} octets int8 + {4 * (2 * (16 + 32) + 2 * 10)} octets d'echelles")


if __name__ == "__main__":
    main()
