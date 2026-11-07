"""Quantifie le MobileNetV2 du TP 4 pour ESP-DL (ESP32-S3) et prépare les données de test de la carte.

A lancer avec le venv esp-ppq (.ppqvenv), depuis la racine du toolchain, apres le notebook 04 :
    .ppqvenv/Scripts/python.exe tps/tp7/espdl_quantize.py --out <dossier main du projet ESP-IDF>
Modele elague du TP 6 (export ONNX) au lieu du modele PyTorch du TP 4 :
    .ppqvenv/Scripts/python.exe tps/tp7/espdl_quantize.py --onnx models/cifar4_mobilenetv2_pruned50.onnx --out ...

Produit dans --out :
    model.espdl          modele int8 pour ESP-DL (+ .info et .json a cote)
    test_images.bin      N images RGB 96x96 uint8, HWC, a la suite
    test_meta.h          etiquettes, predictions de reference (PC float32, PC int8 simule), normalisation
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_model(path: str) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


class ImagesOnly(Dataset):
    """Images normalisees (sans etiquette), pour la calibration."""

    def __init__(self, files):
        self.files = files
        self.tf = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB"))


def to_tensor_from_uint8(rgb: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return transforms.Normalize(MEAN, STD)(x)[None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cifar4_mobilenetv2.pt")
    ap.add_argument("--onnx", default=None, help="modele ONNX a quantifier a la place de --model (TP 6)")
    ap.add_argument("--data", default="data/cifar4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--optim", choices=["none", "eq", "eq-tqt"], default="eq",
                    help="none : reglages par defaut ; eq : calibration KL + egalisation des couches ; "
                         "eq-tqt : idem + affinage des echelles (TQT), plus long")
    ap.add_argument("--test-image", type=int, default=None,
                    help="index d'une image de test dont les valeurs sont exportees pour model->test() "
                         "(par defaut : entree aleatoire)")
    args = ap.parse_args()

    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx, espdl_quantize_torch
    from esp_ppq.executor import TorchExecutor

    # Reglages recommandes par la doc ESP-DL pour MobileNetV2 (how_to_deploy_mobilenetv2)
    setting = QuantizationSettingFactory.espdl_setting()
    if args.optim != "none":
        setting.quantize_activation_setting.calib_algorithm = "kl"
        setting.equalization = True
        setting.equalization_setting.iterations = 4
        setting.equalization_setting.value_threshold = 0.1
        setting.equalization_setting.opt_level = 2
    if args.optim == "eq-tqt":
        setting.tqt_optimization = True
        setting.tqt_optimization_setting.collecting_device = "cpu"
        setting.tqt_optimization_setting.steps = 300
        setting.tqt_optimization_setting.lr = 1e-5
        setting.tqt_optimization_setting.block_size = 4

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = None if args.onnx else load_model(args.model)
    classes = sorted(p.name for p in Path(args.data, "test").iterdir() if p.is_dir())

    train_files = sorted(Path(args.data, "train").rglob("*.png"))
    rng = np.random.default_rng(0)
    calib_files = [train_files[i] for i in rng.permutation(len(train_files))[:args.n_calib]]
    calib = DataLoader(ImagesOnly(calib_files), batch_size=16, shuffle=False)

    # Images de test : RGB uint8 96x96 (ce que fournira la camera), 10 par classe
    imgs, labels = [], []
    for c, name in enumerate(classes):
        for f in sorted(Path(args.data, "test", name).glob("*.png"))[:args.n_per_class]:
            imgs.append(np.asarray(Image.open(f).convert("RGB").resize((96, 96), Image.BILINEAR), np.uint8))
            labels.append(c)
    imgs = np.stack(imgs)
    (out / "test_images.bin").write_bytes(imgs.tobytes())

    # Valeurs exportees pour model->test() : une vraie image si demande, sinon une entree aleatoire
    test_kwargs = {"input_shape": [1, 3, 96, 96]}
    if args.test_image is not None:
        test_kwargs["inputs"] = [to_tensor_from_uint8(imgs[args.test_image])]

    common = dict(
        espdl_export_file=str(out / "model.espdl"),
        calib_dataloader=calib,
        calib_steps=len(calib),
        target="esp32s3",
        setting=setting,
        collate_fn=lambda x: x.to("cpu"),
        device="cpu",
        error_report=False,
        export_test_values=True,
        verbose=1,
    )
    if args.onnx:
        import onnxruntime as ort

        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        float_model = lambda x: torch.from_numpy(sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0])
        graph = espdl_quantize_onnx(onnx_import_file=args.onnx, num_of_bits=8, **common, **test_kwargs)
    else:
        float_model = model
        graph = espdl_quantize_torch(model=model, quant_type="w8a8", **common, **test_kwargs)

    executor = TorchExecutor(graph=graph, device="cpu")
    pred_f, pred_q = [], []
    with torch.no_grad():
        for rgb in imgs:
            x = to_tensor_from_uint8(rgb)
            pred_f.append(int(float_model(x).argmax(1)))
            pred_q.append(int(executor.forward(inputs=x)[0].argmax(1)))
    labels = np.array(labels)
    pred_f, pred_q = np.array(pred_f), np.array(pred_q)

    def arr(name, v):
        return f"static const uint8_t {name}[{len(v)}] = {{{', '.join(str(int(x)) for x in v)}}};\n"

    meta = [
        "// Genere par edge-ai-toolchain/tps/tp7/espdl_quantize.py : ne pas modifier a la main.\n#pragma once\n#include <stdint.h>\n\n",
        f"#define N_TEST {len(labels)}\n#define IMG_H 96\n#define IMG_W 96\n\n",
        f"static const char *CLASS_NAMES[{len(classes)}] = {{{', '.join(json.dumps(c) for c in classes)}}};\n",
        f"static const float NORM_MEAN[3] = {{{MEAN[0]}f, {MEAN[1]}f, {MEAN[2]}f}};\n",
        f"static const float NORM_STD[3] = {{{STD[0]}f, {STD[1]}f, {STD[2]}f}};\n\n",
        arr("test_labels", labels), arr("ref_pred_float", pred_f), arr("ref_pred_int8", pred_q),
    ]
    (out / "test_meta.h").write_text("".join(meta), encoding="utf-8")

    print(f"modele : {out / 'model.espdl'} ({(out / 'model.espdl').stat().st_size} octets)")
    print(f"precision PC float32 : {(pred_f == labels).mean():.1%} sur {len(labels)} images")
    print(f"precision PC int8 (esp-ppq) : {(pred_q == labels).mean():.1%}, accord int8 / float32 {(pred_q == pred_f).mean():.1%}")


if __name__ == "__main__":
    main()
