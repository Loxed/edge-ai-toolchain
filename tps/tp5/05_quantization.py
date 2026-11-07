import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 05 : quantification int8, PTQ et QAT (TP 5)

    Point de départ : le MobileNetV2 fine-tuné du TP 4 (`models/cifar4_mobilenetv2.pt`). Si le fichier
    n'existe pas, exécuter d'abord `tps/tp4/04_transfer_learning.py`.

    1. quantification d'une couche à la main (numpy), comparée à PyTorch ;
    2. quantification après entraînement (PTQ), effet du jeu de calibration ;
    3. entraînement avec quantification simulée (QAT) ;
    4. tableau de synthèse : taille, précision, latence.

    Les modèles int8 de PyTorch s'exécutent sur CPU (moteur `onednn`) : la latence est mesurée sur le CPU du PC.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import os


    def find_repo_root(start: Path) -> Path:
        current = start.resolve()
        while not (current / "pyproject.toml").exists():
            if current.parent == current:
                raise FileNotFoundError("pyproject.toml introuvable")
            current = current.parent
        return current


    os.chdir(find_repo_root(Path.cwd()))
    return (Path,)


@app.cell
def _(Path):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import Subset, DataLoader

    from edgekit.compression.quantization import (dequantize_numpy, ptq, qat_convert, qat_prepare,
                                                  quantize_per_tensor_numpy, state_dict_bytes)
    from edgekit.datasets.imagefolder import fixed_subset_loader, imagefolder_loaders
    from edgekit.models.mobilenet import mobilenet_v2_transfer
    from edgekit.training.torch_loop import evaluate_accuracy, get_device, measure_latency_ms, train_classifier

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True   # resultats reproductibles sur GPU
    torch.backends.cudnn.benchmark = False
    device = get_device()
    cpu = torch.device("cpu")
    print("device d'entraînement :", device, "| inférence int8 : cpu (onednn)")
    if not Path("models/cifar4_mobilenetv2.pt").exists():
        raise FileNotFoundError("models/cifar4_mobilenetv2.pt absent : exécuter d'abord le notebook 04")
    return (DataLoader, Subset, cpu, dequantize_numpy, device, evaluate_accuracy, fixed_subset_loader,
            imagefolder_loaders, measure_latency_ms, mobilenet_v2_transfer, np, plt, ptq, qat_convert, qat_prepare,
            quantize_per_tensor_numpy, state_dict_bytes, torch, train_classifier)


@app.cell
def _(cpu, evaluate_accuracy, fixed_subset_loader, imagefolder_loaders, measure_latency_ms, mobilenet_v2_transfer,
      state_dict_bytes, torch):
    train_plain, val_loader, test_loader, classes = imagefolder_loaders("data/cifar4", augment=False)
    # Jeu fixe de 256 images d'apprentissage pour la calibration et la mesure d'importance
    train_ds = train_plain.dataset
    calib_loader = fixed_subset_loader(train_ds, 256, seed=0)
    train_aug, _, _, _ = imagefolder_loaders("data/cifar4", augment=True)
    X1 = torch.zeros(1, 3, 96, 96)


    def load_float():
        m = mobilenet_v2_transfer(n_classes=4, pretrained=False)
        m.load_state_dict(torch.load("models/cifar4_mobilenetv2.pt"))
        return m.eval()


    def measure(name, m):
        return (name, state_dict_bytes(m), evaluate_accuracy(m, test_loader, cpu), measure_latency_ms(m, X1, n=50))


    float_model = load_float()
    rows = [measure("float32", float_model)]
    print(f"Résultat : modèle float32 : {rows[0][1] / 1e6:.2f} Mo, précision test {rows[0][2]:.1%}, "
          f"latence CPU {rows[0][3]:.2f} ms")
    return calib_loader, float_model, load_float, measure, rows, test_loader, train_aug, train_ds, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Quantification d'une couche à la main

    Quantification affine par tenseur sur 8 bits signés :
    $q = \mathrm{clip}(\mathrm{round}(w / s) + z, -128, 127)$ et $\hat{w} = (q - z)\,s$, avec
    $s = (w_{max} - w_{min}) / 255$ et $z = \mathrm{round}(-128 - w_{min} / s)$.
    """)
    return


@app.cell
def _(np):
    # SOLUTION
    def my_quantize_per_tensor(w):
        # TODO : à compléter
        raise NotImplementedError

    return (my_quantize_per_tensor,)


@app.cell
def _(dequantize_numpy, float_model, my_quantize_per_tensor, np, plt, quantize_per_tensor_numpy, torch):
    _layer = float_model.features[6].conv[2]          # convolution 1x1 d'un bloc du milieu
    _w = _layer.weight.detach().numpy().astype(np.float32)
    _q, _s, _z = my_quantize_per_tensor(_w)
    _q_ref, _s_ref, _z_ref = quantize_per_tensor_numpy(_w)
    assert np.array_equal(_q, _q_ref) and _z == _z_ref, "écart avec edgekit.compression.quantization"
    _w_hat = dequantize_numpy(_q, _s, _z)
    _err = np.abs(_w_hat - _w)
    _torch_q = torch.quantize_per_tensor(torch.tensor(_w), _s, _z, torch.qint8).int_repr().numpy()

    # Variante par canal (symetrique), celle qu'utilise PyTorch pour les poids
    _absmax = np.abs(_w).reshape(_w.shape[0], -1).max(axis=1)
    _sc = _absmax / 127.0
    _w_hat_c = np.clip(np.round(_w / _sc[:, None, None, None]), -127, 127) * _sc[:, None, None, None]

    print(f"Résultat : couche features[6].conv[2], poids {tuple(_w.shape)}, {_w.size} valeurs")
    print(f"  plage [{_w.min():.4f}, {_w.max():.4f}], scale = {_s:.6f}, zero_point = {_z}")
    print(f"  erreur max {_err.max():.6f} (demi-pas {_s / 2:.6f}), erreur moyenne {_err.mean():.6f}")
    print(f"  identique à torch.quantize_per_tensor : {np.array_equal(_q, _torch_q)}")
    print(f"  erreur moyenne par canal (symétrique) : {np.abs(_w_hat_c - _w).mean():.6f}")
    print(f"  taille : {_w.size * 4} octets en float32, {_w.size} octets en int8")

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.hist(_w.ravel(), bins=100)
    _ax.set_xlabel("valeur du poids")
    _ax.set_ylabel("nombre")
    _ax.set_title("Distribution des poids de features[6].conv[2]")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Quantification après entraînement (PTQ)

    `fuse_model()` fusionne chaque Conv + BatchNorm + ReLU en une seule opération ; des observateurs
    enregistrent la plage des activations sur un jeu de calibration ; `convert()` remplace les couches par
    leurs versions int8.
    """)
    return


@app.cell
def _(float_model, measure, ptq, rows, calib_loader):
    ptq_model = ptq(float_model, calib_loader, n_images=128)
    rows.append(measure("PTQ int8 (128 images)", ptq_model))
    print(f"Résultat : PTQ : {rows[-1][1] / 1e6:.2f} Mo, précision test {rows[-1][2]:.1%}, "
          f"latence CPU {rows[-1][3]:.2f} ms")
    return


@app.cell
def _(DataLoader, Subset, cpu, evaluate_accuracy, fixed_subset_loader, float_model, ptq, test_loader,
      train_ds):
    _one_class = [i for i, t in enumerate(train_ds.targets) if t == 0][:128]
    _cases = [("8 images", fixed_subset_loader(train_ds, 8, seed=0), 8)]
    _cases += [(f"128 images, tirage {s + 1}", fixed_subset_loader(train_ds, 128, seed=s), 128) for s in range(3)]
    _cases += [("128 images d'une seule classe", DataLoader(Subset(train_ds, _one_class), batch_size=128), 128)]
    print("Résultat : précision test de la PTQ selon le jeu de calibration")
    for _name, _loader, _n in _cases:
        _q = ptq(float_model, _loader, n_images=_n)
        print(f"  {_name:30s} : {evaluate_accuracy(_q, test_loader, cpu):.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Entraînement avec quantification simulée (QAT)

    Le modèle est réentraîné 3 epochs avec des opérations "fake quant" qui arrondissent poids et activations
    comme le fera l'int8, tout en gardant des calculs flottants (donc compatibles GPU). Il s'adapte ainsi
    à l'erreur de quantification. Taux d'apprentissage faible, BatchNorm figées.
    """)
    return


@app.cell
def _(device, float_model, measure, qat_convert, qat_prepare, rows, torch, train_aug, train_classifier,
      val_loader):
    torch.manual_seed(0)
    _t = qat_prepare(float_model)
    _r = train_classifier(_t, train_aug, val_loader, epochs=3, device=device, lr=1e-5, log=None, freeze_bn=True)
    qat_model = qat_convert(_t)
    rows.append(measure("QAT int8", qat_model))
    print(f"Résultat : QAT, 3 epochs en {_r.seconds:.1f} s sur {device} : {rows[-1][1] / 1e6:.2f} Mo, "
          f"précision test {rows[-1][2]:.1%}")
    return


@app.cell
def _(rows):
    print("Résultat : synthèse")
    print(f"{'modèle':>24} {'taille (Mo)':>12} {'rapport':>8} {'précision':>10} {'latence CPU (ms)':>17}")
    _ref = rows[0][1]
    for _name, _size, _acc, _lat in rows:
        print(f"{_name:>24} {_size / 1e6:12.2f} {_ref / _size:8.2f} {_acc:10.1%} {_lat:17.2f}")
    return


if __name__ == "__main__":
    app.run()
