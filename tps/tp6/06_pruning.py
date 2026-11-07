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
    # 06 : pruning et export ONNX (TP 6)

    Point de départ : le MobileNetV2 fine-tuné du TP 4 (`models/cifar4_mobilenetv2.pt`).

    1. pruning **non structuré** : mise à zéro des poids de plus faible valeur absolue ;
    2. pruning **structuré** : suppression de canaux entiers dans chaque bloc ;
    3. combinaison pruning structuré + quantification int8 (TP 5) ;
    4. export ONNX et vérification avec `onnxruntime`.
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
    import copy
    import gzip
    import io

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from edgekit.compression.pruning import (global_magnitude_prune, make_permanent, prune_mobilenet_channels,
                                             sparsity)
    from edgekit.compression.quantization import ptq, state_dict_bytes
    from edgekit.datasets.imagefolder import fixed_subset_loader, imagefolder_loaders
    from edgekit.models.mobilenet import mobilenet_v2_transfer
    from edgekit.models.torch_mlp import count_parameters
    from edgekit.training.torch_loop import evaluate_accuracy, get_device, measure_latency_ms, train_classifier

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True   # resultats reproductibles sur GPU
    torch.backends.cudnn.benchmark = False
    device = get_device()
    cpu = torch.device("cpu")
    print("device d'entraînement :", device)
    if not Path("models/cifar4_mobilenetv2.pt").exists():
        raise FileNotFoundError("models/cifar4_mobilenetv2.pt absent : exécuter d'abord le notebook 04")
    return (copy, count_parameters, cpu, device, evaluate_accuracy, global_magnitude_prune, gzip,
            fixed_subset_loader, imagefolder_loaders, io, make_permanent, measure_latency_ms, mobilenet_v2_transfer, np, plt,
            prune_mobilenet_channels, ptq, sparsity, state_dict_bytes, torch, train_classifier)


@app.cell
def _(count_parameters, cpu, evaluate_accuracy, fixed_subset_loader, gzip, imagefolder_loaders, io,
      measure_latency_ms, mobilenet_v2_transfer, state_dict_bytes, torch):
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


    def gzip_bytes(m):
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        return len(gzip.compress(buf.getvalue()))


    def measure(m):
        m = m.cpu().eval()
        return dict(params=count_parameters(m), size=state_dict_bytes(m), gz=gzip_bytes(m),
                    acc=evaluate_accuracy(m, test_loader, cpu), lat=measure_latency_ms(m, X1, n=50))


    base = measure(load_float())
    print(f"Résultat : modèle d'origine : {base['params']} paramètres, {base['size'] / 1e6:.2f} Mo "
          f"({base['gz'] / 1e6:.2f} Mo compressé gzip), précision {base['acc']:.1%}, latence CPU {base['lat']:.2f} ms")
    return X1, base, load_float, measure, test_loader, train_aug, calib_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Pruning non structuré

    `torch.nn.utils.prune.global_unstructured` classe tous les poids de toutes les convolutions et couches
    denses par valeur absolue et met à zéro la fraction demandée. Les tenseurs gardent leur forme.
    """)
    return


@app.cell
def _(device, global_magnitude_prune, load_float, make_permanent, measure, sparsity, torch, train_aug,
      train_classifier, val_loader):
    unstructured = {}
    print("Résultat : pruning non structuré")
    print(f"{'taux':>5} {'zéros':>6} {'précision sans FT':>18} {'avec FT (3 ep.)':>16} {'taille (Mo)':>12} "
          f"{'gzip (Mo)':>10} {'latence (ms)':>13}")
    for _amount in (0.3, 0.5, 0.7, 0.9):
        _m = global_magnitude_prune(load_float(), _amount)
        _no_ft = measure(_m)["acc"]
        torch.manual_seed(0)
        train_classifier(_m, train_aug, val_loader, epochs=3, device=device, lr=1e-4, log=None, freeze_bn=True)
        make_permanent(_m)
        _r = measure(_m)
        unstructured[_amount] = (_no_ft, _r)
        print(f"{_amount:5.0%} {sparsity(_m):6.1%} {_no_ft:18.1%} {_r['acc']:16.1%} {_r['size'] / 1e6:12.2f} "
              f"{_r['gz'] / 1e6:10.2f} {_r['lat']:13.2f}")
    return (unstructured,)


@app.cell
def _(base, plt, unstructured):
    _x = [0.0] + list(unstructured)
    _fig, _ax = plt.subplots(figsize=(5.5, 3.2))
    _ax.plot(_x, [base["acc"]] + [v[0] for v in unstructured.values()], "o-", label="sans réentraînement")
    _ax.plot(_x, [base["acc"]] + [v[1]["acc"] for v in unstructured.values()], "o-", label="avec 3 epochs")
    _ax.set_xlabel("fraction de poids mis à zéro")
    _ax.set_ylabel("précision test")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Pruning structuré

    Dans chaque bloc MobileNetV2 (expansion $1 \times 1$, convolution depthwise $3 \times 3$, projection
    $1 \times 1$), les canaux internes les moins importants sont supprimés. Importance d'un canal, mesurée
    sur 256 images : activation moyenne $\times$ poids de projection qui la lisent. Les entrées et sorties
    des blocs ne changent pas ; le réseau obtenu est plus petit, avec des tenseurs denses.

    Deux stratégies : élaguer en une passe, ou par étapes avec un réentraînement entre chaque étape.
    """)
    return


@app.cell
def _(device, load_float, measure, prune_mobilenet_channels, torch, train_aug, train_classifier,
      calib_loader, val_loader):
    def finetune(m, epochs=6):
        torch.manual_seed(0)
        train_classifier(m.to(device), train_aug, val_loader, epochs=epochs, device=device, lr=1e-4, log=None,
                         freeze_bn=True)
        return m


    structured = {}
    print("Résultat : pruning structuré des canaux internes")
    print(f"{'stratégie':>26} {'paramètres':>11} {'sans FT':>8} {'avec FT':>8} {'taille (Mo)':>12} {'latence (ms)':>13}")
    for _ratio in (0.25, 0.5):
        _m = prune_mobilenet_channels(load_float().to(device), _ratio, calib_loader)
        _no_ft = measure(_m)["acc"]
        _r = measure(finetune(_m))
        structured[f"une passe {_ratio:.0%}"] = (_m, _r)
        print(f"{'une passe ' + format(_ratio, '.0%'):>26} {_r['params']:11d} {_no_ft:8.1%} {_r['acc']:8.1%} "
              f"{_r['size'] / 1e6:12.2f} {_r['lat']:13.2f}")

    # Par etapes : 25 % puis un tiers des canaux restants = 50 % au total
    _m = load_float().to(device)
    for _step in (0.25, 1 / 3):
        prune_mobilenet_channels(_m, _step, calib_loader)
        finetune(_m)
    _r = measure(_m)
    structured["par étapes 50%"] = (_m, _r)
    print(f"{'par étapes 25 % puis 50 %':>26} {_r['params']:11d} {'':>8} {_r['acc']:8.1%} "
          f"{_r['size'] / 1e6:12.2f} {_r['lat']:13.2f}")
    return (structured,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Pruning structuré puis quantification int8
    """)
    return


@app.cell
def _(base, load_float, measure, ptq, structured, calib_loader):
    _rows = [("origine float32", base), ("origine int8", measure(ptq(load_float(), calib_loader)))]
    for _name, (_m, _r) in structured.items():
        _rows.append((f"{_name} float32", _r))
        _rows.append((f"{_name} int8", measure(ptq(_m.cpu(), calib_loader))))
    print("Résultat : synthèse")
    print(f"{'modèle':>28} {'taille (Mo)':>12} {'rapport':>8} {'précision':>10} {'latence CPU (ms)':>17}")
    for _name, _r in _rows:
        print(f"{_name:>28} {_r['size'] / 1e6:12.2f} {base['size'] / _r['size']:8.1f} {_r['acc']:10.1%} {_r['lat']:17.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Export ONNX

    ONNX est un format d'échange de modèles, indépendant de PyTorch, que lisent les outils de conversion des
    frameworks embarqués. Le modèle élagué à 50 % par étapes (float32) est exporté, puis exécuté avec
    `onnxruntime` ; ses sorties sont comparées à celles de PyTorch sur tout le jeu de test.
    """)
    return


@app.cell
def _(Path, X1, np, structured, test_loader, torch):
    import onnxruntime as ort

    _m = structured["par étapes 50%"][0].cpu().eval()
    _path = Path("models/cifar4_mobilenetv2_pruned50.onnx")
    torch.onnx.export(_m, (X1,), str(_path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    torch.save(_m, "models/cifar4_mobilenetv2_pruned50_full.pt")
    _sess = ort.InferenceSession(str(_path), providers=["CPUExecutionProvider"])
    _max_diff, _same = 0.0, 0
    _n = 0
    with torch.no_grad():
        for _x, _ in test_loader:
            _ref = _m(_x).numpy()
            _out = _sess.run(None, {"image": _x.numpy()})[0]
            _max_diff = max(_max_diff, float(np.abs(_ref - _out).max()))
            _same += int((_ref.argmax(1) == _out.argmax(1)).sum())
            _n += len(_x)
    print(f"Résultat : export ONNX {_path} ({_path.stat().st_size / 1e6:.2f} Mo)")
    print(f"  écart max PyTorch / onnxruntime sur {_n} images : {_max_diff:.2e}, classes identiques : {_same}/{_n}")
    return


if __name__ == "__main__":
    app.run()
