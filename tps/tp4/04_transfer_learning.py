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
    # 04 : petit dataset image et transfer learning (TP 4)

    Situation type d'un projet embarqué : quelques centaines d'images, pas des millions. Le dataset utilisé
    ici simule ce cas : 4 classes de CIFAR-10 (avion, voiture, bateau, camion), 150 images d'apprentissage,
    50 de validation et 100 de test par classe, exportées au format `ImageFolder`
    (`data/cifar4/<split>/<classe>/*.png`), le même format que produira une capture caméra.

    1. dataset au format `ImageFolder` ;
    2. référence : petit CNN entraîné depuis zéro ;
    3. transfer learning avec MobileNetV2 pré-entraîné sur ImageNet, backbone gelé (features en cache) ;
    4. fine-tuning des derniers blocs ;
    5. évaluation : matrice de confusion, erreurs, courbe d'apprentissage ;
    6. taille du modèle et sauvegarde pour les TP 5 et 6.
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
def _():
    import json
    import time

    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    import torch.nn.functional as F

    from edgekit.datasets.cifar10_subset import export_subset
    from edgekit.datasets.imagefolder import imagefolder_loaders
    from edgekit.models.mobilenet import IMAGENET_MEAN, IMAGENET_STD, mobilenet_v2_transfer, set_trainable
    from edgekit.models.torch_mlp import count_parameters, model_size_bytes
    from edgekit.training.torch_loop import evaluate_accuracy, get_device, measure_latency_ms, train_classifier

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True   # resultats reproductibles sur GPU
    torch.backends.cudnn.benchmark = False
    device = get_device()
    print("device :", device, f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    return (F, IMAGENET_MEAN, IMAGENET_STD, count_parameters, device, evaluate_accuracy, export_subset,
            imagefolder_loaders, json, measure_latency_ms, mobilenet_v2_transfer, model_size_bytes, nn, plt,
            set_trainable, time, torch, train_classifier)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Le dataset au format `ImageFolder`

    Un dossier par split, un sous-dossier par classe. `torchvision.datasets.ImageFolder` déduit les étiquettes
    des noms de dossiers (ordre alphabétique). Les images CIFAR font $32 \times 32$ ; elles sont agrandies à
    $96 \times 96$, la taille d'entrée retenue pour la suite.
    """)
    return


@app.cell
def _(IMAGENET_MEAN, IMAGENET_STD, Path, export_subset, imagefolder_loaders, plt, torch):
    CLASSES_KEPT = ["airplane", "automobile", "ship", "truck"]
    ROOT = export_subset("data/cifar4", CLASSES_KEPT, n_train=150, n_val=50, n_test=100)
    train_loader, val_loader, test_loader, classes = imagefolder_loaders(ROOT, size=96, augment=True)

    print("Résultat : dataset")
    for _split in ("train", "val", "test"):
        _counts = {c: len(list((Path(ROOT) / _split / c).glob("*.png"))) for c in classes}
        print(f"  {_split:5s} : {_counts}")
    print("  classes (index = ordre alphabétique) :", dict(enumerate(classes)))

    _mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    _std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    _fig, _axes = plt.subplots(1, 8, figsize=(10, 1.8))
    for _i, _a in enumerate(_axes):
        _xk, _yk = test_loader.dataset[_i * 50]
        _a.imshow((_xk * _std + _mean).permute(1, 2, 0).clamp(0, 1))
        _a.set_title(classes[_yk], fontsize=8)
        _a.axis("off")
    _fig
    return ROOT, classes, test_loader, train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Référence : un petit CNN entraîné depuis zéro

    Trois convolutions, un pooling global, une couche dense. 30 epochs avec augmentation.
    """)
    return


@app.cell
def _(count_parameters, device, evaluate_accuracy, nn, test_loader, torch, train_classifier, train_loader,
      val_loader):
    torch.manual_seed(0)
    scratch = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(64, 4),
    )
    _r = train_classifier(scratch, train_loader, val_loader, epochs=30, device=device, lr=1e-3, log=None)
    acc_scratch = evaluate_accuracy(scratch, test_loader, device)
    print(f"Résultat : CNN depuis zéro, {count_parameters(scratch)} paramètres, 30 epochs en {_r.seconds:.1f} s")
    print(f"  précision validation finale {_r.test_accuracy[-1]:.1%}, précision test {acc_scratch:.1%}")
    return (acc_scratch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Transfer learning, backbone gelé

    MobileNetV2 pré-entraîné sur ImageNet (1.2 million d'images, 1 000 classes). La dernière couche est
    remplacée par une couche à 4 sorties ; tout le reste est gelé. Comme le backbone ne change pas, ses
    sorties (1 280 valeurs par image) se calculent **une seule fois** ; seule la couche finale s'entraîne,
    sur ces features en cache.
    """)
    return


@app.cell
def _(F, ROOT, device, imagefolder_loaders, mobilenet_v2_transfer, set_trainable, time, torch):
    model = mobilenet_v2_transfer(n_classes=4, pretrained=True, seed=0)
    n_head = set_trainable(model, last_blocks=0)
    n_total = sum(p.numel() for p in model.parameters())
    model.to(device).eval()


    @torch.no_grad()
    def extract(loader):
        feats, labels = [], []
        for x, y in loader:
            f = F.adaptive_avg_pool2d(model.features(x.to(device)), 1).flatten(1)
            feats.append(f.cpu())
            labels.append(y)
        return torch.cat(feats), torch.cat(labels)


    cache = {}
    return cache, extract, model, n_head, n_total


@app.cell
def _(cache, extract, imagefolder_loaders, ROOT, test_loader, time, val_loader):
    _t0 = time.perf_counter()
    # Features calculees sans augmentation, pour que le cache soit deterministe
    _plain_train, _, _, _ = imagefolder_loaders(ROOT, size=96, augment=False)
    cache["train"] = extract(_plain_train)
    cache["val"] = extract(val_loader)
    cache["test"] = extract(test_loader)
    t_extract = time.perf_counter() - _t0
    print(f"Résultat : features extraites en {t_extract:.1f} s, forme {tuple(cache['train'][0].shape)}")
    return


@app.cell
def _(cache, n_head, n_total, nn, time, torch):
    def train_head(xtr, ytr, epochs=300, lr=1e-2, seed=0):
        torch.manual_seed(seed)
        head = nn.Linear(xtr.shape[1], 4)
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            opt.zero_grad()
            nn.functional.cross_entropy(head(xtr), ytr).backward()
            opt.step()
        return head


    def head_acc(head, split):
        x, y = cache[split]
        with torch.no_grad():
            return (head(x).argmax(1) == y).float().mean().item()


    _t0 = time.perf_counter()
    head = train_head(*cache["train"])
    t_head = time.perf_counter() - _t0
    acc_frozen = head_acc(head, "test")
    print(f"Résultat : backbone gelé, {n_head} paramètres entraînés sur {n_total} ({n_head / n_total:.2%})")
    print(f"  entraînement de la tête : {t_head:.2f} s, précision validation {head_acc(head, 'val'):.1%}, "
          f"test {acc_frozen:.1%}")
    return acc_frozen, head, head_acc, train_head


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Fine-tuning des derniers blocs

    La tête entraînée est copiée dans le modèle, puis les 4 derniers blocs de `features` sont dégelés et
    réentraînés avec un taux d'apprentissage 10 fois plus faible, avec augmentation.
    """)
    return


@app.cell
def _(device, evaluate_accuracy, head, mobilenet_v2_transfer, model, set_trainable, test_loader, torch,
      train_classifier, train_loader, val_loader):
    def start_from_head(m):
        with torch.no_grad():
            m.classifier[1].weight.copy_(head.weight)
            m.classifier[1].bias.copy_(head.bias)
        return set_trainable(m, last_blocks=4)


    print("Résultat : fine-tuning des 4 derniers blocs, 6 epochs, lr = 1e-4")
    # Piege classique : en mode train(), les BatchNorm recalculent leurs statistiques sur des petits
    # batchs augmentes et derivent. Comparaison avec et sans BatchNorm figees.
    _naive = mobilenet_v2_transfer(n_classes=4, pretrained=True, seed=0)
    start_from_head(_naive)
    torch.manual_seed(0)
    _r = train_classifier(_naive, train_loader, val_loader, epochs=6, device=device, lr=1e-4, log=None)
    acc_ft_naive = evaluate_accuracy(_naive, test_loader, device)
    print(f"  BatchNorm mises à jour : précision test {acc_ft_naive:.1%}")

    n_ft = start_from_head(model)
    torch.manual_seed(0)
    _r = train_classifier(model, train_loader, val_loader, epochs=6, device=device, lr=1e-4, log=None,
                          freeze_bn=True)
    acc_ft = evaluate_accuracy(model, test_loader, device)
    print(f"  BatchNorm figées        : précision test {acc_ft:.1%} ({n_ft} paramètres entraînés, {_r.seconds:.1f} s)")
    print(f"  précision validation par epoch : " + " ".join(f"{a:.1%}" for a in _r.test_accuracy))
    return (acc_ft,)


@app.cell
def _(acc_frozen, acc_ft, acc_scratch):
    print("Résultat : synthèse")
    print(f"  CNN depuis zéro       : {acc_scratch:.1%}")
    print(f"  MobileNetV2 gelé      : {acc_frozen:.1%}")
    print(f"  MobileNetV2 fine-tuné : {acc_ft:.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Évaluation
    """)
    return


@app.cell
def _(IMAGENET_MEAN, IMAGENET_STD, classes, device, model, plt, test_loader, torch):
    model.eval()
    _conf = torch.zeros(4, 4, dtype=torch.long)
    _wrong = []
    with torch.no_grad():
        for _x, _y in test_loader:
            _p = model(_x.to(device)).argmax(1).cpu()
            for _xi, _t, _q in zip(_x, _y, _p):
                _conf[_t, _q] += 1
                if _t != _q and len(_wrong) < 4:
                    _wrong.append((_xi, _t.item(), _q.item()))
    print("Résultat : matrice de confusion (lignes = vraie classe, colonnes = prédite)")
    print("            " + " ".join(f"{c[:8]:>9}" for c in classes))
    for _i, _c in enumerate(classes):
        print(f"  {_c[:9]:>9} " + " ".join(f"{v:9d}" for v in _conf[_i].tolist()))

    _mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    _std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    _fig, _axes = plt.subplots(1, len(_wrong), figsize=(2.2 * len(_wrong), 2.4))
    for _a, (_xi, _t, _q) in zip(_axes, _wrong):
        _a.imshow((_xi * _std + _mean).permute(1, 2, 0).clamp(0, 1))
        _a.set_title(f"vrai {classes[_t]}\nprédit {classes[_q]}", fontsize=8)
        _a.axis("off")
    _fig
    return


@app.cell
def _(cache, head_acc, torch, train_head):
    print("Résultat : précision test (backbone gelé) en fonction du nombre d'images d'apprentissage par classe")
    _x, _y = cache["train"]
    for _n in (10, 25, 50, 100, 150):
        _idx = torch.cat([torch.nonzero(_y == c).flatten()[:_n] for c in range(4)])
        _h = train_head(_x[_idx], _y[_idx])
        print(f"  {_n:3d} images par classe : {head_acc(_h, 'test'):.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Taille du modèle et sauvegarde
    """)
    return


@app.cell
def _(Path, classes, count_parameters, json, measure_latency_ms, model, model_size_bytes, torch):
    _cpu = model.cpu().eval()
    print("Résultat : taille et latence du MobileNetV2 fine-tuné")
    print(f"  paramètres : {count_parameters(_cpu)}, taille float32 : {model_size_bytes(_cpu) / 1e6:.2f} Mo")
    print(f"  latence CPU (PC), 1 image 96x96 : {measure_latency_ms(_cpu, torch.zeros(1, 3, 96, 96), n=50):.2f} ms")
    Path("models").mkdir(exist_ok=True)
    torch.save(_cpu.state_dict(), "models/cifar4_mobilenetv2.pt")
    Path("models/cifar4_classes.json").write_text(json.dumps(classes))
    print(f"  sauvegardé : models/cifar4_mobilenetv2.pt ({Path('models/cifar4_mobilenetv2.pt').stat().st_size / 1e6:.2f} Mo)")
    return


if __name__ == "__main__":
    app.run()
