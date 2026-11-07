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
    # 03 : réseaux de neurones avec PyTorch, du MLP au CNN (TP 3)

    Le workflow standard d'un ingénieur ML, sur le modèle du tutoriel officiel PyTorch :

    1. **XOR** : pourquoi une couche cachée ;
    2. **données** : `Dataset` et `DataLoader` sur FashionMNIST (téléchargé dans `data/fashion/`) ;
    3. **MLP** : modèle, boucle d'entraînement, boucle de test ;
    4. **CNN** : même boucle, autre modèle, comparaison en précision, taille et latence ;
    5. **surapprentissage** : petit dataset, augmentation de données et dropout ;
    6. **sauvegarde** : `state_dict`, rechargement, vérification.

    Les cellules qui commencent par `# SOLUTION` sont à écrire par l'élève.
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
    import time

    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    from edgekit.models.torch_mlp import count_parameters, model_size_bytes
    from edgekit.training.torch_loop import get_device, measure_latency_ms

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True   # resultats reproductibles sur GPU
    torch.backends.cudnn.benchmark = False
    device = get_device()
    print("device :", device, f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    return (DataLoader, Subset, count_parameters, datasets, device, measure_latency_ms,
            model_size_bytes, nn, plt, time, torch, transforms)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. XOR : pourquoi une couche cachée

    Un modèle linéaire (une seule couche `nn.Linear`) contre un MLP à 4 neurones cachés, entraînés sur les
    4 lignes de la table de vérité de XOR.
    """)
    return


@app.cell
def _(nn, torch):
    _X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    print("Résultat : XOR, sortie du modèle pour (0,0) (0,1) (1,0) (1,1)")
    for _name, _make in (("linéaire {2, 1}", lambda: nn.Sequential(nn.Linear(2, 1))),
                         ("MLP {2, 4, 1}  ", lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)))):
        torch.manual_seed(0)
        _m = _make()
        _opt = torch.optim.Adam(_m.parameters(), lr=0.05)
        _loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(500):
            _opt.zero_grad()
            _loss = _loss_fn(_m(_X), _y)
            _loss.backward()
            _opt.step()
        _p = torch.sigmoid(_m(_X)).squeeze(1).tolist()
        print(f"  {_name} : " + " ".join(f"{v:.2f}" for v in _p) + f"   (loss {_loss.item():.4f})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Données : `Dataset` et `DataLoader`

    FashionMNIST : 60 000 images d'apprentissage et 10 000 de test, $28 \times 28$ en niveaux de gris,
    10 classes de vêtements. `transforms.ToTensor()` convertit chaque image en tenseur `float32` dans $[0, 1]$.
    """)
    return


@app.cell
def _(DataLoader, datasets, plt, torch, transforms):
    CLASSES = ["T-shirt", "Pantalon", "Pull", "Robe", "Manteau", "Sandale", "Chemise", "Basket", "Sac", "Bottine"]
    BATCH = 64
    train_data = datasets.FashionMNIST("data/fashion", train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.FashionMNIST("data/fashion", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(0))
    test_loader = DataLoader(test_data, batch_size=1000)

    _xb, _yb = next(iter(train_loader))
    print("Résultat : données")
    print(f"  {len(train_data)} images d'apprentissage, {len(test_data)} de test")
    print(f"  un batch : x {tuple(_xb.shape)} {_xb.dtype}, y {tuple(_yb.shape)} {_yb.dtype}")
    print(f"  {len(train_loader)} batchs par epoch")

    _fig, _axes = plt.subplots(1, 8, figsize=(10, 1.8))
    for _i, _a in enumerate(_axes):
        _a.imshow(_xb[_i, 0], cmap="gray")
        _a.set_title(CLASSES[_yb[_i]], fontsize=8)
        _a.axis("off")
    _fig
    return CLASSES, test_data, test_loader, train_data, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Le MLP et les boucles d'entraînement et de test

    Modèle du tutoriel PyTorch : `Flatten`, puis $784 \rightarrow 512 \rightarrow 512 \rightarrow 10$ avec ReLU.
    La sortie donne 10 scores (logits) ; la classe prédite est celle de score maximal.
    """)
    return


@app.cell
def _(nn):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            return self.net(x)

    return (MLP,)


@app.cell
def _(time, torch):
    # SOLUTION
    def train_one_epoch(model, loader, loss_fn, optimizer, device):
        # TODO : à compléter
        raise NotImplementedError


    # SOLUTION
    @torch.no_grad()
    def evaluate(model, loader, device):
        # TODO : à compléter
        raise NotImplementedError


    def fit(model, train_loader, test_loader, device, epochs, lr=1e-3, log=True):
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        hist = []
        t0 = time.perf_counter()
        for ep in range(1, epochs + 1):
            loss, acc_train = train_one_epoch(model, train_loader, loss_fn, opt, device)
            acc_test = evaluate(model, test_loader, device)
            hist.append((loss, acc_train, acc_test))
            if log:
                print(f"  epoch {ep:2d} : loss {loss:.4f}, précision apprentissage {acc_train:.2%}, test {acc_test:.2%}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        return hist, time.perf_counter() - t0

    return evaluate, fit


@app.cell
def _(MLP, count_parameters, device, fit, model_size_bytes, test_loader, torch, train_loader):
    EPOCHS = 3
    torch.manual_seed(0)
    mlp = MLP()
    print(f"Résultat : MLP, {count_parameters(mlp)} paramètres ({model_size_bytes(mlp)} octets en float32)")
    mlp_hist, mlp_time = fit(mlp, train_loader, test_loader, device, EPOCHS)
    print(f"  durée : {mlp_time:.1f} s ({mlp_time / EPOCHS:.1f} s par epoch) sur {device}")
    return EPOCHS, mlp, mlp_hist, mlp_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Le CNN

    Deux blocs (convolution $3 \times 3$, ReLU, max pooling $2 \times 2$) puis une couche dense. Même
    fonction `fit()`, seul le modèle change.
    """)
    return


@app.cell
def _(nn):
    class CNN(nn.Module):
        def __init__(self, dropout: float = 0.0):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 x 14 x 14
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 x 7 x 7
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(32 * 7 * 7, 10))

        def forward(self, x):
            return self.classifier(self.features(x))

    return (CNN,)


@app.cell
def _(CNN, EPOCHS, count_parameters, device, fit, model_size_bytes, test_loader, torch, train_loader):
    torch.manual_seed(0)
    cnn = CNN()
    print(f"Résultat : CNN, {count_parameters(cnn)} paramètres ({model_size_bytes(cnn)} octets en float32)")
    cnn_hist, cnn_time = fit(cnn, train_loader, test_loader, device, EPOCHS)
    print(f"  durée : {cnn_time:.1f} s ({cnn_time / EPOCHS:.1f} s par epoch) sur {device}")
    return cnn, cnn_hist, cnn_time


@app.cell
def _(EPOCHS, cnn, cnn_hist, cnn_time, count_parameters, measure_latency_ms, mlp, mlp_hist, mlp_time,
      model_size_bytes, torch):
    _x1 = torch.zeros(1, 1, 28, 28)
    _rows = []
    for _name, _m, _h, _t in (("MLP", mlp, mlp_hist, mlp_time), ("CNN", cnn, cnn_hist, cnn_time)):
        _m_cpu = _m.cpu()
        _rows.append((_name, count_parameters(_m), model_size_bytes(_m), _h[-1][2], _t / EPOCHS,
                      measure_latency_ms(_m_cpu, _x1)))
    print("Résultat : comparaison MLP / CNN")
    print(f"{'modèle':>6} {'params':>9} {'octets':>10} {'précision test':>15} {'s/epoch':>8} {'latence CPU (ms)':>17}")
    for _r in _rows:
        print(f"{_r[0]:>6} {_r[1]:9d} {_r[2]:10d} {_r[3]:15.2%} {_r[4]:8.1f} {_r[5]:17.3f}")
    return


@app.cell
def _(CNN, DataLoader, device, fit, test_data, time, torch, train_data):
    # Une epoch du CNN sur CPU, pour comparer au GPU
    if device.type == "cuda":
        torch.manual_seed(0)
        _t0 = time.perf_counter()
        fit(CNN(), DataLoader(train_data, batch_size=64, shuffle=True), DataLoader(test_data, batch_size=1000),
            torch.device("cpu"), epochs=1, log=False)
        print(f"Résultat : une epoch du CNN sur CPU : {time.perf_counter() - _t0:.1f} s")
    else:
        print("Résultat : pas de GPU, comparaison CPU / GPU sautée")
    return


@app.cell
def _(CLASSES, cnn, device, plt, test_data, torch):
    # Matrice de confusion du CNN
    cnn.to(device).eval()
    _conf = torch.zeros(10, 10, dtype=torch.long)
    with torch.no_grad():
        for _x, _y in torch.utils.data.DataLoader(test_data, batch_size=1000):
            _p = cnn(_x.to(device)).argmax(1).cpu()
            for _t, _q in zip(_y, _p):
                _conf[_t, _q] += 1
    _off = _conf.clone()
    _off.fill_diagonal_(0)
    _i, _j = divmod(_off.argmax().item(), 10)
    print(f"Résultat : confusion la plus fréquente : {CLASSES[_i]} prédit {CLASSES[_j]} ({_off[_i, _j].item()} fois)")
    _fig, _ax = plt.subplots(figsize=(5, 4.5))
    _ax.imshow(_conf, cmap="Blues")
    _ax.set_xticks(range(10), CLASSES, rotation=90, fontsize=7)
    _ax.set_yticks(range(10), CLASSES, fontsize=7)
    _ax.set_xlabel("prédit")
    _ax.set_ylabel("vrai")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Surapprentissage sur un petit dataset

    Le dataset du projet (images de composants prises avec la caméra) comptera quelques centaines d'images.
    Ce cas se simule ici avec 600 images d'apprentissage. Trois versions du même CNN, 30 epochs :
    sans régularisation, avec dropout, avec dropout et augmentation de données (petits décalages et rotations).
    """)
    return


@app.cell
def _(CNN, DataLoader, Subset, datasets, device, fit, test_loader, torch, transforms):
    _idx = torch.randperm(60_000, generator=torch.Generator().manual_seed(0))[:600].tolist()
    _plain = Subset(datasets.FashionMNIST("data/fashion", train=True, transform=transforms.ToTensor()), _idx)
    _aug_tf = transforms.Compose([transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)), transforms.ToTensor()])
    _aug = Subset(datasets.FashionMNIST("data/fashion", train=True, transform=_aug_tf), _idx)

    print("Résultat : 600 images d'apprentissage, 30 epochs")
    print(f"{'version':>22} {'précision apprentissage':>24} {'précision test':>15} {'écart':>7}")
    for _name, _ds, _drop in (("sans régularisation", _plain, 0.0), ("dropout 0.5", _plain, 0.5),
                              ("dropout + augmentation", _aug, 0.5)):
        torch.manual_seed(0)
        _loader = DataLoader(_ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))
        _h, _ = fit(CNN(dropout=_drop), _loader, test_loader, device, epochs=30, log=False)
        _tr, _te = _h[-1][1], _h[-1][2]
        print(f"{_name:>22} {_tr:24.2%} {_te:15.2%} {_tr - _te:7.2%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Sauvegarde et rechargement

    Seuls les poids (`state_dict`) sont sauvegardés. Pour recharger, il faut la classe du modèle.
    """)
    return


@app.cell
def _(CNN, Path, cnn, test_loader, torch):
    Path("models").mkdir(exist_ok=True)
    _path = Path("models/fashion_cnn.pt")
    cnn.cpu().eval()
    torch.save(cnn.state_dict(), _path)
    _reloaded = CNN()
    _reloaded.load_state_dict(torch.load(_path))
    _reloaded.eval()
    _x, _ = next(iter(test_loader))
    with torch.no_grad():
        _same = torch.equal(cnn(_x), _reloaded(_x))
    print(f"Résultat : {_path} ({_path.stat().st_size} octets), sorties identiques après rechargement : {_same}")
    return


if __name__ == "__main__":
    app.run()
