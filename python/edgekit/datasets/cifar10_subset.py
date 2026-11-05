# python/edgekit/datasets/cifar10_subset.py
#
# Sous-ensemble de CIFAR-10 exporte au format ImageFolder (<root>/<split>/<classe>/*.png), pour
# simuler un petit dataset "maison" de quelques centaines d'images par classe.
# Source : miroir Hugging Face (uoft-cs/cifar10, fichiers parquet), le serveur d'origine etant lent.

import io
import json
import urllib.request
from pathlib import Path

import numpy as np

HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"
CACHE = Path("data/cifar10_hf")
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _parquet(split: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}.parquet"
    if not path.exists():
        urllib.request.urlretrieve(HF_URL.format(split=split), path)
    return path


def _load(split: str):
    import pyarrow.parquet as pq

    table = pq.read_table(_parquet(split)).to_pydict()
    images = [d["bytes"] for d in table["img"]]
    return images, np.array(table["label"])


def export_subset(root: str | Path, classes: list[str], n_train: int = 150, n_val: int = 50,
                  n_test: int = 100, seed: int = 0) -> Path:
    """Exporte n_train + n_val images par classe (issues du train CIFAR) et n_test (issues du test).

    Ne refait rien si le dossier existe deja avec les memes parametres.
    """
    root = Path(root)
    meta = {"classes": classes, "n_train": n_train, "n_val": n_val, "n_test": n_test, "seed": seed}
    marker = root / "dataset.json"
    if marker.exists() and json.loads(marker.read_text()) == meta:
        return root
    rng = np.random.default_rng(seed)
    for source, splits in (("train", (("train", n_train), ("val", n_val))), ("test", (("test", n_test),))):
        images, labels = _load(source)
        for cls in classes:
            idx = rng.permutation(np.flatnonzero(labels == CIFAR10_CLASSES.index(cls)))
            start = 0
            for split, n in splits:
                out = root / split / cls
                out.mkdir(parents=True, exist_ok=True)
                for k, i in enumerate(idx[start:start + n]):
                    (out / f"{k:04d}.png").write_bytes(images[i])
                start += n
    marker.write_text(json.dumps(meta))
    return root
