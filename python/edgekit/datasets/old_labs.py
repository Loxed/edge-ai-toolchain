# python/edgekit/datasets/old_labs.py
#
# Jeux de donnees repris des anciens TP (Arduino Due, depot ECE-ELEC-CE),
# stockes en CSV dans data/old_labs/.

import csv
import re
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "old_labs"


def _load_xy(name: str, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    with open(DATA_DIR / name, newline="") as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(r[x_col]) for r in rows])
    y = np.array([float(r[y_col]) for r in rows])
    return x, y


def load_motors() -> tuple[np.ndarray, np.ndarray]:
    """4 moteurs "identiques" : rapport cyclique (%) -> vitesse (milliers de tr/min). 24 mesures."""
    return _load_xy("motors.csv", "duty_percent", "speed_krpm")


def load_line_camera() -> tuple[np.ndarray, np.ndarray]:
    """Suivi de ligne : position vue par la camera (-255 gauche, 255 droite) -> tourner a droite (1) ou non (0)."""
    return _load_xy("line_camera.csv", "position", "turn_right")


# --- Tableaux C des anciens sketchs (data/old_labs/raw/) ---

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_c_numbers(text: str) -> np.ndarray:
    """Tous les nombres d'un initialiseur C, commentaires // et /* */ retires."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return np.array([float(v) for v in _NUMBER.findall(text)])


def parse_c_array(source: str, name: str) -> np.ndarray:
    """Valeurs du tableau `name` dans un fichier C (forme `name[...] = { ... };`)."""
    m = re.search(rf"\b{re.escape(name)}\s*((?:\[\s*\d*\s*\])+)\s*=\s*\{{", source)
    if m is None:
        raise KeyError(name)
    dims = [int(d) for d in re.findall(r"\d+", m.group(1))]
    end = source.index("};", m.end())
    values = parse_c_numbers(source[m.end():end])
    return values.reshape(dims)


def load_shapes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cercles (0) et croix (1) en 10x10 pixels binaires, aplatis en 100 valeurs.

    Retourne (x_train (50, 100), y_train (50,), x_test (8, 100)). Les 4 premieres images de test
    sont cercle, cercle, croix, croix ; les 4 dernieres sont a classer (question du TP 4).
    """
    src = (DATA_DIR / "raw" / "shapes_10x10.ino").read_text(encoding="utf-8", errors="replace")
    x_train = parse_c_array(src, "inputs")
    y_train = parse_c_array(src, "expectedOutput").reshape(-1)
    x_test = parse_c_array(src, "testData")
    return x_train, y_train, x_test


def load_mnist_1v8() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extrait MNIST de l'ancien TP 5 : 25 "1" puis 25 "8" en apprentissage, 15 + 15 en test.

    Images 28x28 deja normalisees dans [0, 1]. Label 0 pour "1", 1 pour "8".
    """
    raw = DATA_DIR / "raw"
    x_train = parse_c_numbers((raw / "mnist_1v8_train.txt").read_text()).reshape(-1, 28, 28)
    x_test = parse_c_numbers((raw / "mnist_1v8_test.txt").read_text()).reshape(-1, 28, 28)
    y_train = np.repeat([0.0, 1.0], len(x_train) // 2)
    y_test = np.repeat([0.0, 1.0], len(x_test) // 2)
    return x_train, y_train, x_test, y_test
