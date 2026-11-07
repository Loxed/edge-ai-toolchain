# Code des TP, un dossier par TP

Notebooks marimo (fichiers `.py`) et scripts, rangés par TP du livret. Tout se lance depuis la racine du
toolchain : les données vont dans `data/`, les modèles entraînés dans `models/`.

| Dossier | TP du livret | Contenu | État |
| --- | --- | --- | --- |
| `tp3/` | TP 3 : réseaux de neurones avec PyTorch | `03_pytorch_mlp_cnn.py` : XOR, MLP et CNN sur FashionMNIST, surapprentissage, sauvegarde | tourne |
| `tp4/` | TP 4 : petit dataset et transfer learning | `04_transfer_learning.py` : dataset ImageFolder, MobileNetV2 | tourne |
| `tp5/` | TP 5 : quantification | `05_quantization.py` : quantification à la main, PTQ, QAT | tourne |
| `tp6/` | TP 6 : pruning et export | `06_pruning.py` : pruning non structuré et structuré, export ONNX | tourne |
| `tp7/` | TP 7 : déploiement sur l'ESP32-S3 | `export_esp32_cnn.py` : CNN du TP 3 vers un en-tête C ; `espdl_quantize.py` : MobileNetV2 du TP 4 pour ESP-DL (venv esp-ppq) | testé sur carte |
| `annexes/` | hors parcours | perceptron et MLP codés à la main (`01_perceptron_and_or.py`, `02_mlp_xor.py`) | référence |

Les TP 0 à 2 et la partie carte du TP 7 sont dans le dépôt des TP
([ece-calcul-embarque-ESP32-2026](https://github.com/Loxed/ece-calcul-embarque-ESP32-2026)), dans les dossiers
`tp0/` à `tp7/` du même nom.

## Lancer un notebook

```bash
marimo edit tps/tp3/03_pytorch_mlp_cnn.py
python tps/tp3/03_pytorch_mlp_cnn.py
```

La seconde forme exécute le notebook comme un script. Vérification de tous les notebooks :
`pwsh scripts/check.ps1 -NoLatex`.

## Scripts du TP 7

```bash
python tps/tp7/export_esp32_cnn.py --out <depot des TP>/tp7/cnn_inference/src/model_data.h
.ppqvenv/Scripts/python.exe tps/tp7/espdl_quantize.py --optim eq-tqt --test-image 15 --out <depot des TP>/tp7/espdl/src
```

Le premier part de `models/fashion_cnn.pt` (TP 3), le second de `models/cifar4_mobilenetv2.pt` (TP 4).
Installation du venv esp-ppq : voir `tp7/espdl/README.md` dans le dépôt des TP.

## Conventions

- Graines aléatoires fixées. Le device utilisé (`cuda` ou `cpu`) est affiché.
- Chaque valeur demandée dans le livret est affichée par une cellule dont la sortie commence par "Résultat".
- Les cellules de solution commencent par `# SOLUTION`.
- Chaque notebook termine en moins de 5 minutes sur CPU.

## Version élève

`python scripts/make_student.py` écrit dans `student/tps/` une copie de chaque notebook où les fonctions
marquées `# SOLUTION` sont remplacées par un `TODO`.
