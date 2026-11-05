# Notebooks

Notebooks marimo (fichiers `.py`). Chacun s'exécute aussi comme un script :

```bash
python notebooks/vision/03_pytorch_mlp_cnn.py
```

Édition interactive : `marimo edit notebooks/vision/03_pytorch_mlp_cnn.py`.
Vérification de tous les notebooks : `pwsh scripts/check.ps1 -NoLatex`.

## Correspondance avec le livret

| Notebook | TP du livret | État |
| --- | --- | --- |
| `vision/03_pytorch_mlp_cnn.py` | TP 3 : XOR, MLP et CNN sur FashionMNIST, surapprentissage, sauvegarde | tourne |
| `vision/04_transfer_learning.py` | TP 4 : petit dataset ImageFolder, transfer learning MobileNetV2 | tourne |
| `compression/05_quantization.py` | TP 5 : quantification à la main, PTQ et QAT | tourne |
| `compression/06_pruning.py` | TP 6 : pruning non structuré et structuré, export ONNX | tourne |

Le TP 7 (déploiement sur l'ESP32-S3) se fait sur la carte, sans notebook.

Hors parcours (gardés comme référence) : `fundamentals/01_perceptron_and_or.py`, `fundamentals/02_mlp_xor.py`
(perceptron et MLP codés à la main).

## Conventions

- Graines aléatoires fixées. Le device utilisé (`cuda` ou `cpu`) est affiché.
- Chaque valeur demandée dans le livret est affichée par une cellule dont la sortie commence par "Résultat".
- Les cellules de solution commencent par `# SOLUTION`.
- Chaque notebook termine en moins de 5 minutes sur CPU.

## Version élève

`python scripts/make_student.py` écrit dans `student/notebooks/` une copie de chaque notebook où les
fonctions marquées `# SOLUTION` sont remplacées par un `TODO`.
