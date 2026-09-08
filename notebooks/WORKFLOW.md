# Workflow de test

## Principe

On teste sa propre logique, jamais le framework qu'on appelle. Un test
sur `nn.Linear(784, 128)` qui vérifie une shape de sortie ne teste rien
d'utile: PyTorch fait ça correctement depuis des années. On teste le
câblage qu'on a écrit soi-même, pas l'API qu'on invoque.

Question à se poser avant d'écrire un test: "si ce test casse, est-ce
que ça révèle un bug dans mon code, ou juste un changement de
comportement d'une lib externe que je ne contrôle pas ?" Si c'est la
seconde réponse, le test n'a pas sa place ici.

## Trois catégories

- `tests/unit/` : rapide, déterministe, aucun réseau, aucune donnée
  réelle. Tourne à chaque commit. Exemples: parsing CSV, shape/nombre
  de paramètres d'un modèle, reproductibilité avec seed, un split
  train/val qui ne se chevauche pas.

- `tests/integration/` : a besoin de données réelles ou du réseau
  (téléchargement MNIST, sauvegarde/rechargement de poids sur disque).
  Marqué `@pytest.mark.integration`, isolé de la suite rapide, tourne
  à la demande ou en CI dédiée.

- `tests/ml/` : validation empirique (la loss diminue, XOR est appris
  par le MLP mais pas par le perceptron). Pas de comparaison
  d'architectures ici (MLP vs CNN, accuracy sur vrai MNIST), ça, c'est
  du benchmarking qui vit dans les notebooks, pas dans une suite
  pytest stricte.

## Ce qui vaut le coup d'être testé

- Parsing/chargement de données maison (CSV, wrapper dataset)
- Nombre de paramètres entraînables d'un modèle, calculé à la main
  dans le docstring du test, comparé à `sum(p.numel() for p in
  model.parameters())`
- Reproductibilité avec seed fixe
- Sauvegarde puis rechargement d'un modèle: la sortie doit être
  identique avant/après
- Freeze/unfreeze de couches en fine-tuning: les bons paramètres ont
  `requires_grad=False`
- Export ONNX/TFLite: équivalence numérique avec la sortie PyTorch, à
  une tolérance donnée
- Masques de pruning ou calibration PTQ écrits à la main

## Ce qui ne vaut pas le coup

- Vérifier qu'un modèle "apprend bien" ou "overfit un petit batch"
  (utile en debug ponctuel, pas comme gate de CI)
- Comparer l'accuracy de deux architectures (c'est du benchmarking,
  pas un pass/fail)
- Tout ce qui teste le comportement documenté d'une lib externe