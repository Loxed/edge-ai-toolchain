The repo is structured as follows:

```
README.md           (Introduction and project overview)
GETTING_STARTED.md  (Instructions for setting up the virtual environment and running the notebooks)
STRUCTURE.md        (This file, describing the repo structure)
pyproject.toml      (Python project configuration file, used during pip install)
.gitignore          (Git ignore file, to avoid committing large datasets and temporary files)
├── data/           (Folder containing downloaded datasets like logic_gates/ or mnist/, usually gitignored for larger ones)
├── notebooks/      (Folder containing Example Notebooks for training, evaluation, compression and deployment)
├── python/edgekit/ (Folder containing reusable Python code for loading widely used datasets, models, training, evaluation and export code snippets)
├── scripts/        (Folder containing useful scripts)
└── tests/          (Folder containing unit, integration, ml and dataset tests to make sure the reusable code works as expected)
```
