# Getting Started

## Activate the virtual environment

From the repository root:

```bash
source .venv/bin/activate
```

## Leave the environment

```bash
deactivate
```

## Create the virtual environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
````

## Install the project

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install pytest
```

## Run the tests

```bash
pytest
```

## Verify the installation

```bash
python -c "import edgekit; print(edgekit.__file__)"
```

It should point to the local `python/edgekit` directory.