import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01: The Perceptron, from clean data to noisy data

    This notebook is split into three parts:

    1. Training a basic perceptron on the exact 0/1 truth tables for AND, OR, NAND and NOR, then looking at what happens with XOR.
    2. Replacing the four exact points with noisy floating-point samples. The perceptron still works, but the final accuracy can depend quite a bit on when training stops.
    3. Adding the pocket algorithm, which snapshots the best solution found during training instead of simply returning the last one.

    ---

    With noisy data, the perceptron may never find one perfect solution. As it keeps training, it can sometimes move from a good solution to a worse one.

    The pocket algorithm fixes this by keeping a copy of the best weights found so far.

    Instead of returning the last version of the perceptron, it returns the best version it saw during training.

    ---

    The main code used here (`Perceptron`, `train_perceptron`, `LogicGatesDataset`, `sample_noisy_points`) lives in the `edgekit` package and is covered by unit tests. The notebook is mostly here to experiment with it and visualize the results.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Jump to the repo root, wherever marimo happened to start from.
    from pathlib import Path
    import os

    def find_repo_root(start: Path) -> Path:
        current = start.resolve()
        while not (current / "pyproject.toml").exists():
            if current.parent == current:
                raise FileNotFoundError("pyproject.toml not found in any parent directory")
            current = current.parent
        return current

    os.chdir(find_repo_root(Path.cwd()))
    print("cwd:", Path.cwd())
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from edgekit.datasets.logic_gates import LogicGatesDataset
    from edgekit.datasets.synthetic import sample_noisy_points
    from edgekit.models.perceptron import Perceptron
    from edgekit.training.perceptron import train_perceptron

    GATES = ["and", "or", "nand", "nor"]
    return (
        GATES,
        LogicGatesDataset,
        Perceptron,
        np,
        plt,
        sample_noisy_points,
        train_perceptron,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: The classic demo

    A perceptron with 2 inputs computes `w1*x1 + w2*x2 + b`, and predicts
    1 when that's positive. So its boundary is always a straight line.

    AND, OR, NAND, NOR can all be split by a single line, so the
    perceptron learns them easily, in just a few passes over the 4 exact
    points of each truth table.
    """)
    return


@app.cell
def _(GATES, LogicGatesDataset, Perceptron, train_perceptron):
    discrete_models = {}

    for _gate in GATES:
        _dataset = LogicGatesDataset(f"data/logic_gates/{_gate}.csv")
        _model = Perceptron(n_inputs=2)

        train_perceptron(_model, _dataset, epochs=200, learning_rate=0.1)

        _predictions = [_model.predict(x) for x, _ in _dataset]
        _expected = [y for _, y in _dataset]

        discrete_models[_gate] = _model

        _status = "OK" if _predictions == _expected else "FAILED"
        print(f"{_gate.upper():5s}  predictions={_predictions}  expected={_expected}  [{_status}]")
    return (discrete_models,)


@app.cell
def _(np):
    def plot_decision_boundary(ax, model, dataset, title, point_size=200, point_alpha=1.0):
        xs = [x for x, _ in dataset]
        ys = [y for _, y in dataset]

        x1_vals = [x[0] for x in xs]
        x2_vals = [x[1] for x in xs]

        colors = ["tab:red" if y == 0 else "tab:blue" for y in ys]
        ax.scatter(x1_vals, x2_vals, c=colors, s=point_size, alpha=point_alpha,
                   edgecolors="black" if point_size >= 100 else "none", zorder=3)

        w1, w2 = model.weights
        b = model.bias

        grid = np.linspace(-0.5, 1.5, 200)

        if abs(w2) > 1e-9:
            # w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b) / w2
            boundary_x2 = -(w1 * grid + b) / w2
            ax.plot(grid, boundary_x2, "k--", zorder=2)
        elif abs(w1) > 1e-9:
            # vertical boundary: x1 = -b / w1
            ax.axvline(-b / w1, linestyle="--", color="black", zorder=2)

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(title)

    return (plot_decision_boundary,)


@app.cell
def _(GATES, LogicGatesDataset, discrete_models, plot_decision_boundary, plt):
    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4))

    for _ax, _gate in zip(_axes, GATES):
        _dataset = LogicGatesDataset(f"data/logic_gates/{_gate}.csv")
        plot_decision_boundary(_ax, discrete_models[_gate], _dataset, _gate.upper())

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What about XOR?

    XOR can't be split by any single line: its two "1" points and its
    two "0" points sit on opposite corners. Watch the perceptron fail.
    """)
    return


@app.cell
def _(
    LogicGatesDataset,
    Perceptron,
    plot_decision_boundary,
    plt,
    train_perceptron,
):
    xor_corners = LogicGatesDataset("data/logic_gates/xor.csv")
    xor_discrete_model = Perceptron(n_inputs=2)

    train_perceptron(xor_discrete_model, xor_corners, epochs=100, learning_rate=0.1)

    _predictions = [xor_discrete_model.predict(x) for x, _ in xor_corners]
    _expected = [y for _, y in xor_corners]

    print(f"predictions={_predictions}  expected={_expected}")
    print("Perceptron fails to learn XOR:", _predictions != _expected)

    _fig, _ax = plt.subplots(figsize=(4, 4))
    plot_decision_boundary(_ax, xor_discrete_model, xor_corners, "XOR (expected failure)")
    plt.gca()
    return (xor_corners,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: Adding realistic noise

    Real measurements never land exactly on 0 or 1. We simulate that:
    every "0" becomes ~30 points scattered around 0.4, every "1" becomes
    ~30 points scattered around 0.6 (`sample_noisy_points`, tested in
    `tests/unit/datasets/test_synthetic.py`).

    We train the same `train_perceptron`, but tell it to just keep
    whatever it has after the last pass (`keep_best=False`), with no
    safety net. Let's see what happens at different training lengths.
    """)
    return


@app.cell
def _(GATES, LogicGatesDataset, sample_noisy_points):
    noisy_datasets = {
        _gate: sample_noisy_points(
            LogicGatesDataset(f"data/logic_gates/{_gate}.csv"),
            n_per_class=50, noise_std=0.05, seed=0,
        )
        for _gate in GATES
    }

    def accuracy(model, dataset):
        predictions = [model.predict(x) for x, _ in dataset]
        expected = [y for _, y in dataset]
        return sum(p == e for p, e in zip(predictions, expected)) / len(dataset)

    return accuracy, noisy_datasets


@app.cell
def _(GATES, Perceptron, accuracy, noisy_datasets, train_perceptron):
    print(f"{'epochs':>8s}  " + "  ".join(f"{_g.upper():>6s}" for _g in GATES))
    for _epochs in [20, 100, 500, 1000, 2500, 5000]:
        _row = []
        for _gate in GATES:
            _model = Perceptron(n_inputs=2)
            train_perceptron(_model, noisy_datasets[_gate], epochs=_epochs,
                             learning_rate=0.1, seed=0, keep_best=False)
            _row.append(f"{accuracy(_model, noisy_datasets[_gate]):.1%}")
        print(f"{_epochs:8d}  " + "  ".join(f"{v:>6s}" for v in _row))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the accuracy doesn't settle down: it bounces up and down as
    we train longer, sometimes getting *worse* with more epochs. That
    looks like a bug, but it's actually a known limitation: once the
    data isn't perfectly clean (which is basically always true with
    noise), the plain perceptron has no guarantee it will ever settle on
    a good answer. It just keeps nudging its weights forever, and can
    wander from a good solution to a bad one and back. Training longer
    is not a reliable fix here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3: The fix (keep the best version seen)

    A simple, classic fix (called the "pocket" algorithm): instead of
    keeping whatever the perceptron looks like after the last epoch,
    remember the best version seen at any point during training, and use
    that one at the end.

    `train_perceptron` already does this by default (`keep_best=True`).
    Let's rerun the same table, but this time with far fewer epochs,
    since the fix turns out to also converge a lot faster.
    """)
    return


@app.cell
def _(GATES, Perceptron, accuracy, noisy_datasets, train_perceptron):
    print(f"{'epochs':>8s}  " + "  ".join(f"{_g.upper():>6s}" for _g in GATES))
    for _epochs in [1, 2, 3, 5, 10, 20,50,100]:
        _row = []
        for _gate in GATES:
            _model = Perceptron(n_inputs=2)
            train_perceptron(_model, noisy_datasets[_gate], epochs=_epochs,
                             learning_rate=0.1, seed=0)  # keep_best=True by default
            _row.append(f"{accuracy(_model, noisy_datasets[_gate]):.1%}")
        print(f"{_epochs:8d}  " + "  ".join(f"{v:>6s}" for v in _row))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two things stand out compared to Part 2:

    - **It's stable.** No more bouncing around as training gets longer.
    - **It's fast.** By epoch 5 to 10 it's already about as good as it's
      going to get. No need for the thousands of epochs Part 2 used.
      Keeping the best version doesn't just avoid bad luck at the end,
      it also means we can stop training much sooner.

    Let's train once at a short, fixed epoch count and plot the
    boundaries, this time over the full noisy clouds instead of 4 exact
    points.
    """)
    return


@app.cell
def _(
    GATES,
    Perceptron,
    accuracy,
    noisy_datasets,
    plot_decision_boundary,
    plt,
    train_perceptron,
):
    final_models = {}

    for _gate in GATES:
        _model = Perceptron(n_inputs=2)
        train_perceptron(_model, noisy_datasets[_gate], epochs=10, learning_rate=0.1, seed=0)
        final_models[_gate] = _model
        print(f"{_gate.upper():5s}  accuracy={accuracy(_model, noisy_datasets[_gate]):.2%}")

    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4))
    for _ax, _gate in zip(_axes, GATES):
        plot_decision_boundary(_ax, final_models[_gate], noisy_datasets[_gate], _gate.upper(),
                               point_size=25, point_alpha=0.7)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Accuracy sits around 90% to 98%, not 100%. This is expected. Some noisy points end up close enough to the boundary that they
    land on the wrong side just by chance. That's a natural limit set by
    how close together the clusters are relative to the noise, and real
    sensor data behaves the same way.

    XOR, unsurprisingly, doesn't get any better from any of this: the
    problem was never about training speed or stability, no single line
    can separate it, noisy or not.
    """)
    return


@app.cell
def _(
    Perceptron,
    accuracy,
    plot_decision_boundary,
    plt,
    sample_noisy_points,
    train_perceptron,
    xor_corners,
):
    xor_noisy = sample_noisy_points(xor_corners, n_per_class=30, noise_std=0.05, seed=0)

    xor_model = Perceptron(n_inputs=2)
    train_perceptron(xor_model, xor_noisy, epochs=100, learning_rate=0.1, seed=0)

    print(f"accuracy={accuracy(xor_model, xor_noisy):.2%}  (a single line can get at most 3 of the 4 clusters right on XOR, close to that ceiling)")

    _fig, _ax = plt.subplots(figsize=(4, 4))
    plot_decision_boundary(_ax, xor_model, xor_noisy, "XOR, noisy + pocket (still fails)",
                           point_size=25, point_alpha=0.7)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrap-up

    - On clean data, the plain perceptron just works.
    - Add real-world noise, and it can become unstable no matter how
      long you train it. This is a real limitation of the algorithm.
    - Keeping the best version fixes both the instability *and* the slow
      training.
    - But it's still just a straight line. It never helps with XOR.

    That last point is exactly why we need a multilayer perceptron
    (MLP), next up in `02_mlp_xor.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
