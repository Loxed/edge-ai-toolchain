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

    3. Adding the pocket algorithm, which keeps the best solution found during training instead of simply returning the last one.

    ---

    With noisy data, the perceptron may never find one perfect solution. As it keeps training, it can sometimes move from a good solution to a worse one.

    The pocket algorithm keeps a copy of the best weights found so far.

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
                raise FileNotFoundError(
                    "pyproject.toml not found in any parent directory"
                )
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
    A perceptron with two inputs is a very small model:

    \[
    \begin{array}{rcl}
    x_1 & \xrightarrow{\;w_1\;} & \\
    && \displaystyle \sum_i w_i x_i + b
       \;\longrightarrow\;
       \text{activation}
       \;\longrightarrow\;
       \{0,1\} \\
    x_2 & \xrightarrow{\;w_2\;} &
    \end{array}
    \]

    It first computes


    \[z = w_1x_1 + w_2x_2 + b\]

    and then applies a step activation:

    \[
    \hat y =
    \begin{cases}
    1 & \text{if } z \ge 0 \\\\
    0 & \text{otherwise}
    \end{cases}
    \]

    The points where \(z = 0\) form the decision boundary:

    \[
    w_1x_1 + w_2x_2 + b = 0
    \]

    With two inputs, that boundary is always a straight line.

    AND, OR, NAND and NOR can all be split by a single line, so a perceptron can represent them:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("tps/annexes/images/logic-gates.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How does it learn?
    The perceptron starts with some weights and makes a prediction for each training point.

    When it makes a mistake, it updates its weights:

    \[
    w_i \leftarrow w_i + \eta (y - \hat y)x_i
    \]

    and its bias:

    \[
    b \leftarrow b + \eta (y - \hat y)
    \]

    The learning rate \(\eta\) controls the size of each correction.

    - A small learning rate makes small adjustments.
    - A large learning rate makes larger adjustments.

    An **epoch** is one complete pass through the training dataset.

    The learning rate does not change what a perceptron can represent. It changes how far its decision boundary moves after a mistake.
    """)
    return


@app.cell
def _(np):
    def plot_decision_boundary(ax, model, dataset, title, point_size=200, point_alpha=1.0):
        xs = [x for x, _ in dataset]
        ys = [y for _, y in dataset]
        x1_vals = [x[0] for x in xs]
        x2_vals = [x[1] for x in xs]

        for value, color in [(0, "tab:gray"), (1, "tab:green")]:
            mask = [y == value for y in ys]
            ax.scatter(
                [x for x, keep in zip(x1_vals, mask) if keep],
                [x for x, keep in zip(x2_vals, mask) if keep],
                c=color,
                s=point_size,
                alpha=point_alpha,
                edgecolors="black" if point_size >= 100 else "none",
                label=str(value),
                zorder=3,
            )

        w1, w2 = model.weights
        b = model.bias
        grid = np.linspace(-0.5, 1.5, 200)

        if abs(w2) > 1e-9:
            # w1*x1 + w2*x2 + b = 0
            # x2 = -(w1*x1 + b) / w2
            boundary_x2 = -(w1 * grid + b) / w2
            ax.plot(grid, boundary_x2, "k--", zorder=2)
        elif abs(w1) > 1e-9:
            # Vertical boundary: x1 = -b / w1
            ax.axvline(-b / w1, linestyle="--", color="black", zorder=2)

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(1.2, -0.2)  # y inverted: 0 on top, 1 on bottom
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("A")
        ax.set_ylabel("B")
        ax.set_title(title)

    return (plot_decision_boundary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Watching the perceptron learn
    We can see the perceptron adapt after each iteration of its training and see the decision boundary move as the perceptron corrects its mistakes.
    """)
    return


@app.cell
def _(GATES, LogicGatesDataset, Perceptron, np, train_perceptron):
    discrete_models = {}
    discrete_epoch_history = {}

    for _gate in GATES:
        _dataset = LogicGatesDataset(f"data/logic_gates/{_gate}.csv")
        _model = Perceptron(n_inputs=2)
        _history = []

        # Instead of running 20 epochs, we manually iterate epoch by epoch in a for loop to save the training history
        for _epoch in range(20):
            train_perceptron(_model, _dataset, epochs=20, learning_rate=0.1, keep_best=False)
            _history.append({
                "weights": np.asarray(_model.weights, dtype=float).copy(),
                "bias": float(_model.bias),
            })

        # Can be replaced by a simple:
        # train_perceptron(_model, _dataset, epochs=20, learning_rate=0.1, keep_best=False)

        discrete_models[_gate] = _model
        discrete_epoch_history[_gate] = _history

        _predictions = [_model.predict(x) for x, _ in _dataset]
        _expected = [y for _, y in _dataset]
        _status = "OK" if _predictions == _expected else "FAILED"
        print(
            f"{_gate.upper():5s}  "
            f"predictions={_predictions}  "
            f"expected={_expected}  "
            f"[{_status}]"
        )
    return discrete_epoch_history, discrete_models


@app.cell
def _(np):
    def plot_epoch_boundary(ax, snapshot, dataset):
        xs = [x for x, _ in dataset]
        ys = [y for _, y in dataset]
        x1_vals = [x[0] for x in xs]
        x2_vals = [x[1] for x in xs]

        for value, color in [(0, "slategray"), (1, "lime")]:
            mask = [y == value for y in ys]
            ax.scatter(
                [x for x, keep in zip(x1_vals, mask) if keep],
                [x for x, keep in zip(x2_vals, mask) if keep],
                c=color,
                s=50,
                edgecolors="black",
                linewidths=0.5,
                label=str(value),
                zorder=3,
            )

        w1, w2 = snapshot["weights"]
        b = snapshot["bias"]
        grid = np.linspace(-0.5, 1.5, 100)

        if abs(w2) > 1e-9:
            boundary_x2 = -(w1 * grid + b) / w2
            ax.plot(grid, boundary_x2, "k--", linewidth=1, zorder=2)
        elif abs(w1) > 1e-9:
            ax.axvline(-b / w1, linestyle="--", color="black", linewidth=1, zorder=2)

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(1.2, -0.2)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("A", fontsize=7, labelpad=1)
        ax.set_ylabel("B", fontsize=7, labelpad=1)
        ax.tick_params(axis="both", labelsize=7, pad=1)

    return (plot_epoch_boundary,)


@app.cell
def _(
    GATES,
    LogicGatesDataset,
    discrete_epoch_history,
    plot_epoch_boundary,
    plt,
):
    _fig, _axes = plt.subplots(20, 4, figsize=(11, 34), sharex=False, sharey=False)

    for _epoch in range(20):
        for _gate_index, _gate in enumerate(GATES):
            _ax = _axes[_epoch, _gate_index]
            _dataset = LogicGatesDataset(f"data/logic_gates/{_gate}.csv")
            _snapshot = discrete_epoch_history[_gate][_epoch]
            plot_epoch_boundary(_ax, _snapshot, _dataset)

            if _epoch == 0:
                _ax.set_title(_gate.upper(), fontsize=12)

            if _gate_index == 0:
                _ax.text(
                    -0.36,
                    0.5,
                    f"Epoch {_epoch + 1}",
                    transform=_ax.transAxes,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                )

    _handles, _labels = _axes[0, 0].get_legend_handles_labels()
    _fig.legend(
        _handles,
        _labels,
        title="Output",
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.997),
    )

    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once a gate has been separated correctly, later epochs may stop changing the boundary.

    The important part is the mechanism: each wrong prediction causes a weight update, and those updates move or rotate the line.
    """)
    return


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
    XOR cannot be split by any single line because its two "1" points and its two "0" points sit on opposite corners.

    This is not a learning-rate or training-time problem. A single perceptron simply cannot represent XOR.
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

    train_perceptron(xor_discrete_model, xor_corners, epochs=100, learning_rate=0.1, keep_best=False)

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
    Real measurements rarely land on perfectly clean values.

    We can replace the four exact truth-table points with random floating points inside the range [0;1] using `sample_noisy_points`.

    We then train the same perceptron with `keep_best=False`, which means we simply keep whatever weights it has after the last epoch.

    Let's see what happens at different training lengths.
    """)
    return


@app.cell
def _(GATES, LogicGatesDataset, sample_noisy_points):
    noisy_datasets = {
        _gate: sample_noisy_points(
            LogicGatesDataset(f"data/logic_gates/{_gate}.csv"),
            n_per_class=50,
            noise_std=0.05,
            seed=0,
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

    for _epochs in [1,2,3,4,5,10,20, 100, 500, 1000, 2000, 3000, 4000, 5000]:
        _row = []
        for _gate in GATES:
            _model = Perceptron(n_inputs=2)
            train_perceptron(
                _model,
                noisy_datasets[_gate],
                epochs=_epochs,
                learning_rate=0.1,
                seed=42,
                keep_best=False,
            )
            _row.append(f"{accuracy(_model, noisy_datasets[_gate]):.1%}")

        print(f"{_epochs:8d}  " + "  ".join(f"{value:>6s}" for value in _row))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Training the model for longer does not guarantee that it becomes better. It can sometimes lose accuracy. It can move from a good solution to a worse one and back again.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3: Pocket, keep the best version seen
    The pocket algorithm makes one small change to address this problem: while the perceptron trains normally, it also keeps a copy of the best weights seen so far.

    Suppose the accuracy during training looks like this:

    ```text
    epoch 1: 85%   <- saved in the pocket
    epoch 2: 90%   <- saved
    epoch 3: 89%   <- worse than epoch 2, ignored
    epoch 4: 91%   <- saved
    ```

    Epoch 3 is not skipped. Training continues from the current weights, even when they are worse than the weights stored in the pocket. The pocket is only a snapshot of the best result seen so far.

    At the end of the training loop, we can return the best snapshot instead of the last one of the loop.

    `train_perceptron` does this by default with `keep_best=True`.
    """)
    return


@app.cell
def _(GATES, Perceptron, accuracy, noisy_datasets, train_perceptron):
    print(f"{'epochs':>8s}  " + "  ".join(f"{_g.upper():>6s}" for _g in GATES))

    _pockets = {}

    for _epochs in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        _row = []

        for _gate in GATES:
            _model = Perceptron(n_inputs=2)

            _pocket = train_perceptron(
                _model,
                noisy_datasets[_gate],
                epochs=_epochs,
                learning_rate=0.1,
                seed=42,
                keep_best=True,
            )

            _pockets[_gate] = _pocket

            _row.append(
                f"{accuracy(_model, noisy_datasets[_gate]):.1%}"
            )

        print(
            f"{_epochs:8d}  "
            + "  ".join(f"{value:>6s}" for value in _row)
        )


    print("\nPocket:")
    for _gate in GATES:
        _pocket = _pockets[_gate]
        _accuracy = _pocket.correct / len(noisy_datasets[_gate])

        print(
            f"{_gate.upper():5s}  "
            f"epoch={_pocket.epoch:3d}  "
            f"accuracy={_accuracy:.1%}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compared with the plain perceptron, the returned result is much more stable because a good boundary is not lost just because later updates make it worse.

    In this example, a good boundary is often found within the first few epochs. The pocket preserves it if training later wanders away.

    This does **not** mean that pocket inherently makes the perceptron converge faster. It simply means that the final model can use a good solution that appeared earlier in training.

    Pocket also does not guarantee the best possible boundary. It only keeps the best boundary the perceptron actually visited.
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
        plot_decision_boundary(
            _ax,
            final_models[_gate],
            noisy_datasets[_gate],
            _gate.upper(),
            point_size=25,
            point_alpha=0.7,
        )

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The accuracy is no longer necessarily 100%.

    Some noisy points can end up on the wrong side of the decision boundary. That is expected when the classes overlap or cannot be perfectly separated by one line.

    XOR, unsurprisingly, does not get any better from pocket. Its problem was never training speed or stability: no single line can separate it.
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

    print(f"accuracy={accuracy(xor_model, xor_noisy):.2%}")

    _fig, _ax = plt.subplots(figsize=(4, 4))
    plot_decision_boundary(
        _ax,
        xor_model,
        xor_noisy,
        "XOR, noisy + pocket (still fails)",
        point_size=25,
        point_alpha=0.7,
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrap-up
    - A perceptron computes a weighted sum followed by a step activation.
    - With two inputs, it can only learn a straight decision boundary.
    - The learning rate controls the size of each weight correction.
    - An epoch is one complete pass through the training data.
    - AND, OR, NAND and NOR are linearly separable, so one perceptron can learn them.
    - XOR is not linearly separable, so one perceptron cannot learn it.
    - With noisy data, the plain perceptron may keep moving instead of settling on one good solution.
    - Pocket keeps the best boundary seen during training, but it does not search every possible boundary or guarantee a global optimum.

    That last limitation is one reason we move on to a multilayer perceptron (MLP) in `02_mlp_xor.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
