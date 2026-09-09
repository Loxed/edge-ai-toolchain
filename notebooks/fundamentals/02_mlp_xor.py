import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02: The MLP solves XOR

    The previous notebook (`01_perceptron_and_or.ipynb`) showed that a
    perceptron can only draw a linear decision boundary, and that XOR isn't
    linearly separable. Here we add a hidden layer (multilayer perceptron,
    MLP), and show that the decision boundary becomes non-linear, which
    makes learning XOR possible.

    `MLP` and `train_mlp` come from the `edgekit` package (from-scratch
    implementation, no PyTorch, covered by the unit tests in
    `tests/unit/models/mlp/`). This notebook orchestrates and visualizes, it
    doesn't implement anything itself.
    """)
    return


@app.cell
def _():
    # Move to the repo root regardless of where Jupyter's kernel started
    # (a Jupyter kernel's default cwd is often the notebook's own folder,
    # not the repo root). Located via the presence of pyproject.toml.
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
    from edgekit.models.mlp import MLP
    from edgekit.training.mlp import train_mlp
    from edgekit.datasets.synthetic import sample_noisy_points

    return LogicGatesDataset, MLP, np, plt, sample_noisy_points, train_mlp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the MLP on XOR

    Same noisy-cloud treatment as in notebook 01: each of the 4 truth-table
    corners becomes ~30 points scattered around 0.4/0.6 on each axis
    (`sample_noisy_points`), instead of training on the 4 exact points.

    Architecture: 2 inputs -> 4 hidden neurons (sigmoid) -> 1 output
    (sigmoid). We track `history["loss"]` to confirm the training actually
    converges, not just that the final prediction happens to be correct.
    """)
    return


@app.cell
def _(LogicGatesDataset, MLP, sample_noisy_points, train_mlp):
    xor_corners = LogicGatesDataset("data/logic_gates/xor.csv")
    xor_dataset = sample_noisy_points(xor_corners, n_per_class=30, noise_std=0.05, seed=0)

    model = MLP(n_inputs=2, n_hidden=4, n_outputs=1, seed=42)

    history = train_mlp(model, xor_dataset, epochs=1000, learning_rate=0.5)

    predictions = [model.predict(x) for x, _ in xor_dataset]
    expected = [y for _, y in xor_dataset]
    accuracy = sum(p == e for p, e in zip(predictions, expected)) / len(xor_dataset)

    print(f"accuracy={accuracy:.2%}")
    print("MLP learns XOR:", accuracy > 0.9)
    return history, model, xor_dataset


@app.cell
def _(history, plt):
    plt.figure(figsize=(5, 3))
    plt.plot(history["loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss (mean MSE)")
    plt.title("Training convergence on XOR")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing the decision boundary

    Unlike the perceptron, the MLP's boundary is no longer a straight line:
    it can't be expressed as a simple `w1*x1 + w2*x2 + b = 0` equation
    anymore. So we visualize it differently: we compute the model's
    prediction over a whole grid of points in the (x1, x2) plane, and shade
    each region by its predicted class. This is the standard technique for
    visualizing a non-linear decision boundary in 2D.
    """)
    return


@app.cell
def _(model, np, plt, xor_dataset):
    def plot_mlp_decision_regions(ax, model, dataset, title, resolution=200):
        grid = np.linspace(-0.5, 1.5, resolution)
        xx1, xx2 = np.meshgrid(grid, grid)
        zz = np.zeros_like(xx1)
        for i in range(resolution):
            for j in range(resolution):
                zz[i, j] = model.predict([xx1[i, j], xx2[i, j]])
        ax.contourf(xx1, xx2, zz, levels=[-0.5, 0.5, 1.5], colors=['#f4b6b6', '#b6c6f4'], alpha=0.8)
        xs = [x for x, _ in dataset]
        ys = [y for _, y in dataset]
        x1_vals = [x[0] for x in xs]
        x2_vals = [x[1] for x in xs]
        colors = ['tab:red' if y == 0 else 'tab:blue' for y in ys]
        ax.scatter(x1_vals, x2_vals, c=colors, s=25, alpha=0.7, edgecolors='none', zorder=3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(title)
    _fig, ax = plt.subplots(figsize=(5, 5))
    plot_mlp_decision_regions(ax, model, xor_dataset, 'MLP on XOR')
    plt.show()
    return (plot_mlp_decision_regions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Direct comparison with the perceptron

    We retrain a plain perceptron on XOR (as in notebook 01) and show both
    boundaries side by side: the failing line, and the successful
    non-linear region.
    """)
    return


@app.cell
def _(model, np, plot_mlp_decision_regions, plt, xor_dataset):
    from edgekit.models.perceptron import Perceptron
    from edgekit.training.perceptron import train_perceptron
    perceptron_model = Perceptron(n_inputs=2)
    train_perceptron(perceptron_model, xor_dataset, epochs=100, learning_rate=0.1)

    def plot_perceptron_boundary(ax, model, dataset, title):
        xs = [x for x, _ in dataset]
        ys = [y for _, y in dataset]
        x1_vals = [x[0] for x in xs]
        x2_vals = [x[1] for x in xs]
        colors = ['tab:red' if y == 0 else 'tab:blue' for y in ys]
        ax.scatter(x1_vals, x2_vals, c=colors, s=25, alpha=0.7, edgecolors='none', zorder=3)
        w1, w2 = model.weights
        b = model.bias
        grid = np.linspace(-0.5, 1.5, 200)
        if abs(w2) > 1e-09:
            boundary_x2 = -(w1 * grid + b) / w2
            ax.plot(grid, boundary_x2, 'k--', zorder=2)
        elif abs(w1) > 1e-09:
            ax.axvline(-b / w1, linestyle='--', color='black', zorder=2)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(title)
    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_perceptron_boundary(axes[0], perceptron_model, xor_dataset, 'Perceptron (fails)')
    plot_mlp_decision_regions(axes[1], model, xor_dataset, 'MLP (succeeds)')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    The hidden layer lets the MLP compose several linear boundaries into a
    non-linear global boundary, enough to separate XOR. This is the same
    core mechanism we'll see again, at a much larger scale, in the networks
    used for MNIST (`vision/01_mnist_mlp.ipynb`, then
    `vision/02_mnist_cnn.ipynb`).
    """)
    return


if __name__ == "__main__":
    app.run()