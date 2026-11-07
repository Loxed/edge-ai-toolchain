import marimo

app = marimo.App()

@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)

@app.cell
def _(plt):
    plt.plot([1, 2, 3])
    plt.show()
    return
