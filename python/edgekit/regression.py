# python/edgekit/regression.py
#
# Regression lineaire et logistique a une variable, par descente de gradient
# "batch", en numpy pur. Sert au TP 3 (notebook 00_regression.py).
#
# Notations du livret :
#   h(x) = theta0 + theta1 * x                          (lineaire)
#   h(x) = sigmoid(theta0 + theta1 * x)                 (logistique)
#   J lineaire    = 1/(2m) * sum (h(x_i) - y_i)^2
#   J logistique  = -1/m * sum [y log h + (1 - y) log(1 - h)]
#   theta_j <- theta_j - alpha * dJ/dtheta_j
#
# Le critere d'arret reprend celui de l'ancien TP (J < eps), avec un garde-fou
# max_iter pour qu'un mauvais reglage ne bloque pas le notebook.

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GDResult:
    theta: np.ndarray                     # [theta0, theta1]
    n_iter: int                           # nombre de cycles effectues
    converged: bool                       # True si J < eps avant max_iter
    history: list[float] = field(default_factory=list)  # J a chaque cycle


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


def h_linear(theta: np.ndarray, x: np.ndarray | float) -> np.ndarray | float:
    return theta[0] + theta[1] * x


def h_logistic(theta: np.ndarray, x: np.ndarray | float) -> np.ndarray | float:
    return sigmoid(theta[0] + theta[1] * x)


def cost_linear(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    r = h_linear(theta, x) - y
    return float(np.mean(r * r) / 2.0)


def cost_logistic(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    h = np.clip(h_logistic(theta, x), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(h) + (1.0 - y) * np.log(1.0 - h)))


def _gradient_descent(h, cost, x, y, alpha, eps, max_iter, theta0) -> GDResult:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    theta = np.array(theta0, dtype=np.float64)
    history = [cost(theta, x, y)]
    n = 0
    with np.errstate(over="ignore", invalid="ignore"):
        while history[-1] > eps and n < max_iter:
            # Meme gradient pour les deux modeles : (1/m) sum (h - y) * [1, x]
            r = h(theta, x) - y
            grad = np.array([np.mean(r), np.mean(r * x)])
            theta = theta - alpha * grad      # mise a jour simultanee de theta0 et theta1
            history.append(cost(theta, x, y))
            n += 1
            if not np.isfinite(history[-1]):
                break                         # divergence : alpha trop grand
    return GDResult(theta=theta, n_iter=n, converged=history[-1] <= eps, history=history)


def fit_linear_gd(x, y, alpha: float, eps: float, max_iter: int = 1_000_000,
                  theta0=(0.0, 0.0)) -> GDResult:
    return _gradient_descent(h_linear, cost_linear, x, y, alpha, eps, max_iter, theta0)


def fit_logistic_gd(x, y, alpha: float, eps: float, max_iter: int = 1_000_000,
                    theta0=(0.0, 0.0)) -> GDResult:
    return _gradient_descent(h_logistic, cost_logistic, x, y, alpha, eps, max_iter, theta0)


def fit_linear_closed_form(x, y) -> np.ndarray:
    """Moindres carres exacts (equation normale), pour verifier la descente de gradient."""
    x = np.asarray(x, dtype=np.float64)
    a = np.column_stack([np.ones_like(x), x])
    theta, *_ = np.linalg.lstsq(a, np.asarray(y, dtype=np.float64), rcond=None)
    return theta


def theta_to_c_array(theta: np.ndarray, name: str = "theta") -> str:
    """Parametres sous forme de tableau C, a copier dans un sketch ESP32."""
    values = ", ".join(f"{v:.6f}f" for v in theta)
    return f"const float {name}[{len(theta)}] = {{{values}}};"
