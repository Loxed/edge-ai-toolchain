import numpy as np

from edgekit.datasets.old_labs import load_line_camera, load_motors
from edgekit.regression import (
    cost_linear,
    fit_linear_closed_form,
    fit_linear_gd,
    fit_logistic_gd,
    h_logistic,
    theta_to_c_array,
)


def test_motor_dataset_shape():
    x, y = load_motors()
    assert x.shape == y.shape == (24,)
    xl, yl = load_line_camera()
    assert xl.shape == yl.shape == (10,)
    assert set(yl.tolist()) == {0.0, 1.0}


def test_linear_gd_matches_closed_form_on_exact_line():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 + 0.5 * x
    r = fit_linear_gd(x, y, alpha=0.1, eps=1e-10, max_iter=100_000)
    assert r.converged
    np.testing.assert_allclose(r.theta, [2.0, 0.5], atol=1e-4)
    np.testing.assert_allclose(fit_linear_closed_form(x, y), [2.0, 0.5], atol=1e-12)


def test_linear_gd_cost_decreases_on_motors():
    x, y = load_motors()
    r = fit_linear_gd(x, y, alpha=5e-4, eps=0.5)
    assert r.converged
    assert all(a >= b for a, b in zip(r.history, r.history[1:]))
    # Le critere d'arret laxiste laisse theta0 loin de l'optimum : c'est le point du TP.
    assert cost_linear(fit_linear_closed_form(x, y), x, y) < r.history[-1]


def test_linear_gd_diverges_when_alpha_too_large():
    x, y = load_motors()
    r = fit_linear_gd(x, y, alpha=1e-3, eps=0.5, max_iter=10_000)
    assert not r.converged


def test_logistic_gd_separates_line_camera():
    x, y = load_line_camera()
    r = fit_logistic_gd(x, y, alpha=1e-3, eps=0.05, max_iter=100_000)
    assert r.converged
    pred = h_logistic(r.theta, x) > 0.5
    assert np.array_equal(pred, y.astype(bool))


def test_theta_to_c_array():
    s = theta_to_c_array(np.array([1.5, -0.25]), name="th")
    assert s == "const float th[2] = {1.500000f, -0.250000f};"
