"""The hand-written backpropagation must agree with finite differences."""

import numpy as np

from fromscratch.functions import make_quadratic, numerical_grad, rosenbrock
from fromscratch.mlp import MLP


def relative_error(a, b):
    return np.abs(a - b).max() / max(1e-12, np.abs(a).max() + np.abs(b).max())


def test_mlp_backprop_matches_finite_differences():
    rng = np.random.default_rng(1)
    model = MLP([5, 7, 6, 4], seed=3)
    X = rng.normal(size=(9, 5))
    y = rng.integers(0, 4, size=9)
    _, grads = model.loss_and_grads(X, y)

    h = 1e-6
    for p, g in zip(model.params, grads):
        numeric = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + h
            loss_plus, _ = model.loss_and_grads(X, y)
            p[idx] = old - h
            loss_minus, _ = model.loss_and_grads(X, y)
            p[idx] = old
            numeric[idx] = (loss_plus - loss_minus) / (2 * h)
        assert relative_error(numeric, g) < 1e-6


def test_initial_loss_is_close_to_log_num_classes():
    # With small random logits, softmax is near uniform, so the loss starts near log(10).
    rng = np.random.default_rng(0)
    model = MLP([64, 32, 10], seed=0)
    for W in model.W:
        W *= 0.01
    loss, _ = model.loss_and_grads(rng.normal(size=(50, 64)), rng.integers(0, 10, size=50))
    assert abs(loss - np.log(10)) < 0.01


def test_test_function_gradients():
    rng = np.random.default_rng(2)
    quad = make_quadratic(condition_number=50, angle=30)
    for _ in range(5):
        p = rng.normal(size=2)
        for f in (rosenbrock, quad):
            assert relative_error(numerical_grad(f, p), f(p)[1]) < 1e-6
