"""Behaviour that does not need PyTorch: convergence, input checks, known closed forms."""

import numpy as np
import pytest

from fromscratch import optim

LRS = {"sgd": 0.1, "momentum": 0.05, "nesterov": 0.05, "adagrad": 0.5, "rmsprop": 0.01,
       "adam": 0.05, "adamw": 0.05}


@pytest.mark.parametrize("name", optim.NAMES)
def test_every_optimizer_minimises_a_quadratic(name):
    target = np.array([3.0, -2.0, 0.5])
    p = np.zeros(3)
    opt = optim.build(name, [p], lr=LRS[name])
    start = np.sum((p - target) ** 2)
    for _ in range(1000):
        opt.step([2 * (p - target)])
    assert np.sum((p - target) ** 2) < 1e-2 * start


def test_parameters_are_updated_in_place():
    p = np.ones(4)
    alias = p
    optim.SGD([p], lr=0.5).step([np.ones(4)])
    assert alias is p and np.allclose(p, 0.5)


def test_none_gradient_leaves_parameter_untouched():
    a, b = np.ones(2), np.ones(2)
    optim.Adam([a, b], lr=0.1).step([np.ones(2), None])
    assert np.all(a < 1) and np.all(b == 1)


def test_bad_inputs_raise():
    p = np.zeros(3)
    with pytest.raises(ValueError):
        optim.SGD([p]).step([np.zeros(4)])                 # wrong shape
    with pytest.raises(ValueError):
        optim.SGD([p]).step([np.zeros(3), np.zeros(3)])    # wrong count
    with pytest.raises(TypeError):
        optim.SGD([np.zeros(3, dtype=int)])                # integer parameters
    with pytest.raises(ValueError):
        optim.SGD([p], nesterov=True)                      # Nesterov without momentum
    with pytest.raises(ValueError):
        optim.build("lbfgs", [p])


def test_adam_first_step_is_exactly_lr_in_size():
    # With bias correction, step 1 gives m_hat = g and v_hat = g^2, so |dp| = lr * |g| / (|g| + eps).
    for g in (1e-3, 1.0, 1e3):
        p = np.zeros(1)
        optim.Adam([p], lr=0.01).step([np.array([g])])
        assert np.isclose(-p[0], 0.01 * g / (g + 1e-8))


def test_adam_without_bias_correction_matches_closed_form():
    # Constant gradient: m_t = (1 - b1^t) g and v_t = (1 - b2^t) g^2, so the uncorrected step is
    # lr * (1 - b1^t) / sqrt(1 - b2^t): over 6x larger than lr around t = 10.
    b1, b2, lr = 0.9, 0.999, 1e-3
    p = np.zeros(1)
    opt = optim.Adam([p], lr=lr, bias_correction=False, eps=0.0)
    for t in range(1, 51):
        before = p.copy()
        opt.step([np.ones(1)])
        assert np.isclose(before[0] - p[0], lr * (1 - b1 ** t) / np.sqrt(1 - b2 ** t))


def test_adamw_decay_with_zero_gradient_is_exponential():
    p = np.ones(1)
    opt = optim.AdamW([p], lr=0.1, weight_decay=0.5)
    for _ in range(10):
        opt.step([np.zeros(1)])
    assert np.isclose(p[0], (1 - 0.1 * 0.5) ** 10)


@pytest.mark.parametrize("name, limit", [
    ("sgd", 2.0),                        # plain gradient descent: lr * curvature < 2
    ("momentum", 2 * (1 + 0.9)),         # heavy ball: lr * curvature < 2 (1 + beta)
    ("nesterov", 2 * (1 + 0.9) / (1 + 2 * 0.9)),   # Nesterov: 2 (1 + beta) / (1 + 2 beta)
])
def test_stability_limit_on_a_quadratic(name, limit):
    # On f(x) = curvature/2 * x^2 these methods converge exactly when lr * curvature < limit.
    curvature = 50.0

    def final_size(lr_times_curvature):
        p = np.array([1.0])
        opt = optim.build(name, [p], lr=lr_times_curvature / curvature)
        for _ in range(3000):
            opt.step([curvature * p])
        return abs(p[0])

    assert final_size(0.98 * limit) < 1e-3
    assert final_size(1.02 * limit) > 1e3
