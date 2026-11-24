"""2-D test functions with analytic gradients, for visualising optimizer trajectories.

Each function takes a parameter vector p = [x, y] and returns (value, gradient).
"""

import numpy as np


def rosenbrock(p, a=1.0, b=100.0):
    """f(x, y) = (a - x)^2 + b (y - x^2)^2. Minimum 0 at (a, a^2), inside a long curved valley."""
    x, y = p
    f = (a - x) ** 2 + b * (y - x * x) ** 2
    grad = np.array([-2 * (a - x) - 4 * b * x * (y - x * x), 2 * b * (y - x * x)])
    return f, grad


def rotation(degrees):
    th = np.deg2rad(degrees)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def make_quadratic(condition_number=100.0, angle=0.0):
    """f(p) = 1/2 p^T H p with H = R diag(1, kappa) R^T: an elongated bowl rotated by `angle`.

    Minimum 0 at the origin. Rotating the bowl does not change the problem's difficulty for
    any rotation-invariant method (plain SGD, momentum), but it does for methods that scale
    each coordinate separately (AdaGrad, RMSProp, Adam).
    """
    R = rotation(angle)
    H = R @ np.diag([1.0, condition_number]) @ R.T

    def f(p):
        Hp = H @ p
        return 0.5 * float(p @ Hp), Hp

    f.H = H
    f.R = R
    return f


def numerical_grad(f, p, h=1e-6):
    """Central-difference gradient, used by the tests."""
    p = np.asarray(p, dtype=float)
    g = np.zeros_like(p)
    for i in range(p.size):
        e = np.zeros_like(p)
        e.flat[i] = h
        g.flat[i] = (f(p + e)[0] - f(p - e)[0]) / (2 * h)
    return g
