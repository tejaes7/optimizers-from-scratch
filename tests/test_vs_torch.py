"""Every optimizer must produce the same parameters as torch.optim, step by step.

Both sides start from the same parameters. At each step the gradient is computed from that
side's own current parameters, so any mismatch in an update rule would compound over the
200 steps instead of hiding. Everything runs in float64.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fromscratch import optim as ours

SHAPES = [(3, 4), (5,), (2, 3, 2)]
STEPS = 200
TOL = 1e-10

CASES = {
    "sgd": (lambda p: ours.SGD(p, lr=0.05),
            lambda p: torch.optim.SGD(p, lr=0.05)),
    "sgd_momentum": (lambda p: ours.SGD(p, lr=0.02, momentum=0.9),
                     lambda p: torch.optim.SGD(p, lr=0.02, momentum=0.9)),
    "sgd_momentum_dampening_wd": (
        lambda p: ours.SGD(p, lr=0.02, momentum=0.8, dampening=0.3, weight_decay=0.01),
        lambda p: torch.optim.SGD(p, lr=0.02, momentum=0.8, dampening=0.3, weight_decay=0.01)),
    "nesterov_wd": (
        lambda p: ours.SGD(p, lr=0.02, momentum=0.9, nesterov=True, weight_decay=0.01),
        lambda p: torch.optim.SGD(p, lr=0.02, momentum=0.9, nesterov=True, weight_decay=0.01)),
    "adagrad": (lambda p: ours.AdaGrad(p, lr=0.1),
                lambda p: torch.optim.Adagrad(p, lr=0.1)),
    "adagrad_decay_wd_init": (
        lambda p: ours.AdaGrad(p, lr=0.1, lr_decay=0.01, weight_decay=0.01,
                               initial_accumulator_value=0.1),
        lambda p: torch.optim.Adagrad(p, lr=0.1, lr_decay=0.01, weight_decay=0.01,
                                      initial_accumulator_value=0.1)),
    "rmsprop": (lambda p: ours.RMSProp(p, lr=0.01),
                lambda p: torch.optim.RMSprop(p, lr=0.01)),
    "rmsprop_centered_momentum_wd": (
        lambda p: ours.RMSProp(p, lr=0.005, alpha=0.9, centered=True, momentum=0.9,
                               weight_decay=0.01),
        lambda p: torch.optim.RMSprop(p, lr=0.005, alpha=0.9, centered=True, momentum=0.9,
                                      weight_decay=0.01)),
    "adam": (lambda p: ours.Adam(p, lr=0.01),
             lambda p: torch.optim.Adam(p, lr=0.01)),
    "adam_l2_amsgrad_betas": (
        lambda p: ours.Adam(p, lr=0.01, betas=(0.8, 0.99), weight_decay=0.01, amsgrad=True),
        lambda p: torch.optim.Adam(p, lr=0.01, betas=(0.8, 0.99), weight_decay=0.01,
                                   amsgrad=True)),
    "adamw": (lambda p: ours.AdamW(p, lr=0.01, weight_decay=0.1),
              lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1)),
    "adamw_amsgrad": (lambda p: ours.AdamW(p, lr=0.01, weight_decay=0.1, amsgrad=True),
                      lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1, amsgrad=True)),
    "adam_decoupled_equals_adamw": (
        lambda p: ours.Adam(p, lr=0.01, weight_decay=0.1, decoupled_weight_decay=True),
        lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1)),
}


def grad_fn(params, noise):
    """A gradient that depends non-linearly on the parameters, plus fixed noise."""
    return [np.cos(3 * p) + 0.3 * p + n for p, n in zip(params, noise)]


@pytest.mark.parametrize("case", list(CASES))
def test_matches_torch(case):
    make_ours, make_torch = CASES[case]
    rng = np.random.default_rng(0)
    init = [rng.normal(size=s) for s in SHAPES]
    noise = [[rng.normal(scale=0.1, size=s) for s in SHAPES] for _ in range(STEPS)]

    our_params = [x.copy() for x in init]
    torch_params = [torch.tensor(x.copy(), dtype=torch.float64, requires_grad=True) for x in init]
    opt_ours, opt_torch = make_ours(our_params), make_torch(torch_params)

    for t in range(STEPS):
        opt_ours.step(grad_fn(our_params, noise[t]))

        current = [tp.detach().numpy() for tp in torch_params]
        for tp, g in zip(torch_params, grad_fn(current, noise[t])):
            tp.grad = torch.from_numpy(g)
        opt_torch.step()

        for ours_p, tp in zip(our_params, torch_params):
            np.testing.assert_allclose(ours_p, tp.detach().numpy(), rtol=0, atol=TOL,
                                       err_msg=f"{case}: mismatch at step {t + 1}")
