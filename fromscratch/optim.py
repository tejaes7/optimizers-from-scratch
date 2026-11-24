"""Gradient-descent optimizers written from scratch in NumPy.

Every optimizer follows the same contract:

    opt = Adam(params, lr=1e-3)   # params: list of NumPy arrays, updated IN PLACE
    grads = ...                   # list of arrays, same order and shapes as params
    opt.step(grads)

The update rules follow the PyTorch reference implementations (torch.optim) exactly,
including their small conventions (where eps goes, how the first momentum step works),
so tests/test_vs_torch.py can check every optimizer against PyTorch step by step.

Notation used in the comments:
    p   parameter          g   gradient of the loss w.r.t. p
    lr  learning rate      t   step number (1, 2, 3, ...)
    m   first moment (running mean of g)
    v   second moment (running mean of g**2)
"""

import math

import numpy as np


class Optimizer:
    """Base class: stores the parameters and one state dict per parameter."""

    def __init__(self, params, **hyperparams):
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        for p in self.params:
            if not isinstance(p, np.ndarray) or not np.issubdtype(p.dtype, np.floating):
                raise TypeError("every parameter must be a floating-point NumPy array")
        for name, value in hyperparams.items():
            setattr(self, name, value)          # e.g. self.lr, so a schedule can change it
        self.state = [{} for _ in self.params]  # per-parameter buffers (momentum, m, v, ...)

    def step(self, grads):
        grads = list(grads)
        if len(grads) != len(self.params):
            raise ValueError(f"expected {len(self.params)} gradients, got {len(grads)}")
        for p, g, state in zip(self.params, grads, self.state):
            if g is None:                        # parameter not used this step: leave it alone
                continue
            g = np.asarray(g, dtype=p.dtype)
            if g.shape != p.shape:
                raise ValueError(f"gradient shape {g.shape} does not match parameter {p.shape}")
            self._update(p, g, state)

    def _update(self, p, g, state):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with momentum or Nesterov momentum.

    plain:     p <- p - lr * g
    momentum:  b <- mu * b + (1 - dampening) * g      (b = g on the first step)
               p <- p - lr * b
    Nesterov:  p <- p - lr * (g + mu * b)             (look-ahead form used by PyTorch)
    """

    def __init__(self, params, lr=1e-2, momentum=0.0, dampening=0.0, nesterov=False,
                 weight_decay=0.0):
        if lr < 0 or momentum < 0 or weight_decay < 0:
            raise ValueError("lr, momentum and weight_decay must be non-negative")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum needs momentum > 0 and dampening = 0")
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                         nesterov=nesterov, weight_decay=weight_decay)

    def _update(self, p, g, state):
        if self.weight_decay:
            g = g + self.weight_decay * p                    # gradient of (wd/2) * ||p||^2
        if self.momentum:
            buf = state.get("momentum_buffer")
            if buf is None:
                buf = state["momentum_buffer"] = g.copy()    # first step: no dampening
            else:
                buf *= self.momentum
                buf += (1 - self.dampening) * g
            g = g + self.momentum * buf if self.nesterov else buf
        p -= self.lr * g


class AdaGrad(Optimizer):
    """AdaGrad (Duchi et al., 2011): per-coordinate step sizes from the SUM of squared gradients.

    s <- s + g**2
    p <- p - lr_t * g / (sqrt(s) + eps),   lr_t = lr / (1 + (t - 1) * lr_decay)

    Because s only grows, the effective step size only shrinks; this suits sparse features
    and is why it can stall on long non-convex runs.
    """

    def __init__(self, params, lr=1e-2, lr_decay=0.0, weight_decay=0.0,
                 initial_accumulator_value=0.0, eps=1e-10):
        if lr < 0 or lr_decay < 0 or weight_decay < 0 or eps < 0:
            raise ValueError("hyperparameters must be non-negative")
        super().__init__(params, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                         initial_accumulator_value=initial_accumulator_value, eps=eps)

    def _update(self, p, g, state):
        if not state:
            state["step"] = 0
            state["sum"] = np.full_like(p, self.initial_accumulator_value)
        state["step"] += 1
        if self.weight_decay:
            g = g + self.weight_decay * p
        lr_t = self.lr / (1 + (state["step"] - 1) * self.lr_decay)
        state["sum"] += g * g
        p -= lr_t * g / (np.sqrt(state["sum"]) + self.eps)


class RMSProp(Optimizer):
    """RMSProp (Hinton, 2012): like AdaGrad but with a DECAYING average of squared gradients.

    v <- alpha * v + (1 - alpha) * g**2
    p <- p - lr * g / (sqrt(v) + eps)

    centered=True subtracts the squared running mean of g, so the denominator estimates the
    gradient's variance instead of its raw second moment.
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0,
                 centered=False):
        if lr < 0 or not 0 <= alpha <= 1 or eps < 0 or weight_decay < 0 or momentum < 0:
            raise ValueError("invalid RMSProp hyperparameter")
        super().__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                         momentum=momentum, centered=centered)

    def _update(self, p, g, state):
        if not state:
            state["square_avg"] = np.zeros_like(p)
            if self.centered:
                state["grad_avg"] = np.zeros_like(p)
            if self.momentum:
                state["momentum_buffer"] = np.zeros_like(p)
        if self.weight_decay:
            g = g + self.weight_decay * p
        v = state["square_avg"]
        v *= self.alpha
        v += (1 - self.alpha) * g * g
        if self.centered:
            mean = state["grad_avg"]
            mean *= self.alpha
            mean += (1 - self.alpha) * g
            denom = np.sqrt(v - mean * mean) + self.eps
        else:
            denom = np.sqrt(v) + self.eps
        if self.momentum:
            buf = state["momentum_buffer"]
            buf *= self.momentum
            buf += g / denom
            p -= self.lr * buf
        else:
            p -= self.lr * g / denom


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015): momentum (m) + RMSProp-style scaling (v) + bias correction.

    m <- b1 * m + (1 - b1) * g
    v <- b2 * v + (1 - b2) * g**2
    m_hat = m / (1 - b1**t),   v_hat = v / (1 - b2**t)        # bias correction
    p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    m and v start at zero, so early on they are biased towards zero; dividing by
    (1 - b**t) removes that bias. experiments/adam_bias_correction.py shows what goes wrong
    without it (bias_correction=False, an option PyTorch does not have).

    amsgrad=True (Reddi et al., 2018) uses the running MAXIMUM of v, so the effective step
    size can never grow.

    weight_decay is classic L2 (added to g, so it also gets divided by sqrt(v_hat)) unless
    decoupled_weight_decay=True, which is AdamW: see the AdamW class below.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 amsgrad=False, decoupled_weight_decay=False, bias_correction=True):
        b1, b2 = betas
        if lr < 0 or eps < 0 or weight_decay < 0 or not (0 <= b1 < 1 and 0 <= b2 < 1):
            raise ValueError("invalid Adam hyperparameter")
        super().__init__(params, lr=lr, betas=(b1, b2), eps=eps, weight_decay=weight_decay,
                         amsgrad=amsgrad, decoupled_weight_decay=decoupled_weight_decay,
                         bias_correction=bias_correction)

    def _update(self, p, g, state):
        if not state:
            state["step"] = 0
            state["m"] = np.zeros_like(p)
            state["v"] = np.zeros_like(p)
            if self.amsgrad:
                state["v_max"] = np.zeros_like(p)
        state["step"] += 1
        t = state["step"]
        b1, b2 = self.betas

        if self.weight_decay:
            if self.decoupled_weight_decay:
                p *= 1 - self.lr * self.weight_decay     # AdamW: shrink p directly
            else:
                g = g + self.weight_decay * p            # L2: goes through the adaptive scaling

        m, v = state["m"], state["v"]
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * g * g
        if self.amsgrad:
            np.maximum(state["v_max"], v, out=state["v_max"])
            v = state["v_max"]

        if self.bias_correction:
            bc1 = 1 - b1 ** t
            bc2 = 1 - b2 ** t
        else:
            bc1 = bc2 = 1.0
        denom = np.sqrt(v) / math.sqrt(bc2) + self.eps   # = sqrt(v_hat) + eps
        p -= (self.lr / bc1) * m / denom                 # = lr * m_hat / (sqrt(v_hat) + eps)


class AdamW(Adam):
    """AdamW (Loshchilov & Hutter, 2019): Adam with DECOUPLED weight decay.

    p <- p * (1 - lr * wd)        then the normal Adam step on g (without the wd term)

    With L2 inside Adam, the decay term wd * p is divided by sqrt(v_hat), so weights with
    large gradients are barely regularised. AdamW decays every weight at the same rate.
    See experiments/adamw_vs_l2.py.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                 amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         amsgrad=amsgrad, decoupled_weight_decay=True)


def build(name, params, **kwargs):
    """Build an optimizer by its short name, as used by the experiments."""
    presets = {
        "sgd": lambda: SGD(params, **kwargs),
        "momentum": lambda: SGD(params, momentum=0.9, **kwargs),
        "nesterov": lambda: SGD(params, momentum=0.9, nesterov=True, **kwargs),
        "adagrad": lambda: AdaGrad(params, **kwargs),
        "rmsprop": lambda: RMSProp(params, **kwargs),
        "adam": lambda: Adam(params, **kwargs),
        "adamw": lambda: AdamW(params, **kwargs),
    }
    if name not in presets:
        raise ValueError(f"unknown optimizer {name!r}; choose from {sorted(presets)}")
    return presets[name]()


NAMES = ["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adam", "adamw"]
LABELS = {
    "sgd": "SGD", "momentum": "SGD + momentum", "nesterov": "Nesterov", "adagrad": "AdaGrad",
    "rmsprop": "RMSProp", "adam": "Adam", "adamw": "AdamW",
}
