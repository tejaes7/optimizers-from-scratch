"""A small multilayer perceptron with hand-written backpropagation (no autograd).

Architecture: Linear -> ReLU -> ... -> Linear -> softmax cross-entropy.
tests/test_gradcheck.py checks every gradient against finite differences.
"""

import numpy as np


class MLP:
    def __init__(self, sizes, seed=0, dtype=np.float64):
        """sizes: layer widths including input and output, e.g. [64, 128, 64, 10]."""
        if len(sizes) < 2:
            raise ValueError("need at least an input and an output size")
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # He initialisation: keeps the activation variance roughly constant through ReLUs.
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)).astype(dtype))
            self.b.append(np.zeros(fan_out, dtype=dtype))

    @property
    def params(self):
        """[W0, b0, W1, b1, ...]: the arrays themselves, so optimizers update them in place."""
        return [arr for pair in zip(self.W, self.b) for arr in pair]

    def logits(self, X):
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
        return h

    def predict(self, X):
        return self.logits(X).argmax(axis=1)

    def loss_and_grads(self, X, y):
        """Mean cross-entropy loss and its gradient w.r.t. every parameter (same order as params)."""
        n, L = X.shape[0], len(self.W)

        # Forward pass, keeping what the backward pass needs.
        inputs, pre = [], []          # inputs[i] = input to layer i, pre[i] = its pre-activation
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            inputs.append(h)
            z = h @ W + b
            pre.append(z)
            h = np.maximum(z, 0.0) if i < L - 1 else z

        # Softmax cross-entropy, computed stably via log-sum-exp.
        shifted = h - h.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        loss = -log_probs[np.arange(n), y].mean()

        # Backward pass. For softmax + cross-entropy, dL/dlogits = (softmax - one_hot) / n.
        dz = np.exp(log_probs)
        dz[np.arange(n), y] -= 1.0
        dz /= n
        dW, db = [None] * L, [None] * L
        for i in reversed(range(L)):
            dW[i] = inputs[i].T @ dz          # z = x W + b  =>  dL/dW = x^T dL/dz
            db[i] = dz.sum(axis=0)            #                  dL/db = sum over the batch
            if i > 0:
                dh = dz @ self.W[i].T         #                  dL/dx = dL/dz W^T
                dz = dh * (pre[i - 1] > 0)    # ReLU passes gradient only where its input was > 0

        grads = [g for pair in zip(dW, db) for g in pair]
        return loss, grads
