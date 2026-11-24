"""Shared helpers for the experiments: plot style, saving, and the MLP training loop."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fromscratch import MLP, build
from fromscratch.data import batches

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

# One fixed colour per optimizer (validated categorical order, never re-assigned by rank).
COLORS = {
    "sgd": "#2a78d6", "momentum": "#eb6834", "nesterov": "#1baf7a", "adagrad": "#eda100",
    "rmsprop": "#e87ba4", "adam": "#008300", "adamw": "#6250d6",
}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9.5,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "lines.linewidth": 1.8, "legend.frameon": False, "legend.fontsize": 8.5,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")


def write_json(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2))
    print(f"saved results/{name}")


def accuracy(model, X, y):
    return float((model.predict(X) == y).mean())


def train_mlp(name, lr, data, seed=0, epochs=30, batch_size=64, sizes=(64, 128, 64, 10),
              max_steps=None, **opt_kwargs):
    """Train the MLP with one optimizer. Returns per-epoch curves and final metrics.

    A run counts as diverged if the loss becomes non-finite or exceeds 1e3; its accuracy is
    then reported as that of the final (broken) model, which is near chance.
    """
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = data
    model = MLP(list(sizes), seed=seed)
    opt = build(name, model.params, lr=lr, **opt_kwargs)
    rng = np.random.default_rng(1000 + seed)
    epoch_loss, step_loss, val_acc = [], [], []
    diverged = False
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(epochs):
            losses = []
            for Xb, yb in batches(X_tr, y_tr, batch_size, rng):
                loss, grads = model.loss_and_grads(Xb, yb)
                if not np.isfinite(loss) or loss > 1e3:
                    diverged = True
                    break
                losses.append(loss)
                step_loss.append(float(loss))
                opt.step(grads)
                if max_steps and len(step_loss) >= max_steps:
                    break
            if diverged:
                break
            epoch_loss.append(float(np.mean(losses)))
            val_acc.append(accuracy(model, X_val, y_val))
            if max_steps and len(step_loss) >= max_steps:
                break
    with np.errstate(over="ignore", invalid="ignore"):
        final_val = accuracy(model, X_val, y_val)
        final_test = accuracy(model, X_te, y_te)
    return {
        "optimizer": name, "lr": lr, "seed": seed, "diverged": diverged,
        "epoch_loss": epoch_loss, "step_loss": step_loss, "val_acc": val_acc,
        "final_train_loss": epoch_loss[-1] if epoch_loss and not diverged else None,
        "final_val_acc": final_val, "final_test_acc": final_test,
    }
