"""Experiment 1: every optimizer on the Rosenbrock valley, each with its own tuned learning rate.

The learning rate for each optimizer is picked by grid search (the one with the lowest loss
after STEPS steps), so the comparison is fair rather than "default settings".

Run: python -m experiments.rosenbrock
"""

import matplotlib.pyplot as plt
import numpy as np

from fromscratch import LABELS, NAMES, build
from fromscratch.functions import rosenbrock

from .common import COLORS, INK, INK_2, MUTED, save, write_json

START = np.array([-1.2, 1.0])     # the classic starting point
MINIMUM = np.array([1.0, 1.0])
STEPS = 2000
LR_GRID = 10.0 ** np.arange(-5, 0.01, 0.25)


def run(name, lr, steps=STEPS):
    p = START.copy()
    opt = build(name, [p], lr=lr)
    path, losses = [p.copy()], [rosenbrock(p)[0]]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            _, g = rosenbrock(p)
            opt.step([g])
            f = rosenbrock(p)[0]
            if not np.isfinite(f) or np.abs(p).max() > 1e6:
                return np.array(path), np.array(losses), True
            path.append(p.copy())
            losses.append(f)
    return np.array(path), np.array(losses), False


def first_step_below(losses, threshold):
    hits = np.nonzero(losses < threshold)[0]
    return int(hits[0]) if hits.size else None


def main():
    results = {}
    for name in NAMES:
        best = None
        for lr in LR_GRID:
            path, losses, diverged = run(name, lr)
            if diverged:
                continue
            if best is None or losses[-1] < best[1][-1]:
                best = (path, losses, lr)
        path, losses, lr = best
        results[name] = {
            "best_lr": float(lr), "final_loss": float(losses[-1]),
            "distance_to_minimum": float(np.linalg.norm(path[-1] - MINIMUM)),
            "steps_to_loss_below_1e-3": first_step_below(losses, 1e-3),
            "path": path, "losses": losses,
        }
        print(f"{LABELS[name]:15s} lr={lr:.2e}  final f={losses[-1]:.2e}  "
              f"steps to f<1e-3: {results[name]['steps_to_loss_below_1e-3']}")

    # Small multiples: one panel per optimizer, plus the loss curves.
    xs, ys = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-1, 3, 400))
    zs = (1 - xs) ** 2 + 100 * (ys - xs ** 2) ** 2
    levels = np.logspace(-1, 3.5, 14)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    for ax, name in zip(axes.flat, NAMES):
        r = results[name]
        ax.contour(xs, ys, zs, levels=levels, colors=MUTED, linewidths=0.5, alpha=0.6)
        ax.plot(r["path"][:, 0], r["path"][:, 1], color=COLORS[name], lw=1.6)
        ax.plot(*START, "o", color=INK, ms=4)
        ax.plot(*MINIMUM, "*", color=INK, ms=9)
        ax.set_title(f"{LABELS[name]}  (lr {r['best_lr']:.1e})", fontsize=9.5)
        ax.text(0.03, 0.04, f"final loss {r['final_loss']:.1e}", transform=ax.transAxes,
                fontsize=8, color=INK_2)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
        ax.grid(False)
    ax = axes.flat[-1]
    for name in NAMES:
        ax.semilogy(results[name]["losses"], color=COLORS[name], label=LABELS[name])
    ax.set_title("Loss vs step (tuned lr each)")
    ax.set_xlabel("step")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.suptitle(f"Rosenbrock from (-1.2, 1): {STEPS} steps, learning rate tuned per optimizer",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "rosenbrock.png")

    write_json("rosenbrock.json", {
        name: {k: v for k, v in r.items() if k not in ("path", "losses")}
        for name, r in results.items()})


if __name__ == "__main__":
    main()
