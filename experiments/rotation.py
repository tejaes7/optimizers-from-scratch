"""Experiment 2: adaptive optimizers are not rotation-invariant.

Same ill-conditioned bowl (condition number 100) in two orientations:
  - aligned with the axes (0 degrees): the steep and flat directions are x and y themselves
  - rotated by 45 degrees: each axis mixes the steep and flat directions

The start point is rotated with the bowl, so the two problems are identical up to a rotation.
SGD, momentum and Nesterov only use the gradient vector, which rotates with the problem, so
their loss curves are exactly the same in both orientations. AdaGrad, RMSProp and Adam
rescale each coordinate separately; that acts like a diagonal preconditioner, which can only
fix ill-conditioning that lines up with the axes.

Run: python -m experiments.rotation
"""

import matplotlib.pyplot as plt
import numpy as np

from fromscratch import LABELS, NAMES, build
from fromscratch.functions import make_quadratic

from .common import COLORS, INK, INK_2, MUTED, save, write_json

KAPPA = 100.0
ANGLES = (0.0, 45.0)
STEPS = 1000
THRESHOLD = 1e-8
LR_GRID = 10.0 ** np.arange(-4, 0.51, 0.125)
START_ALIGNED = np.array([0.93, 0.61])   # asymmetric, so no lucky one-step landing


def run(name, lr, f, start, steps=STEPS, **opt_kwargs):
    p = start.copy()
    opt = build(name, [p], lr=lr, **opt_kwargs)
    losses, path = [f(p)[0]], [p.copy()]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            opt.step([f(p)[1]])
            loss = f(p)[0]
            if not np.isfinite(loss) or loss > 1e8:
                return None, None
            losses.append(loss)
            path.append(p.copy())
    return np.array(losses), np.array(path)


def steps_to(losses, threshold=THRESHOLD):
    """First step after which the loss STAYS below the threshold (None if it never settles)."""
    above = np.nonzero(losses >= threshold)[0]
    if above.size and above[-1] == len(losses) - 1:
        return None
    return int(above[-1] + 1) if above.size else 0


def score(losses):
    """Lower is better: steps to reach the threshold, else STEPS + how far off we ended."""
    s = steps_to(losses)
    return s if s is not None else STEPS + np.log10(losses[-1] / THRESHOLD)


def main():
    results = {}
    for angle in ANGLES:
        f = make_quadratic(KAPPA, angle)
        start = f.R @ START_ALIGNED
        for name in NAMES:
            # AdamW's default weight decay would move the minimum away from 0; turn it off here,
            # which makes AdamW identical to Adam on this problem.
            kw = {"weight_decay": 0.0} if name == "adamw" else {}
            best = None
            for lr in LR_GRID:
                losses, path = run(name, lr, f, start, **kw)
                if losses is None:
                    continue
                if best is None or score(losses) < score(best[0]):
                    best = (losses, path, lr)
            losses, path, lr = best
            results[(name, angle)] = {"lr": float(lr), "steps": steps_to(losses),
                                      "final_loss": float(losses[-1]), "losses": losses,
                                      "path": path}
            print(f"{angle:4.0f} deg  {LABELS[name]:15s} lr={lr:.2e}  "
                  f"steps to f<{THRESHOLD:g}: {steps_to(losses)}  final {losses[-1]:.1e}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    shown = ["sgd", "momentum", "rmsprop", "adam"]
    for ax, angle in zip(axes[:2], ANGLES):
        f = make_quadratic(KAPPA, angle)
        g = np.linspace(-1.6, 1.6, 300)
        xs, ys = np.meshgrid(g, g)
        pts = np.stack([xs.ravel(), ys.ravel()])
        zs = 0.5 * np.einsum("in,ij,jn->n", pts, f.H, pts).reshape(xs.shape)
        ax.contour(xs, ys, zs, levels=np.logspace(-2, 2.2, 12), colors=MUTED, linewidths=0.5,
                   alpha=0.6)
        for name in shown:
            path = results[(name, angle)]["path"][:150]
            ax.plot(path[:, 0], path[:, 1], color=COLORS[name], lw=1.4, label=LABELS[name])
        ax.plot(*(f.R @ START_ALIGNED), "o", color=INK, ms=4)
        ax.plot(0, 0, "*", color=INK, ms=9)
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.grid(False)
        ax.set_title(f"Bowl rotated {angle:.0f}°  (first 150 steps)")
    axes[0].legend(loc="lower left", fontsize=7.5)

    ax = axes[2]
    x = np.arange(len(NAMES))
    width = 0.38
    for i, (angle, shade) in enumerate(zip(ANGLES, (1.0, 0.45))):
        vals = [results[(n, angle)]["steps"] or STEPS for n in NAMES]
        bars = ax.bar(x + (i - 0.5) * width, vals, width - 0.03,
                      color=[COLORS[n] for n in NAMES], alpha=shade,
                      label=f"{angle:.0f}° (aligned)" if angle == 0 else f"{angle:.0f}° (rotated)")
        for xi, n, v in zip(x, NAMES, vals):
            txt = str(v) if results[(n, angle)]["steps"] is not None else f">{STEPS}"
            ax.text(xi + (i - 0.5) * width, v + 12, txt, ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(x, [LABELS[n] for n in NAMES], rotation=25, ha="right")
    ax.set_ylabel(f"steps until loss stays < {THRESHOLD:g}")
    ax.set_title("Steps needed, lr tuned separately for each case")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=INK_2, alpha=1.0, label="0° (aligned): darker bar"),
                       Patch(color=INK_2, alpha=0.45, label="45° (rotated): lighter bar")],
              loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.suptitle(f"Same bowl (condition number {KAPPA:.0f}), two orientations", fontweight="bold",
                 color=INK)
    fig.tight_layout()
    save(fig, "rotation.png")

    write_json("rotation.json", {
        f"{name}@{angle:.0f}deg": {k: v for k, v in r.items() if k not in ("losses", "path")}
        for (name, angle), r in results.items()})

    # Check the claim in the docstring: identical curves for the rotation-invariant methods.
    for name in ("sgd", "momentum", "nesterov"):
        a, b = results[(name, 0.0)], results[(name, 45.0)]
        if a["lr"] == b["lr"]:
            diff = np.max(np.abs(a["losses"] - b["losses"]) / np.maximum(a["losses"], 1e-300))
            print(f"{LABELS[name]}: max relative loss difference between orientations {diff:.1e}")


if __name__ == "__main__":
    main()
