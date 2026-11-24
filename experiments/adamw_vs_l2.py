"""Experiment 5: weight decay inside Adam (L2) is not the same as AdamW's decoupled decay.

Setup that isolates the effect: four groups of 2,000 weights, all starting at 1.0. The loss
gradient of each weight is pure zero-mean noise, so the loss itself pulls the weights
nowhere; only weight decay moves them on average. The groups differ only in how large
their gradients are (noise std 0.01, 0.1, 1 and 10).

  Adam + L2:  the decay term wd * p is added to the gradient and then divided by
              sqrt(v_hat) ~ gradient size. Weights with big gradients barely decay;
              weights with tiny gradients are pulled to zero at a fixed speed of ~lr per step.
  AdamW:      p <- p * (1 - lr * wd) every step, for every weight alike.

We plot the MEAN weight of each group (the noise averages out over 2,000 weights).

Run: python -m experiments.adamw_vs_l2
"""

import matplotlib.pyplot as plt
import numpy as np

from fromscratch import Adam, AdamW

from .common import INK, MUTED, save, write_json

NOISE_STDS = [0.01, 0.1, 1.0, 10.0]
SIZE = 2000
STEPS = 3000
LR = 1e-2
WD = 0.1
SHADES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # one blue ramp: darker = noisier


def simulate(make_opt, seed=0):
    rng = np.random.default_rng(seed)
    params = [np.ones(SIZE) for _ in NOISE_STDS]
    opt = make_opt(params)
    means = np.empty((STEPS + 1, len(NOISE_STDS)))
    means[0] = 1.0
    for step in range(1, STEPS + 1):
        opt.step([rng.normal(0.0, s, SIZE) for s in NOISE_STDS])
        means[step] = [p.mean() for p in params]
    return means


def main():
    curves = {
        "Adam + L2 (weight_decay=0.1)": simulate(lambda ps: Adam(ps, lr=LR, weight_decay=WD)),
        "AdamW (weight_decay=0.1)": simulate(lambda ps: AdamW(ps, lr=LR, weight_decay=WD)),
    }
    steps = np.arange(STEPS + 1)
    expected = (1 - LR * WD) ** steps

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    summary = {}
    for ax, (title, means) in zip(axes, curves.items()):
        for j, (std, color) in enumerate(zip(NOISE_STDS, SHADES)):
            ax.plot(steps, means[:, j], color=color, label=f"gradient noise std {std:g}")
        ax.plot(steps, expected, "--", color=MUTED, lw=1.2, label=r"expected AdamW decay $(1-\mathrm{lr}\cdot\mathrm{wd})^t$")
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_xlim(0, STEPS)
        summary[title] = {f"noise_std={s:g}": float(means[-1, j]) for j, s in enumerate(NOISE_STDS)}
    axes[0].set_ylabel("mean weight in the group (started at 1)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Same weight decay, different meaning (lr {LR}, wd {WD}, loss gradient = pure noise)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "adamw_vs_l2.png")
    summary["expected_adamw_mean_at_end"] = float(expected[-1])
    write_json("adamw_vs_l2.json", summary)
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
