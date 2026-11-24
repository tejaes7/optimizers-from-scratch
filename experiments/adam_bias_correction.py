"""Experiment 4: why Adam needs bias correction, and why the bug is overshoot, not undershoot.

m and v start at 0. With a constant gradient g, after t steps:
    m_t = (1 - b1^t) g        v_t = (1 - b2^t) g^2
so WITHOUT correction the step is
    lr * m_t / sqrt(v_t) = lr * (1 - b1^t) / sqrt(1 - b2^t)
With b1 = 0.9, b2 = 0.999 that is 3.2x lr at step 1 and peaks at 6.6x lr at step 12:
v is "more biased" than m because b2 is closer to 1. The corrected step is exactly lr.

Part A plots this ratio measured from the optimizer (dots) against the formula (lines).
Part B trains the digits MLP with and without correction at several learning rates.

Run: python -m experiments.adam_bias_correction
"""

import matplotlib.pyplot as plt
import numpy as np

from fromscratch import Adam
from fromscratch.data import load

from .common import COLORS, INK, INK_2, save, train_mlp, write_json

B1 = 0.9
B2_VALUES = [0.99, 0.999, 0.9999]
STEPS = 3000
LRS = [1e-3, 3e-3, 1e-2, 3e-2]
SEEDS = [0, 1, 2, 3, 4]
SHADES = ["#2a78d6", "#eb6834", "#1baf7a"]


def measured_ratio(b2, bias_correction, steps=STEPS, lr=1e-3):
    p = np.zeros(1)
    opt = Adam([p], lr=lr, betas=(B1, b2), eps=0.0, bias_correction=bias_correction)
    ratios = []
    for _ in range(steps):
        before = p[0]
        opt.step([np.ones(1)])
        ratios.append((before - p[0]) / lr)
    return np.array(ratios)


def main():
    t = np.arange(1, STEPS + 1)
    part_a = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for b2, color in zip(B2_VALUES, SHADES):
        formula = (1 - B1 ** t) / np.sqrt(1 - b2 ** t)
        measured = measured_ratio(b2, bias_correction=False)
        assert np.allclose(measured, formula), "optimizer disagrees with the closed form"
        peak = int(np.argmax(formula))
        part_a[str(b2)] = {"step_1_ratio": float(formula[0]), "peak_ratio": float(formula[peak]),
                           "peak_step": peak + 1}
        ax1.plot(t, formula, color=color, label=f"no correction, b2 = {b2}")
        idx = np.unique(np.logspace(0, np.log10(STEPS), 14).astype(int)) - 1   # even on a log axis
        ax1.plot(t[idx], measured[idx], "o", ms=4, color=color)
        ax1.annotate(f"peak {formula[peak]:.1f}x at step {peak + 1}", (t[peak], formula[peak]),
                     xytext=(8, 4), textcoords="offset points", fontsize=8, color=INK_2)
    corrected = measured_ratio(0.999, bias_correction=True)
    assert np.allclose(corrected, 1.0)
    ax1.plot(t, corrected, color=COLORS["adam"], lw=2.2, label="with bias correction (any b2)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("step t (constant gradient)")
    ax1.set_ylabel("step size / learning rate")
    ax1.set_title("Without correction, early steps are too LARGE (dots: measured, lines: formula)")
    ax1.legend(loc="upper right", fontsize=8)
    print("part A:", part_a)

    # Part B: does it matter when training a real network?
    data = load(seed=0)
    part_b = {}
    for lr in LRS:
        for bc in (True, False):
            runs = [train_mlp("adam", lr, data, seed=s, epochs=30, bias_correction=bc)
                    for s in SEEDS]
            part_b[f"lr={lr:g}, correction={bc}"] = {
                "test_acc_mean": float(np.mean([r["final_test_acc"] for r in runs])),
                "diverged": int(sum(r["diverged"] for r in runs)),
                "max_loss_first_100_steps_mean": float(np.mean(
                    [max(r["step_loss"][:100]) if r["step_loss"] else np.inf for r in runs])),
            }
            print(lr, bc, part_b[f"lr={lr:g}, correction={bc}"])

    width = 0.38
    x = np.arange(len(LRS))
    for i, (bc, color, label) in enumerate([(True, COLORS["adam"], "with correction"),
                                            (False, "#eb6834", "without correction")]):
        vals = [100 * part_b[f"lr={lr:g}, correction={bc}"]["test_acc_mean"] for lr in LRS]
        ax2.bar(x + (i - 0.5) * width, vals, width - 0.03, color=color, label=label)
        for xi, v in zip(x, vals):
            ax2.text(xi + (i - 0.5) * width, v + 1, f"{v:.1f}", ha="center", fontsize=8,
                     color=INK_2)
    ax2.set_xticks(x, [f"lr {lr:g}" for lr in LRS])
    ax2.set_ylabel("test accuracy (%), mean of 5 seeds")
    ax2.set_ylim(0, 118)
    ax2.set_title("Digits MLP, 30 epochs")
    ax2.legend(loc="upper left", ncol=2, fontsize=8)
    ax2.grid(axis="x", visible=False)
    fig.suptitle("Adam bias correction (b1 = 0.9)", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "adam_bias_correction.png")
    write_json("adam_bias_correction.json", {"part_a": part_a, "part_b": part_b})


if __name__ == "__main__":
    main()
