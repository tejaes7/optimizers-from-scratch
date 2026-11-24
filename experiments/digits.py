"""Experiment 3: training a real network: an MLP (64-128-64-10) on 8x8 handwritten digits.

For each optimizer, sweep 9 learning rates x 5 seeds (30 epochs, batch 64). The learning rate
is chosen by mean VALIDATION accuracy; test accuracy is reported only for that choice.
Also reported: how many of the 9 learning rates land within 1 point of that optimizer's best
validation accuracy, as a simple measure of how forgiving it is to tune.

Run: python -m experiments.digits      (about a minute)
"""

import matplotlib.pyplot as plt
import numpy as np

from fromscratch import LABELS, NAMES
from fromscratch.data import load

from .common import COLORS, INK, INK_2, save, train_mlp, write_json

LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 30


def main():
    data = load(seed=0)
    runs = {}
    for name in NAMES:
        for lr in LRS:
            runs[(name, lr)] = [train_mlp(name, lr, data, seed=s, epochs=EPOCHS) for s in SEEDS]
        print(f"done {LABELS[name]}")

    summary = {}
    for name in NAMES:
        val = {lr: np.mean([r["final_val_acc"] for r in runs[(name, lr)]]) for lr in LRS}
        best_lr = max(LRS, key=lambda lr: (val[lr], -lr))
        best_runs = runs[(name, best_lr)]
        test = np.array([r["final_test_acc"] for r in best_runs])
        summary[name] = {
            "best_lr": best_lr,
            "val_acc_mean": float(val[best_lr]),
            "test_acc_mean": float(test.mean()),
            "test_acc_std": float(test.std(ddof=1)),
            "final_train_loss_mean": float(np.mean([r["final_train_loss"] for r in best_runs])),
            "lrs_within_1pt_of_best": int(sum(val[lr] >= val[best_lr] - 0.01 for lr in LRS)),
            "diverged_runs": {str(lr): int(sum(r["diverged"] for r in runs[(name, lr)]))
                              for lr in LRS},
            "val_acc_by_lr": {str(lr): float(val[lr]) for lr in LRS},
        }
        s = summary[name]
        print(f"{LABELS[name]:15s} best lr {best_lr:<7g} test {100 * s['test_acc_mean']:.1f} "
              f"+/- {100 * s['test_acc_std']:.1f}%  robust lrs {s['lrs_within_1pt_of_best']}/9")
    write_json("digits.json", {"lrs": LRS, "seeds": SEEDS, "epochs": EPOCHS, "results": summary})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for name in NAMES:
        accs = np.array([[r["final_val_acc"] for r in runs[(name, lr)]] for lr in LRS]) * 100
        ax1.plot(LRS, accs.mean(axis=1), "-o", ms=4, color=COLORS[name], label=LABELS[name])
    ax1.set_xscale("log")
    ax1.set_xlabel("learning rate")
    ax1.set_ylabel("validation accuracy (%), mean of 5 seeds")
    ax1.set_ylim(0, 100)
    ax1.set_title("Learning-rate sensitivity")
    ax1.legend(loc="lower center", ncol=2, fontsize=8)

    for name in NAMES:
        curves = np.array([r["epoch_loss"] for r in runs[(name, summary[name]["best_lr"])]])
        epochs = np.arange(1, curves.shape[1] + 1)
        ax2.plot(epochs, curves.mean(axis=0), color=COLORS[name],
                 label=f"{LABELS[name]} (lr {summary[name]['best_lr']:g})")
        ax2.fill_between(epochs, curves.min(axis=0), curves.max(axis=0), color=COLORS[name],
                         alpha=0.12, lw=0)
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("training loss (band = min-max over seeds)")
    ax2.set_title("Training loss at each optimizer's best learning rate")
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle("MLP on handwritten digits: 9 learning rates x 5 seeds per optimizer",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "digits.png")


if __name__ == "__main__":
    main()
