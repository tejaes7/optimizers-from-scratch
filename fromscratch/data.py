"""The handwritten-digits dataset that ships with scikit-learn (no download needed).

1,797 images of 8x8 pixels, 10 classes. Split 60/20/20 into train / validation / test:
learning rates are chosen on validation, and only the final numbers use test.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load(seed=0):
    X, y = load_digits(return_X_y=True)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)
    # Standardise with statistics from the training split only.
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    norm = lambda a: (a - mean) / std
    return (norm(X_train), y_train), (norm(X_val), y_val), (norm(X_test), y_test)


def batches(X, y, batch_size, rng):
    order = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]
