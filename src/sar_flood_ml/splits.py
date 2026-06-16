from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def repeated_stratified_splits(
    y: np.ndarray,
    n_repeats: int,
    test_size: float,
    random_state: int,
):
    splitter = StratifiedShuffleSplit(
        n_splits=n_repeats,
        test_size=test_size,
        random_state=random_state,
    )
    dummy = np.zeros(len(y))
    yield from splitter.split(dummy, y)


def stratified_subsample_indices(
    y: np.ndarray,
    fraction: float,
    random_state: int,
) -> np.ndarray:
    """Return a class-balanced subset of row indices."""
    rng = np.random.default_rng(random_state)
    selected = []
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        n = max(1, int(round(len(label_indices) * fraction)))
        selected.append(rng.choice(label_indices, size=min(n, len(label_indices)), replace=False))
    return np.sort(np.concatenate(selected))
