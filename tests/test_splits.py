from __future__ import annotations

import numpy as np

from sar_flood_ml.splits import stratified_subsample_indices


def test_stratified_subsample_keeps_each_class():
    y = np.array([1, 1, 1, 2, 2, 2])
    idx = stratified_subsample_indices(y, 0.34, random_state=1)
    assert set(y[idx]) == {1, 2}
