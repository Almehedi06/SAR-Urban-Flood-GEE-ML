from __future__ import annotations

import numpy as np
import pandas as pd

from sar_flood_ml.features import build_feature_frame, build_feature_matrix


def test_derived_ratio_feature_frame():
    df = pd.DataFrame({"VV": [2.0, 4.0], "VH": [1.0, 2.0]})
    out = build_feature_frame(df, ["VV", "VV_div_VH"])
    assert list(out.columns) == ["VV", "VV_div_VH"]
    assert np.allclose(out["VV_div_VH"], [2.0, 2.0])


def test_build_feature_matrix_from_arrays():
    matrix = build_feature_matrix({"VV": np.array([2.0]), "VH": np.array([4.0])}, ["VV", "VH_minus_VV"])
    assert matrix.shape == (1, 2)
    assert matrix[0, 1] == 2.0
