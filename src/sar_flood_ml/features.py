from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype="float32"), where=b != 0)


def resolve_feature(columns: Mapping[str, Any], name: str) -> np.ndarray:
    """Resolve a raw or derived feature from a column/array mapping."""
    if name in columns:
        return np.asarray(columns[name])

    operators = {
        "_div_": _safe_divide,
        "_minus_": lambda a, b: a - b,
        "_plus_": lambda a, b: a + b,
        "_times_": lambda a, b: a * b,
    }
    for token, op in operators.items():
        if token in name:
            left, right = name.split(token, 1)
            if left not in columns or right not in columns:
                raise KeyError(f"Cannot derive {name!r}; missing {left!r} or {right!r}.")
            return op(np.asarray(columns[left], dtype="float32"), np.asarray(columns[right], dtype="float32"))

    raise KeyError(f"Feature {name!r} is not present and is not a supported derived feature.")


def build_feature_frame(samples: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame containing raw and derived feature columns."""
    data = samples.copy()
    for feature in feature_names:
        if feature not in data.columns:
            data[feature] = resolve_feature(data, feature)
    return data[feature_names]


def build_feature_matrix(columns: Mapping[str, Any], feature_names: list[str]) -> np.ndarray:
    """Build a numeric matrix from named arrays or Series."""
    resolved = [np.asarray(resolve_feature(columns, name), dtype="float32").reshape(-1) for name in feature_names]
    return np.column_stack(resolved).astype("float32")
