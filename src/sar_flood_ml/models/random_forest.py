from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train(X_train, y_train, classes: list[int], cfg: dict[str, Any], X_val=None, y_val=None) -> dict[str, Any]:
    model = RandomForestClassifier(
        n_estimators=int(cfg.get("n_estimators", 500)),
        max_depth=cfg.get("max_depth"),
        min_samples_split=int(cfg.get("min_samples_split", 2)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
        max_features=cfg.get("max_features", "sqrt"),
        class_weight=cfg.get("class_weight", "balanced"),
        random_state=int(cfg.get("random_state", 42)),
        n_jobs=int(cfg.get("n_jobs", -1)),
    )
    model.fit(np.asarray(X_train, dtype="float32"), np.asarray(y_train, dtype="int32"))
    return {"kind": "sklearn_labels", "model": model}


def predict(model_info: dict[str, Any], X, classes: list[int], flood_class: int) -> tuple[np.ndarray, np.ndarray]:
    model = model_info["model"]
    X_arr = np.asarray(X, dtype="float32")
    labels = model.predict(X_arr).astype("int32")
    probabilities = model.predict_proba(X_arr)
    flood_col = list(model.classes_).index(flood_class)
    return labels, probabilities[:, flood_col].astype("float32")


__all__ = ["predict", "train"]
