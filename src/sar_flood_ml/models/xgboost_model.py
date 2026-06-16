from __future__ import annotations

from typing import Any

import numpy as np

from .utils import label_maps


def train(X_train, y_train, classes: list[int], cfg: dict[str, Any], X_val=None, y_val=None) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("Install the 'xgboost' extra to train XGBoost models.") from exc

    label_to_index, _ = label_maps(classes)
    y_index = np.array([label_to_index[int(label)] for label in y_train], dtype="int32")
    objective = "binary:logistic" if len(classes) == 2 else "multi:softprob"
    eval_metric = "logloss" if len(classes) == 2 else "mlogloss"
    model = xgb.XGBClassifier(
        objective=objective,
        eval_metric=eval_metric,
        n_estimators=int(cfg.get("n_estimators", 300)),
        max_depth=int(cfg.get("max_depth", 4)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        subsample=float(cfg.get("subsample", 0.9)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
        min_child_weight=float(cfg.get("min_child_weight", 1)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        gamma=float(cfg.get("gamma", 0.0)),
        random_state=int(cfg.get("random_state", 42)),
        n_jobs=int(cfg.get("n_jobs", -1)),
    )
    model.fit(np.asarray(X_train, dtype="float32"), y_index)
    return {"kind": "indexed_labels", "model": model}


def predict(model_info: dict[str, Any], X, classes: list[int], flood_class: int) -> tuple[np.ndarray, np.ndarray]:
    label_to_index, index_to_label = label_maps(classes)
    model = model_info["model"]
    X_arr = np.asarray(X, dtype="float32")
    pred_index = model.predict(X_arr).astype("int32")
    labels = np.array([index_to_label[int(idx)] for idx in pred_index], dtype="int32")
    probabilities = model.predict_proba(X_arr)
    flood_col = label_to_index[flood_class]
    return labels, probabilities[:, flood_col].astype("float32")


__all__ = ["predict", "train"]
