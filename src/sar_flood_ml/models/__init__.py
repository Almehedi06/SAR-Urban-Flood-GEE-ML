from __future__ import annotations

from typing import Any

import numpy as np

from . import cnn_1d, random_forest, xgboost_model
from .utils import label_maps, load_model_artifact, load_model_artifacts, save_model_artifact


MODEL_REGISTRY = {
    "random_forest": random_forest,
    "xgboost": xgboost_model,
    "cnn_1d": cnn_1d,
}

MODEL_KIND_REGISTRY = {
    "sklearn_labels": random_forest,
    "indexed_labels": xgboost_model,
    "cnn_1d": cnn_1d,
}

TABULAR_MODELS = set(MODEL_REGISTRY)


def train_model(
    name: str,
    X_train,
    y_train,
    classes: list[int],
    cfg: dict[str, Any],
    X_val=None,
    y_val=None,
) -> dict[str, Any]:
    try:
        model_module = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {name}") from exc
    return model_module.train(X_train, y_train, classes, cfg, X_val=X_val, y_val=y_val)


def predict_labels_and_flood_probability(
    model_info: dict[str, Any],
    X,
    classes: list[int],
    flood_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    kind = model_info["kind"]
    try:
        model_module = MODEL_KIND_REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown model kind: {kind}") from exc
    return model_module.predict(model_info, X, classes, flood_class)


__all__ = [
    "MODEL_KIND_REGISTRY",
    "MODEL_REGISTRY",
    "TABULAR_MODELS",
    "label_maps",
    "load_model_artifact",
    "load_model_artifacts",
    "predict_labels_and_flood_probability",
    "save_model_artifact",
    "train_model",
]
