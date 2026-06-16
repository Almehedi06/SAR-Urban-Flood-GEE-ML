from __future__ import annotations

from typing import Any

from .config import active_feature_names, active_model_names, model_config, sample_columns
from .models import train_model
from .data import load_samples, train_test_from_split, validate_samples


def load_training_data(cfg: dict[str, Any], feature_set: str | None = None):
    feature_names = active_feature_names(cfg, feature_set)
    cols = sample_columns(cfg)
    samples = load_samples(cfg["paths"]["samples"])
    samples = validate_samples(samples, feature_names, cols["label"], cols["split"])
    X_train, X_test, y_train, y_test, samples = train_test_from_split(
        samples,
        feature_names,
        cols["label"],
        cols["split"],
        cols["train_value"],
        cols["test_value"],
    )
    classes = sorted(samples[cols["label"]].astype(int).unique().tolist())
    return X_train, X_test, y_train, y_test, samples, feature_names, classes


def train_models(
    cfg: dict[str, Any],
    feature_set: str | None = None,
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test, samples, feature_names, classes = load_training_data(cfg, feature_set)
    trained = {}
    errors = {}
    for model_name in active_model_names(cfg, model_names):
        try:
            trained[model_name] = train_model(
                model_name,
                X_train,
                y_train,
                classes,
                model_config(cfg, model_name),
                X_val=X_test,
                y_val=y_test,
            )
        except ImportError as exc:
            errors[model_name] = str(exc)

    return {
        "models": trained,
        "errors": errors,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "samples": samples,
        "feature_names": feature_names,
        "classes": classes,
    }
