from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from .config import model_config
from .models import predict_labels_and_flood_probability, train_model
from .splits import repeated_stratified_splits, stratified_subsample_indices


def evaluate_models(
    models: dict[str, dict[str, Any]],
    X_train,
    y_train,
    X_test,
    y_test,
    classes: list[int],
    flood_class: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    reports = {}
    for model_name, model_info in models.items():
        train_pred, _ = predict_labels_and_flood_probability(model_info, X_train, classes, flood_class)
        test_pred, _ = predict_labels_and_flood_probability(model_info, X_test, classes, flood_class)
        report = classification_report(y_test, test_pred, labels=classes, output_dict=True, zero_division=0)
        reports[model_name] = report
        train_report = classification_report(y_train, train_pred, labels=classes, output_dict=True, zero_division=0)
        flood_report = report[str(flood_class)]
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, test_pred),
                "macro_f1": report["macro avg"]["f1-score"],
                "flood_precision": flood_report["precision"],
                "flood_recall": flood_report["recall"],
                "flood_f1": flood_report["f1-score"],
                "train_macro_f1": train_report["macro avg"]["f1-score"],
                "train_test_macro_f1_gap": train_report["macro avg"]["f1-score"] - report["macro avg"]["f1-score"],
            }
        )
    metrics = pd.DataFrame(rows).sort_values(["flood_f1", "macro_f1"], ascending=False)
    return metrics, reports


def repeated_split_experiment(
    X,
    y,
    cfg: dict[str, Any],
    model_names: list[str],
    classes: list[int],
    flood_class: int,
    n_repeats: int,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    X_np = np.asarray(X, dtype="float32")
    y_np = np.asarray(y, dtype="int32")
    for repeat, (train_idx, test_idx) in enumerate(
        repeated_stratified_splits(y_np, n_repeats=n_repeats, test_size=test_size, random_state=random_state),
        start=1,
    ):
        for model_name in model_names:
            try:
                model_info = train_model(
                    model_name,
                    X_np[train_idx],
                    y_np[train_idx],
                    classes,
                    model_config(cfg, model_name),
                    X_val=X_np[test_idx],
                    y_val=y_np[test_idx],
                )
            except ImportError as exc:
                rows.append({"repeat": repeat, "model": model_name, "error": str(exc)})
                continue
            metrics, _ = evaluate_models(
                {model_name: model_info},
                X_np[train_idx],
                y_np[train_idx],
                X_np[test_idx],
                y_np[test_idx],
                classes,
                flood_class,
            )
            row = metrics.iloc[0].to_dict()
            row.update({"repeat": repeat, "train_size": len(train_idx), "test_size": len(test_idx), "error": ""})
            rows.append(row)
    return pd.DataFrame(rows)


def learning_curve_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    cfg: dict[str, Any],
    model_names: list[str],
    classes: list[int],
    flood_class: int,
    train_fracs: list[float],
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    X_train_np = np.asarray(X_train, dtype="float32")
    y_train_np = np.asarray(y_train, dtype="int32")
    X_test_np = np.asarray(X_test, dtype="float32")
    y_test_np = np.asarray(y_test, dtype="int32")
    for frac in train_fracs:
        for repeat in range(1, n_repeats + 1):
            subset_idx = stratified_subsample_indices(y_train_np, frac, random_state + repeat)
            for model_name in model_names:
                try:
                    model_info = train_model(
                        model_name,
                        X_train_np[subset_idx],
                        y_train_np[subset_idx],
                        classes,
                        model_config(cfg, model_name),
                        X_val=X_test_np,
                        y_val=y_test_np,
                    )
                except ImportError as exc:
                    rows.append({"train_fraction": frac, "repeat": repeat, "model": model_name, "error": str(exc)})
                    continue
                metrics, _ = evaluate_models(
                    {model_name: model_info},
                    X_train_np[subset_idx],
                    y_train_np[subset_idx],
                    X_test_np,
                    y_test_np,
                    classes,
                    flood_class,
                )
                row = metrics.iloc[0].to_dict()
                row.update({"train_fraction": frac, "repeat": repeat, "train_size": len(subset_idx), "error": ""})
                rows.append(row)
    return pd.DataFrame(rows)
