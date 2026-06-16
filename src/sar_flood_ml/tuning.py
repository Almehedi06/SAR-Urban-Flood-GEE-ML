from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .artifacts import make_run_id, write_dataframe, write_json, write_yaml
from .config import artifact_dir, model_config, report_dir, sample_columns
from .evaluate import evaluate_models
from .models import save_model_artifact
from .train import load_training_data


RF_PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "n_estimators": [50, 80, 100, 150, 200, 300, 500, 800],
    "max_depth": [None, 5, 10, 20, 40, 80],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced", "balanced_subsample"],
}


def recommended_search_jobs(requested_jobs: int | None = None) -> int:
    """Pick a sensible search-level parallelism default for a local workstation."""
    if requested_jobs is not None:
        return int(requested_jobs)
    cores = os.cpu_count() or 1
    if cores <= 2:
        return 1
    return max(1, min(24, cores - 2))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _rf_base_params(cfg: dict[str, Any]) -> dict[str, Any]:
    rf_cfg = model_config(cfg, "random_forest")
    return {
        "n_estimators": int(rf_cfg.get("n_estimators", 500)),
        "max_depth": rf_cfg.get("max_depth"),
        "min_samples_split": int(rf_cfg.get("min_samples_split", 2)),
        "min_samples_leaf": int(rf_cfg.get("min_samples_leaf", 2)),
        "max_features": rf_cfg.get("max_features", "sqrt"),
        "class_weight": rf_cfg.get("class_weight", "balanced"),
        "random_state": int(rf_cfg.get("random_state", cfg.get("models", {}).get("random_state", 42))),
    }


def tune_random_forest(
    cfg: dict[str, Any],
    *,
    feature_set: str | None = None,
    n_iter: int = 80,
    cv_splits: int = 3,
    n_jobs: int | None = None,
    param_distributions: dict[str, list[Any]] | None = None,
    save_model: bool = True,
) -> dict[str, Any]:
    """Tune Random Forest hyperparameters on the configured training split."""
    cols = sample_columns(cfg)
    flood_class = int(cols["flood_class"])
    random_state = int(cfg.get("models", {}).get("random_state", 42))
    search_jobs = recommended_search_jobs(n_jobs)

    X_train, X_test, y_train, y_test, _samples, feature_names, classes = load_training_data(cfg, feature_set)
    X_train_np = np.asarray(X_train, dtype="float32")
    X_test_np = np.asarray(X_test, dtype="float32")
    y_train_np = np.asarray(y_train, dtype="int32")
    y_test_np = np.asarray(y_test, dtype="int32")

    min_class_count = int(pd.Series(y_train_np).value_counts().min())
    if cv_splits > min_class_count:
        raise ValueError(
            f"cv_splits={cv_splits} is too high for the training data. "
            f"The smallest class has {min_class_count} samples."
        )

    cv = StratifiedKFold(n_splits=int(cv_splits), shuffle=True, random_state=random_state)
    flood_f1 = make_scorer(f1_score, pos_label=flood_class, zero_division=0)
    estimator = RandomForestClassifier(**_rf_base_params(cfg), n_jobs=1)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions or RF_PARAM_DISTRIBUTIONS,
        n_iter=int(n_iter),
        scoring=flood_f1,
        cv=cv,
        n_jobs=search_jobs,
        random_state=random_state,
        verbose=1,
        return_train_score=True,
        refit=True,
    )
    search.fit(X_train_np, y_train_np)

    best_model = search.best_estimator_
    best_model.set_params(n_jobs=int(model_config(cfg, "random_forest").get("n_jobs", -1)))
    model_info = {"kind": "sklearn_labels", "model": best_model}
    metrics, reports = evaluate_models(
        {"random_forest": model_info},
        X_train_np,
        y_train_np,
        X_test_np,
        y_test_np,
        classes,
        flood_class,
    )

    run_id = cfg.get("run_id") or make_run_id()
    tuning_reports_dir = report_dir(cfg) / "tuning"
    tuning_artifacts_dir = artifact_dir(cfg) / "tuning"
    tuning_reports_dir.mkdir(parents=True, exist_ok=True)
    tuning_artifacts_dir.mkdir(parents=True, exist_ok=True)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results_path = write_dataframe(tuning_reports_dir / "random_forest_random_search_cv_results.csv", cv_results)
    metrics_path = write_dataframe(tuning_reports_dir / "random_forest_tuned_holdout_metrics.csv", metrics)
    reports_path = write_json(tuning_reports_dir / "random_forest_tuned_classification_report.json", reports)

    tuned_params = _json_ready(search.best_params_)
    tuned_model_config = {**model_config(cfg, "random_forest"), **tuned_params}
    tuned_model_config["n_jobs"] = int(model_config(cfg, "random_forest").get("n_jobs", -1))
    best_params_path = write_yaml(tuning_reports_dir / "random_forest_best_params.yaml", tuned_model_config)

    summary = {
        "model": "random_forest",
        "search_type": "randomized_search_cv",
        "run_id": run_id,
        "n_iter": int(n_iter),
        "cv_splits": int(cv_splits),
        "search_n_jobs": search_jobs,
        "estimator_n_jobs_during_search": 1,
        "scoring": "flood_f1",
        "best_cv_flood_f1": float(search.best_score_),
        "best_params": tuned_params,
        "features": feature_names,
        "classes": classes,
        "flood_class": flood_class,
        "train_samples": int(len(X_train_np)),
        "test_samples": int(len(X_test_np)),
        "cv_results_file": str(cv_results_path),
        "best_params_file": str(best_params_path),
        "holdout_metrics_file": str(metrics_path),
        "classification_report_file": str(reports_path),
    }

    artifact_index_path = None
    if save_model:
        config_snapshot = deepcopy(cfg)
        config_snapshot.setdefault("models", {})["random_forest"] = tuned_model_config
        metadata = {
            "run_id": run_id,
            "features": feature_names,
            "feature_set": cfg.get("features", {}).get("active_set"),
            "classes": classes,
            "label_column": cols["label"],
            "split_column": cols["split"],
            "train_value": cols["train_value"],
            "test_value": cols["test_value"],
            "flood_class": flood_class,
            "input_image": cfg.get("paths", {}).get("image"),
            "input_samples": cfg.get("paths", {}).get("samples"),
            "model_config": tuned_model_config,
            "tuning": summary,
        }
        artifact = save_model_artifact(
            "random_forest",
            model_info,
            tuning_artifacts_dir,
            metadata,
            run_id=run_id,
            config_snapshot=config_snapshot,
        )
        artifact_index_path = write_json(tuning_artifacts_dir / "random_forest_tuned_artifacts.json", {"random_forest": artifact})
        summary["artifact_index_file"] = str(artifact_index_path)
        summary["artifact"] = artifact

    summary_path = write_json(tuning_reports_dir / "random_forest_tuning_summary.json", _json_ready(summary))
    return {
        "summary": summary,
        "summary_path": summary_path,
        "cv_results_path": cv_results_path,
        "best_params_path": best_params_path,
        "metrics_path": metrics_path,
        "reports_path": reports_path,
        "artifact_index_path": artifact_index_path,
        "metrics": metrics,
        "best_estimator": best_model,
    }


__all__ = ["RF_PARAM_DISTRIBUTIONS", "recommended_search_jobs", "tune_random_forest"]
