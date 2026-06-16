from __future__ import annotations

from pathlib import Path
from typing import Any
from copy import deepcopy

import pandas as pd

from .config import artifact_dir, load_config, model_config, output_dir, raster_output_dir, report_dir, sample_columns
from .evaluate import evaluate_models, learning_curve_experiment, repeated_split_experiment
from .inference import predict_raster
from .models import load_model_artifacts, save_model_artifact
from .artifacts import make_run_id, write_dataframe, write_json
from .data import sample_summary
from .train import load_training_data, train_models


def validate_data(cfg: dict[str, Any]) -> dict[str, Any]:
    X_train, X_test, y_train, y_test, samples, feature_names, classes = load_training_data(cfg)
    cols = sample_columns(cfg)
    summary = sample_summary(samples, cols["label"], cols["split"])
    return {
        "features": feature_names,
        "classes": classes,
        "n_samples": int(len(samples)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "summary": summary,
    }


def train_and_evaluate(cfg: dict[str, Any], model_names: list[str] | None = None) -> dict[str, Any]:
    out_dir = output_dir(cfg)
    reports_dir = report_dir(cfg)
    artifacts_dir = artifact_dir(cfg)
    cols = sample_columns(cfg)
    run_id = cfg.get("run_id") or make_run_id()
    result = train_models(cfg, model_names=model_names)
    if not result["models"]:
        raise RuntimeError(f"No models were trained. Errors: {result['errors']}")

    metrics, reports = evaluate_models(
        result["models"],
        result["X_train"],
        result["y_train"],
        result["X_test"],
        result["y_test"],
        result["classes"],
        int(cols["flood_class"]),
    )

    metrics_path = write_dataframe(reports_dir / "model_metrics.csv", metrics)
    reports_path = write_json(reports_dir / "classification_reports.json", reports)
    if result["errors"]:
        write_json(reports_dir / "model_training_errors.json", result["errors"])

    artifacts = {}
    config_snapshot = deepcopy(cfg)
    metadata_base = {
        "run_id": run_id,
        "features": result["feature_names"],
        "feature_set": cfg.get("features", {}).get("active_set"),
        "classes": result["classes"],
        "label_column": cols["label"],
        "split_column": cols["split"],
        "train_value": cols["train_value"],
        "test_value": cols["test_value"],
        "flood_class": cols["flood_class"],
        "input_image": cfg.get("paths", {}).get("image"),
        "input_samples": cfg.get("paths", {}).get("samples"),
        "metrics_file": str(metrics_path),
        "classification_reports_file": str(reports_path),
    }
    for model_name, model_info in result["models"].items():
        metadata = {
            **metadata_base,
            "model_config": model_config(cfg, model_name),
        }
        artifacts[model_name] = save_model_artifact(
            model_name,
            model_info,
            artifacts_dir,
            metadata,
            run_id=run_id,
            config_snapshot=config_snapshot,
        )
    artifacts_path = write_json(artifacts_dir / "model_artifacts.json", artifacts)

    return {
        **result,
        "metrics": metrics,
        "reports": reports,
        "artifacts": artifacts,
        "artifacts_path": artifacts_path,
        "run_id": run_id,
        "output_dir": out_dir,
        "report_dir": reports_dir,
        "artifact_dir": artifacts_dir,
    }


def run_inference(cfg: dict[str, Any], trained_result: dict[str, Any]) -> dict[str, dict[str, str]]:
    cols = sample_columns(cfg)
    outputs = predict_raster(
        cfg,
        trained_result["models"],
        trained_result["feature_names"],
        trained_result["classes"],
        int(cols["flood_class"]),
        raster_output_dir(cfg),
    )
    write_json(raster_output_dir(cfg) / "prediction_outputs.json", outputs)
    return outputs


def infer_from_artifacts(
    cfg: dict[str, Any],
    artifact_index_path: str | Path,
    model_names: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    models, metadata = load_model_artifacts(artifact_index_path, model_names=model_names)
    first_metadata = next(iter(metadata.values()))
    feature_names = list(first_metadata["features"])
    classes = [int(value) for value in first_metadata["classes"]]
    flood_class = int(first_metadata["flood_class"])
    outputs = predict_raster(
        cfg,
        models,
        feature_names,
        classes,
        flood_class,
        raster_output_dir(cfg),
    )
    write_json(raster_output_dir(cfg) / "prediction_outputs.json", outputs)
    return outputs


def run_experiments(cfg: dict[str, Any]) -> dict[str, Path]:
    reports = report_dir(cfg)
    cols = sample_columns(cfg)
    X_train, X_test, y_train, y_test, samples, feature_names, classes = load_training_data(cfg)
    outputs: dict[str, Path] = {}
    exp_cfg = cfg.get("experiments", {})

    repeated_cfg = exp_cfg.get("repeated_splits", {})
    if repeated_cfg.get("enabled", False):
        repeated = repeated_split_experiment(
            pd.concat([X_train, X_test], axis=0),
            pd.concat([y_train, y_test], axis=0),
            cfg,
            repeated_cfg.get("models", cfg.get("models", {}).get("active", ["random_forest"])),
            classes,
            int(cols["flood_class"]),
            int(repeated_cfg.get("n_repeats", 30)),
            float(repeated_cfg.get("test_size", 0.30)),
            int(cfg.get("models", {}).get("random_state", 42)),
        )
        outputs["repeated_splits"] = write_dataframe(reports / "generalization_repeated_splits.csv", repeated)

    lc_cfg = exp_cfg.get("learning_curve", {})
    if lc_cfg.get("enabled", False):
        curve = learning_curve_experiment(
            X_train,
            y_train,
            X_test,
            y_test,
            cfg,
            lc_cfg.get("models", cfg.get("models", {}).get("active", ["random_forest"])),
            classes,
            int(cols["flood_class"]),
            list(lc_cfg.get("train_fracs", [0.1, 0.2, 0.4, 0.8, 1.0])),
            int(lc_cfg.get("n_repeats", 10)),
            int(cfg.get("models", {}).get("random_state", 42)),
        )
        outputs["learning_curve"] = write_dataframe(reports / "generalization_learning_curve.csv", curve)
    return outputs


def run_all(config_path: str | Path, run_experiment_suite: bool = False) -> dict[str, Any]:
    cfg = load_config(config_path)
    trained = train_and_evaluate(cfg)
    predictions = run_inference(cfg, trained)
    experiments = run_experiments(cfg) if run_experiment_suite else {}
    return {
        "metrics": str(Path(trained["report_dir"]) / "model_metrics.csv"),
        "artifacts": str(Path(trained["artifact_dir"]) / "model_artifacts.json"),
        "predictions": predictions,
        "experiments": {key: str(path) for key, path in experiments.items()},
    }


# Backward-compatible alias for older site-oriented scripts/notebooks.
validate_site = validate_data
