from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import joblib

from ..artifacts import write_json, write_yaml


def label_maps(classes: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    return label_to_index, index_to_label


def save_model_artifact(
    model_name: str,
    model_info: dict[str, Any],
    out_dir: str | Path,
    metadata: dict[str, Any],
    run_id: str,
    config_snapshot: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = model_info["kind"]
    stem = f"{model_name}_{run_id}"
    metadata_path = out_dir / f"{stem}_metadata.json"
    config_path = out_dir / f"{stem}_config.yaml"

    if kind == "cnn_1d":
        model_path = out_dir / f"{stem}.keras"
        scaler_path = out_dir / f"{stem}_scaler.joblib"
        model_info["model"].save(model_path)
        joblib.dump(model_info["scaler"], scaler_path)
        artifact = {
            "kind": kind,
            "run_id": run_id,
            "model_file": str(model_path),
            "scaler_file": str(scaler_path),
        }
        history = model_info.get("history")
        if history is not None:
            history_path = out_dir / f"{stem}_training_history.csv"
            history.to_csv(history_path, index_label="epoch")
            artifact["history_csv"] = str(history_path)
    else:
        model_path = out_dir / f"{stem}.joblib"
        bundle = {
            "model_name": model_name,
            "kind": kind,
            "model": model_info["model"],
            **metadata,
        }
        joblib.dump(bundle, model_path)
        artifact = {
            "kind": kind,
            "run_id": run_id,
            "model_file": str(model_path),
        }

    full_metadata = {
        "model_name": model_name,
        "kind": kind,
        "run_id": run_id,
        **metadata,
        "artifact": artifact,
    }
    write_json(metadata_path, full_metadata)
    write_yaml(config_path, config_snapshot)
    artifact["metadata_file"] = str(metadata_path)
    artifact["config_file"] = str(config_path)
    return artifact


def load_model_artifact(artifact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one saved model artifact and its metadata."""
    metadata_path = Path(artifact["metadata_file"])
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    kind = artifact["kind"]
    if kind == "cnn_1d":
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError("Install the 'cnn' extra to load CNN artifacts.") from exc
        model = keras.models.load_model(artifact["model_file"])
        scaler = joblib.load(artifact["scaler_file"])
        return {"kind": kind, "model": model, "scaler": scaler}, metadata

    bundle = joblib.load(artifact["model_file"])
    return {"kind": bundle["kind"], "model": bundle["model"]}, metadata


def load_model_artifacts(
    artifact_index_path: str | Path,
    model_names: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load selected models from a model_artifacts.json file."""
    artifact_index_path = Path(artifact_index_path)
    with artifact_index_path.open("r", encoding="utf-8") as f:
        artifact_index = json.load(f)

    selected_names = model_names or list(artifact_index)
    models = {}
    metadata = {}
    for model_name in selected_names:
        if model_name not in artifact_index:
            raise KeyError(f"Model {model_name!r} not found in {artifact_index_path}.")
        models[model_name], metadata[model_name] = load_model_artifact(artifact_index[model_name])
    return models, metadata
