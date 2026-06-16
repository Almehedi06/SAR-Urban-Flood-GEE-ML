from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a site YAML config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(config_path)
    return cfg


def output_dir(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["paths"]["output_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_config_relative_path(cfg: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    config_path = Path(cfg.get("_config_path", "")).resolve()
    candidates = []
    if config_path:
        candidates.extend([config_path.parent / path, config_path.parent.parent / path])
    candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def artifact_dir(cfg: dict[str, Any]) -> Path:
    path_value = cfg.get("paths", {}).get("artifact_dir", "artifacts")
    path = Path(path_value)
    if not path.is_absolute():
        path = output_dir(cfg) / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def raster_output_dir(cfg: dict[str, Any]) -> Path:
    path_value = cfg.get("paths", {}).get("raster_output_dir", "rasters")
    path = Path(path_value)
    if not path.is_absolute():
        path = output_dir(cfg) / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_dir(cfg: dict[str, Any]) -> Path:
    path_value = cfg.get("paths", {}).get("report_dir", "reports")
    path = Path(path_value)
    if not path.is_absolute():
        path = output_dir(cfg) / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def active_feature_names(cfg: dict[str, Any], feature_set: str | None = None) -> list[str]:
    features_cfg = cfg.get("features", {})
    set_name = feature_set or features_cfg.get("active_set")
    feature_sets = features_cfg.get("sets", {})
    if set_name not in feature_sets:
        raise KeyError(f"Unknown feature set {set_name!r}. Available: {sorted(feature_sets)}")
    return list(feature_sets[set_name])


def active_model_names(cfg: dict[str, Any], model_names: list[str] | None = None) -> list[str]:
    if model_names:
        return list(model_names)
    return list(cfg.get("models", {}).get("active", ["random_forest"]))


def model_config(cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    models_cfg = cfg.get("models", {})
    base = {"random_state": models_cfg.get("random_state", 42)}
    config_dir = models_cfg.get("config_dir")
    if config_dir:
        model_path = _resolve_config_relative_path(cfg, Path(config_dir) / f"{model_name}.yaml")
        if model_path.exists():
            with model_path.open("r", encoding="utf-8") as f:
                base.update(yaml.safe_load(f) or {})
    base.update(deepcopy(models_cfg.get(model_name, {})))
    return base


def sample_columns(cfg: dict[str, Any]) -> dict[str, Any]:
    samples_cfg = cfg.get("samples", {})
    return {
        "label": samples_cfg.get("label_column", "classvalue"),
        "split": samples_cfg.get("split_column", "sample"),
        "train_value": samples_cfg.get("train_value", "train"),
        "test_value": samples_cfg.get("test_value", "test"),
        "flood_class": samples_cfg.get("flood_class", 1),
        "coordinate_columns": samples_cfg.get("coordinate_columns"),
    }


def raster_band_names(cfg: dict[str, Any], count: int, descriptions: tuple[str | None, ...]) -> list[str]:
    configured = cfg.get("raster", {}).get("band_names")
    if configured:
        if len(configured) != count:
            raise ValueError(f"Configured {len(configured)} band names for raster with {count} bands.")
        return list(configured)

    described = [desc for desc in descriptions if desc]
    if len(described) == count:
        return list(described)

    return [f"band_{idx}" for idx in range(1, count + 1)]
