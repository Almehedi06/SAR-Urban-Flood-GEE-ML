from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from sar_flood_ml.pipeline import infer_from_artifacts, train_and_evaluate
from sar_flood_ml.tuning import tune_random_forest


def _write_tiny_raster(path):
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 2, 1, 1),
    }
    data = np.array(
        [
            [[0.1, 0.2], [10.0, 11.0]],
            [[0.1, 0.2], [10.0, 11.0]],
            [[0.1, 0.2], [10.0, 11.0]],
            [[0.1, 0.2], [10.0, 11.0]],
        ],
        dtype="float32",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for idx, name in enumerate(["VV", "VV_1", "VH", "VH_1"], start=1):
            dst.set_band_description(idx, name)


def _write_tiny_samples(path):
    rows = [
        {"VV": 0.1, "VV_1": 0.1, "VH": 0.1, "VH_1": 0.1, "classvalue": 1, "sample": "train"},
        {"VV": 0.2, "VV_1": 0.2, "VH": 0.2, "VH_1": 0.2, "classvalue": 1, "sample": "train"},
        {"VV": 10.0, "VV_1": 10.0, "VH": 10.0, "VH_1": 10.0, "classvalue": 2, "sample": "train"},
        {"VV": 11.0, "VV_1": 11.0, "VH": 11.0, "VH_1": 11.0, "classvalue": 2, "sample": "train"},
        {"VV": 0.15, "VV_1": 0.15, "VH": 0.15, "VH_1": 0.15, "classvalue": 1, "sample": "test"},
        {"VV": 10.5, "VV_1": 10.5, "VH": 10.5, "VH_1": 10.5, "classvalue": 2, "sample": "test"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _tiny_cfg(tmp_path):
    image_path = tmp_path / "features.tif"
    sample_path = tmp_path / "samples.csv"
    output_dir = tmp_path / "outputs"
    _write_tiny_raster(image_path)
    _write_tiny_samples(sample_path)
    return {
        "paths": {
            "image": str(image_path),
            "samples": str(sample_path),
            "aoi": None,
            "output_dir": str(output_dir),
            "artifact_dir": "artifacts",
            "raster_output_dir": "rasters",
            "report_dir": "reports",
        },
        "raster": {
            "band_names": ["VV", "VV_1", "VH", "VH_1"],
            "nodata_label": 0,
            "probability_nodata": -9999.0,
            "compress": "lzw",
        },
        "samples": {
            "label_column": "classvalue",
            "split_column": "sample",
            "train_value": "train",
            "test_value": "test",
            "flood_class": 1,
        },
        "features": {
            "active_set": "sar_4band",
            "sets": {"sar_4band": ["VV", "VV_1", "VH", "VH_1"]},
        },
        "models": {
            "active": ["random_forest"],
            "random_state": 42,
            "random_forest": {
                "n_estimators": 5,
                "min_samples_leaf": 1,
                "n_jobs": 1,
            },
        },
    }


def test_train_saves_versioned_artifacts_and_infer_loads_them(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    trained = train_and_evaluate(cfg, model_names=["random_forest"])

    metrics_path = trained["report_dir"] / "model_metrics.csv"
    artifact_index_path = trained["artifact_dir"] / "model_artifacts.json"
    assert metrics_path.exists()
    assert artifact_index_path.exists()

    with artifact_index_path.open("r", encoding="utf-8") as f:
        artifact_index = json.load(f)
    rf_artifact = artifact_index["random_forest"]
    assert rf_artifact["run_id"] == trained["run_id"]
    assert rf_artifact["model_file"].endswith(".joblib")
    assert rf_artifact["metadata_file"].endswith("_metadata.json")
    assert rf_artifact["config_file"].endswith("_config.yaml")
    assert all((tmp_path / "outputs" / "artifacts" / name).exists() for name in [
        rf_artifact["model_file"].split("/")[-1],
        rf_artifact["metadata_file"].split("/")[-1],
        rf_artifact["config_file"].split("/")[-1],
    ])

    with open(rf_artifact["metadata_file"], "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["features"] == ["VV", "VV_1", "VH", "VH_1"]
    assert metadata["model_config"]["n_estimators"] == 5

    outputs = infer_from_artifacts(cfg, artifact_index_path, model_names=["random_forest"])
    class_map = outputs["random_forest"]["class_map"]
    probability_map = outputs["random_forest"]["flood_probability_map"]
    assert class_map.endswith("random_forest_class_map.tif")
    assert probability_map.endswith("random_forest_flood_probability.tif")

    with rasterio.open(class_map) as src:
        assert src.shape == (2, 2)
        assert src.crs.to_string() == "EPSG:4326"
    with rasterio.open(probability_map) as src:
        assert src.shape == (2, 2)


def test_random_forest_tuning_saves_reports_and_tuned_artifact(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    result = tune_random_forest(
        cfg,
        n_iter=2,
        cv_splits=2,
        n_jobs=1,
        param_distributions={
            "n_estimators": [2, 3],
            "max_depth": [None],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "max_features": ["sqrt"],
            "class_weight": ["balanced"],
        },
    )

    assert result["summary_path"].exists()
    assert result["cv_results_path"].exists()
    assert result["best_params_path"].exists()
    assert result["metrics_path"].exists()
    assert result["artifact_index_path"].exists()

    with open(result["artifact_index_path"], "r", encoding="utf-8") as f:
        artifact_index = json.load(f)
    assert "random_forest" in artifact_index
    assert artifact_index["random_forest"]["model_file"].endswith(".joblib")

    best_params_text = result["best_params_path"].read_text(encoding="utf-8")
    assert "n_estimators:" in best_params_text
    assert "n_jobs:" in best_params_text
