from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .features import build_feature_matrix
from .models import predict_labels_and_flood_probability
from .data import read_feature_raster, rasterized_aoi_mask, write_single_band_raster


def predict_raster(
    cfg: dict[str, Any],
    models: dict[str, dict[str, Any]],
    feature_names: list[str],
    classes: list[int],
    flood_class: int,
    out_dir: str | Path,
    batch_size: int = 65536,
) -> dict[str, dict[str, str]]:
    arrays, profile = read_feature_raster(cfg["paths"]["image"], cfg)
    first = next(iter(arrays.values()))
    height, width = first.shape
    feature_matrix = build_feature_matrix(arrays, feature_names)
    valid = np.all(np.isfinite(feature_matrix), axis=1)

    aoi_mask = rasterized_aoi_mask(cfg.get("paths", {}).get("aoi"), profile, (height, width)).reshape(-1)
    valid &= aoi_mask
    valid_indices = np.flatnonzero(valid)

    raster_cfg = cfg.get("raster", {})
    nodata_label = int(raster_cfg.get("nodata_label", 0))
    probability_nodata = float(raster_cfg.get("probability_nodata", -9999.0))
    compress = raster_cfg.get("compress", "lzw")
    out_dir = Path(out_dir)
    outputs = {}

    for model_name, model_info in models.items():
        class_flat = np.full(feature_matrix.shape[0], nodata_label, dtype="uint8")
        probability_flat = np.full(feature_matrix.shape[0], np.nan, dtype="float32")

        for start in range(0, len(valid_indices), batch_size):
            idx = valid_indices[start:start + batch_size]
            labels, probabilities = predict_labels_and_flood_probability(
                model_info,
                feature_matrix[idx],
                classes,
                flood_class,
            )
            class_flat[idx] = labels.astype("uint8")
            probability_flat[idx] = probabilities.astype("float32")

        class_map = class_flat.reshape(height, width)
        probability_map = probability_flat.reshape(height, width)
        probability_to_write = np.where(np.isfinite(probability_map), probability_map, probability_nodata)

        class_path = write_single_band_raster(
            out_dir / f"{model_name}_class_map.tif",
            class_map,
            profile,
            dtype="uint8",
            nodata=nodata_label,
            compress=compress,
        )
        probability_path = write_single_band_raster(
            out_dir / f"{model_name}_flood_probability.tif",
            probability_to_write,
            profile,
            dtype="float32",
            nodata=probability_nodata,
            compress=compress,
        )
        outputs[model_name] = {
            "class_map": str(class_path),
            "flood_probability_map": str(probability_path),
            "valid_pixels": int(valid.sum()),
        }
    return outputs
