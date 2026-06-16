from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

from .config import raster_band_names
from .features import build_feature_frame


def load_samples(path: str | Path) -> pd.DataFrame:
    """Load prepared sample data from CSV or a vector file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".gpkg", ".geojson", ".json", ".shp"}:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError("Install the 'vector' extra to read vector sample files.") from exc
        return gpd.read_file(path)

    raise ValueError(f"Unsupported sample file type: {path.suffix}")


def validate_samples(
    samples: pd.DataFrame,
    feature_names: list[str],
    label_column: str,
    split_column: str | None = None,
) -> pd.DataFrame:
    """Validate prepared samples and attach any configured derived features."""
    required = [label_column]
    if split_column:
        required.append(split_column)

    missing = [column for column in required if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing required sample columns: {missing}")

    feature_frame = build_feature_frame(samples, feature_names)
    checked = samples.copy()
    checked[feature_names] = feature_frame
    checked = checked.dropna(subset=feature_names + required).copy()
    checked[label_column] = checked[label_column].astype(int)
    return checked


def train_test_from_split(
    samples: pd.DataFrame,
    feature_names: list[str],
    label_column: str,
    split_column: str,
    train_value: Any = "train",
    test_value: Any = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Create ML-ready X/y data using the prepared split column."""
    split_values = samples[split_column].astype(str).str.lower()
    train_mask = split_values == str(train_value).lower()
    test_mask = split_values == str(test_value).lower()
    if not train_mask.any() or not test_mask.any():
        raise ValueError(f"Split column {split_column!r} must contain train and test rows.")

    features = build_feature_frame(samples, feature_names)
    X_train = features.loc[train_mask].astype("float32")
    X_test = features.loc[test_mask].astype("float32")
    y_train = samples.loc[train_mask, label_column].astype(int)
    y_test = samples.loc[test_mask, label_column].astype(int)
    return X_train, X_test, y_train, y_test, samples


def sample_summary(samples: pd.DataFrame, label_column: str, split_column: str | None = None) -> pd.DataFrame:
    if split_column and split_column in samples.columns:
        return pd.crosstab(samples[split_column], samples[label_column], margins=True)
    return samples[label_column].value_counts().sort_index().rename_axis(label_column).reset_index(name="count")


def read_feature_raster(path: str | Path, cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a prepared multiband feature raster into a band-name mapping."""
    with rasterio.open(path) as src:
        stack = src.read().astype("float32")
        profile = src.profile.copy()
        band_names = raster_band_names(cfg, src.count, src.descriptions)

    arrays = {name: stack[idx] for idx, name in enumerate(band_names)}
    return arrays, profile


def write_single_band_raster(
    path: str | Path,
    array: np.ndarray,
    profile: dict[str, Any],
    dtype: str,
    nodata: float | int | None,
    compress: str = "lzw",
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, nodata=nodata, compress=compress)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)
    return out_path


def _load_aoi_geometries(path: str | Path | None, dst_crs: Any) -> list[dict[str, Any]]:
    if path in (None, "", "null"):
        return []

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Install the 'vector' extra to use AOI geometry files.") from exc

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"AOI file has no geometries: {path}")
    if gdf.crs is not None and dst_crs is not None and gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)
    return [geom.__geo_interface__ for geom in gdf.geometry if geom is not None and not geom.is_empty]


def rasterized_aoi_mask(
    path: str | Path | None,
    profile: dict[str, Any],
    shape: tuple[int, int],
) -> np.ndarray:
    """Return True inside AOI. If no AOI is configured, everything is True."""
    geometries = _load_aoi_geometries(path, profile.get("crs"))
    if not geometries:
        return np.ones(shape, dtype=bool)

    return geometry_mask(
        geometries,
        out_shape=shape,
        transform=profile["transform"],
        invert=True,
    )
