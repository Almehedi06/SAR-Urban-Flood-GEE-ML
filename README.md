# SAR Urban Flood ML

Config-driven SAR flood mapping tools for prepared multiband SAR feature
rasters and prepared sample tables.

The project is designed to keep large input data outside the repository while
making the model training, evaluation, and GeoTIFF flood-map outputs
reproducible from one main project config and small per-model config files.

## Current Scope

The first supported workflow is tabular/pixel-feature ML:

- input feature raster: multiband GeoTIFF
- sample table: CSV or vector file with feature columns, label column, and train/test split
- optional AOI geometry: Shapefile, GeoPackage, or GeoJSON
- models: Random Forest, optional XGBoost, optional 1D feature CNN
- outputs: class-map GeoTIFF, flood-probability GeoTIFF, metrics, model artifacts

The current Houston sample CSV has this format:

```text
VV,VV_1,VH,VH_1,classvalue,sample
```

It does not include coordinates or spatial block IDs. That means the current
evaluation supports sample-level generalization experiments, but not true
spatial cross-validation. Coordinate-aware and patch-based spatial models are
planned extension points.

## Repository Layout

```text
configs/config.yaml     Main paths, sample columns, feature sets, outputs
configs/models/         Per-model defaults for RF, XGBoost, CNN
src/sar_flood_ml/       Reusable pipeline package
scripts/                CLI entry points
notebooks/              Notebook interfaces and exploration
reports/                Lightweight generated figures/tables
tests/                  Unit tests for reusable logic
```

Existing notebooks at the repo root are left in place for now:

- `P3_Houston_CNN.ipynb`: original Colab/GEE-style notebook
- `P3_Houston_Local_ML.ipynb`: local notebook prototype

## Quick Start

Install the package in the active environment:

```bash
pip install -e .
```

Validate the prepared inputs and sample split:

```bash
python scripts/validate_data.py --config configs/config.yaml
```

Train and evaluate configured models:

```bash
python scripts/train.py --config configs/config.yaml
```

Train and write flood maps in one step:

```bash
python scripts/predict.py --config configs/config.yaml
```

Load saved models and write flood maps without retraining:

```bash
python scripts/infer.py --config configs/config.yaml
```

Tune Random Forest hyperparameters on the configured training split:

```bash
python scripts/tune.py --config configs/config.yaml --model random_forest --n-iter 80 --cv 3
```

Use more local CPU cores for a larger run:

```bash
python scripts/tune.py --config configs/config.yaml --model random_forest --n-iter 120 --cv 5 --n-jobs 24
```

Run repeated-split and learning-curve generalization experiments:

```bash
python scripts/run_all.py --config configs/config.yaml --experiments
```

Run the standard end-to-end workflow:

```bash
python scripts/run_all.py --config configs/config.yaml
```

## Config Pattern

The current repo assumes inputs are already prepared. The main config points to
one feature raster, one sample table, and output folders:

```yaml
paths:
  image: /path/to/multiband_features.tif
  samples: /path/to/samples.csv
  aoi: /path/to/aoi.gpkg
  output_dir: /path/to/outputs
  artifact_dir: artifacts
  raster_output_dir: rasters
  report_dir: reports

raster:
  band_names: [VV, VV_1, VH, VH_1]

samples:
  label_column: classvalue
  split_column: sample
  train_value: train
  test_value: test
  flood_class: 1

features:
  active_set: sar_4band
  sets:
    sar_4band: [VV, VV_1, VH, VH_1]
    sar_with_ratio: [VV, VV_1, VH, VH_1, VV_div_VH]
```

Model hyperparameters live under `configs/models/`. This keeps the main config
short while still making model defaults easy to edit.

RF tuning writes its own outputs under the configured output directory:

- `reports/tuning/random_forest_random_search_cv_results.csv`
- `reports/tuning/random_forest_best_params.yaml`
- `reports/tuning/random_forest_tuned_holdout_metrics.csv`
- `reports/tuning/random_forest_tuning_summary.json`
- `artifacts/tuning/random_forest_tuned_artifacts.json`

The normal training artifact index is not overwritten by tuning. To infer from
the tuned RF artifact, pass the tuned artifact index explicitly:

```bash
python scripts/infer.py --config configs/config.yaml \
  --artifacts /path/to/outputs/artifacts/tuning/random_forest_tuned_artifacts.json
```

## Workflow Design

The code follows a simple ML workflow:

```text
config -> data prep -> feature engineering -> model training -> evaluation -> inference
```

In this repo, data prep means loading already prepared inputs and making ML-ready
`X` and `y`. Feature engineering means selecting configured features and adding
simple derived features such as ratios. Model-specific preprocessing, such as
standardization for CNN/MLP-style models, is fit only on training data and saved
with the model artifact.

Training writes versioned model artifacts under the configured `artifact_dir`.
Each trained model gets:

- a timestamped model file
- a metadata JSON file
- a config YAML snapshot

The latest trained models are indexed in `model_artifacts.json`, which is what
`scripts/infer.py` uses for no-retraining raster prediction.

## Model Notes

Random Forest is the baseline model. XGBoost is a strong optional tabular
model. The 1D CNN is included for comparison with the original notebook, but it
is a feature-sequence CNN over per-pixel bands, not a spatial CNN.

Model implementations live under `src/sar_flood_ml/models/`:

- `random_forest.py`: Random Forest train/predict logic
- `xgboost_model.py`: XGBoost train/predict logic
- `cnn_1d.py`: 1D feature-CNN train/predict logic
- `__init__.py`: registry and dispatch layer used by the training/inference pipeline

True spatial CNNs such as U-Net or patch-based CNNs require coordinates,
spatial labels, masks, or patch datasets. The package structure leaves room for
that next stage without forcing it into the current Houston CSV workflow.

## Citation

Al Mehedi, M. A., Smith, V., & Kremer, P. (2024). A comparative analysis of
urban and peri-urban flood identification using SAR imagery. Villanova Centre
of Resilient Water System and Department of Geography and the Environment,
Villanova University. (Under review).
