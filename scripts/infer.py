from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from sar_flood_ml.config import artifact_dir, load_config  # noqa: E402
from sar_flood_ml.pipeline import infer_from_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Load saved models and write flood rasters without retraining.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--artifacts", help="Path to model_artifacts.json. Defaults to the configured artifact_dir.")
    parser.add_argument("--models", nargs="*", help="Optional model names to load from the artifact index.")
    parser.add_argument("--output-dir", help="Optional output directory override.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir

    artifact_index = args.artifacts or artifact_dir(cfg) / "model_artifacts.json"
    outputs = infer_from_artifacts(cfg, artifact_index, model_names=args.models)
    print(outputs)


if __name__ == "__main__":
    main()
