from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from sar_flood_ml.config import load_config  # noqa: E402
from sar_flood_ml.pipeline import run_inference, train_and_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train configured models and write flood rasters.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models", nargs="*", help="Optional model names to train and predict.")
    parser.add_argument("--output-dir", help="Optional output directory override.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir
    trained = train_and_evaluate(cfg, model_names=args.models)
    outputs = run_inference(cfg, trained)
    print(outputs)


if __name__ == "__main__":
    main()
