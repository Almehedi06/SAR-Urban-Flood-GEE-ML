from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from sar_flood_ml.config import load_config  # noqa: E402
from sar_flood_ml.pipeline import run_all, run_experiments, run_inference, train_and_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training, evaluation, and raster prediction.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", help="Optional output directory override.")
    parser.add_argument("--experiments", action="store_true", help="Also run repeated-split and learning-curve experiments.")
    args = parser.parse_args()
    if args.output_dir:
        cfg = load_config(args.config)
        cfg["paths"]["output_dir"] = args.output_dir
        trained = train_and_evaluate(cfg)
        predictions = run_inference(cfg, trained)
        experiments = run_experiments(cfg) if args.experiments else {}
        print(
            {
                "metrics": str(trained["report_dir"] / "model_metrics.csv"),
                "artifacts": str(trained["artifact_dir"] / "model_artifacts.json"),
                "predictions": predictions,
                "experiments": {key: str(path) for key, path in experiments.items()},
            }
        )
    else:
        print(run_all(args.config, run_experiment_suite=args.experiments))


if __name__ == "__main__":
    main()
