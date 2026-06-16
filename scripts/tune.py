from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from sar_flood_ml.config import load_config  # noqa: E402
from sar_flood_ml.tuning import tune_random_forest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune model hyperparameters on the configured training split.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default="random_forest", choices=["random_forest"])
    parser.add_argument("--feature-set", help="Optional configured feature set name.")
    parser.add_argument("--n-iter", type=int, default=80, help="Random search iterations.")
    parser.add_argument("--cv", type=int, default=3, help="Stratified CV folds on the training split.")
    parser.add_argument("--n-jobs", type=int, help="Parallel CV jobs. Defaults to a safe local workstation value.")
    parser.add_argument("--output-dir", help="Optional output directory override.")
    parser.add_argument("--no-save-model", action="store_true", help="Do not save the tuned model artifact.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir

    if args.model != "random_forest":
        raise ValueError(f"Unsupported model for tuning: {args.model}")

    result = tune_random_forest(
        cfg,
        feature_set=args.feature_set,
        n_iter=args.n_iter,
        cv_splits=args.cv,
        n_jobs=args.n_jobs,
        save_model=not args.no_save_model,
    )
    summary = result["summary"]
    print("Best RF CV flood F1:", summary["best_cv_flood_f1"])
    print("Best RF params:", summary["best_params"])
    print("Holdout metrics:")
    print(result["metrics"].to_string(index=False))
    print("summary:", result["summary_path"])
    print("cv_results:", result["cv_results_path"])
    print("best_params:", result["best_params_path"])
    if result["artifact_index_path"]:
        print("tuned_artifacts:", result["artifact_index_path"])


if __name__ == "__main__":
    main()
