from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from sar_flood_ml.config import load_config  # noqa: E402
from sar_flood_ml.pipeline import validate_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prepared raster/sample inputs.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    result = validate_data(load_config(args.config))
    print("features:", result["features"])
    print("classes:", result["classes"])
    print("n_samples:", result["n_samples"])
    print("n_train:", result["n_train"])
    print("n_test:", result["n_test"])
    print(result["summary"])


if __name__ == "__main__":
    main()
