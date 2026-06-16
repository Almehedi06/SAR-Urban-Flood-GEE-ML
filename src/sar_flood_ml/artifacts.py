from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import pandas as pd
import yaml


def make_run_id() -> str:
    """Create a simple timestamp ID for versioned model outputs."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def write_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def write_dataframe(path: str | Path, frame: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
