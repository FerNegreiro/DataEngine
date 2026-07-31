from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.ml.config import ARTIFACTS_DIR


@dataclass(frozen=True)
class ArtifactPaths:
    model: Path
    metadata: Path
    metrics: Path
    feature_columns: Path
    forecasts: Path
    inventory_risk: Path

    def to_dict(self) -> dict[str, str]:
        return {name: str(path) for name, path in asdict(self).items()}


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Objeto não serializável em JSON: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def save_ml_artifacts(
    *,
    model: Any,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    feature_columns: list[str] | tuple[str, ...],
    forecasts: pd.DataFrame,
    inventory_risk: pd.DataFrame,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> ArtifactPaths:
    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = ArtifactPaths(
        model=directory / "model.joblib",
        metadata=directory / "metadata.json",
        metrics=directory / "metrics.json",
        feature_columns=directory / "feature_columns.json",
        forecasts=directory / "forecasts.parquet",
        inventory_risk=directory / "inventory_risk.parquet",
    )
    joblib.dump(model, paths.model)
    _write_json(paths.metadata, metadata)
    _write_json(paths.metrics, metrics)
    _write_json(paths.feature_columns, list(feature_columns))
    forecasts.to_parquet(paths.forecasts, index=False)
    inventory_risk.to_parquet(paths.inventory_risk, index=False)
    return paths
