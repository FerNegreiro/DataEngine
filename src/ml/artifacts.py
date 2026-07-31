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


@dataclass(frozen=True)
class ExperimentArtifactPaths:
    metrics: Path
    segment_metrics: Path
    demand_segments: Path
    model_comparison: Path
    promotion_decision: Path
    forecasts: Path
    inventory_risk_comparison: Path
    models: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            name: str(path)
            for name, path in asdict(self).items()
            if name != "models"
        }
        payload["models"] = {
            name: str(path) for name, path in self.models.items()
        }
        return payload


@dataclass(frozen=True)
class ProductionArtifactPaths:
    manifest: Path
    forecasts: Path
    inventory_risk: Path
    model_metrics: Path
    model_registry: Path
    pipeline_run: Path

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


def _dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    sanitized = dataframe.astype(object).where(pd.notna(dataframe), None)
    return sanitized.to_dict(orient="records")


def save_experiment_artifacts(
    *,
    aggregate_metrics: pd.DataFrame,
    occurrence_metrics: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    product_metrics: pd.DataFrame,
    demand_segments: pd.DataFrame,
    model_comparison: dict[str, Any],
    promotion_decision: dict[str, Any],
    forecasts: pd.DataFrame,
    inventory_risk_comparison: pd.DataFrame,
    models: dict[str, Any],
    artifacts_dir: Path | str,
) -> ExperimentArtifactPaths:
    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    models_directory = directory / "models"
    models_directory.mkdir(parents=True, exist_ok=True)
    model_paths = {
        name: models_directory / f"{name}.joblib" for name in sorted(models)
    }
    paths = ExperimentArtifactPaths(
        metrics=directory / "metrics.json",
        segment_metrics=directory / "segment_metrics.json",
        demand_segments=directory / "demand_segments.parquet",
        model_comparison=directory / "model_comparison.json",
        promotion_decision=directory / "promotion_decision.json",
        forecasts=directory / "forecasts.parquet",
        inventory_risk_comparison=directory / "inventory_risk_comparison.parquet",
        models=model_paths,
    )
    _write_json(
        paths.metrics,
        {
            "aggregate_metrics": _dataframe_records(aggregate_metrics),
            "occurrence_metrics": _dataframe_records(occurrence_metrics),
        },
    )
    _write_json(
        paths.segment_metrics,
        {
            "segment_metrics": _dataframe_records(segment_metrics),
            "product_metrics": _dataframe_records(product_metrics),
        },
    )
    _write_json(paths.model_comparison, model_comparison)
    _write_json(paths.promotion_decision, promotion_decision)
    demand_segments.to_parquet(paths.demand_segments, index=False)
    forecasts.to_parquet(paths.forecasts, index=False)
    inventory_risk_comparison.to_parquet(
        paths.inventory_risk_comparison, index=False
    )
    for name, model in models.items():
        joblib.dump(model, model_paths[name])
    return paths


def save_production_artifacts(
    *,
    manifest: dict[str, Any],
    forecasts: pd.DataFrame,
    inventory_risk: pd.DataFrame,
    model_metrics: pd.DataFrame,
    model_registry: pd.DataFrame,
    pipeline_run: dict[str, Any],
    artifacts_dir: Path | str,
) -> ProductionArtifactPaths:
    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = ProductionArtifactPaths(
        manifest=directory / "publication_manifest.json",
        forecasts=directory / "sales_forecast.parquet",
        inventory_risk=directory / "inventory_risk.parquet",
        model_metrics=directory / "model_metrics.parquet",
        model_registry=directory / "model_registry.parquet",
        pipeline_run=directory / "pipeline_run.json",
    )
    _write_json(paths.manifest, manifest)
    _write_json(paths.pipeline_run, pipeline_run)
    forecasts.to_parquet(paths.forecasts, index=False)
    inventory_risk.to_parquet(paths.inventory_risk, index=False)
    model_metrics.to_parquet(paths.model_metrics, index=False)
    model_registry.to_parquet(paths.model_registry, index=False)
    return paths
