from __future__ import annotations

import json

import pandas as pd

from src.ml.artifacts import (
    save_experiment_artifacts,
    save_ml_artifacts,
    save_production_artifacts,
)
from src.ml.production import ProductionBundle


def test_save_ml_artifacts_writes_the_required_files(tmp_path: object) -> None:
    forecasts = pd.DataFrame(
        {
            "product_id": ["P1"],
            "forecast_date": [pd.Timestamp("2026-08-01")],
            "horizon_day": [1],
            "predicted_quantity": [1.5],
        }
    )
    risk = pd.DataFrame({"product_id": ["P1"], "risk_class": ["adequate"]})
    paths = save_ml_artifacts(
        model={"kind": "test"},
        metadata={"generated_at": pd.Timestamp("2026-07-31", tz="UTC")},
        metrics={"wape": 0.5},
        feature_columns=["lag_1"],
        forecasts=forecasts,
        inventory_risk=risk,
        artifacts_dir=tmp_path,
    )

    assert all(path.is_file() for path in paths.__dict__.values())
    assert json.loads(paths.feature_columns.read_text(encoding="utf-8")) == ["lag_1"]
    assert pd.read_parquet(paths.forecasts).loc[0, "predicted_quantity"] == 1.5


def test_save_experiment_artifacts_preserves_complete_iteration_handoff(
    tmp_path: object,
) -> None:
    metric = pd.DataFrame([{"split": "fold", "model_name": "m", "wape": 1.0}])
    segments = pd.DataFrame([{"product_id": "P1", "demand_pattern": "intermittent"}])
    forecasts = pd.DataFrame([{"product_id": "P1", "predicted_quantity": 1.0}])
    risk = pd.DataFrame([{"product_id": "P1", "risk_class": "adequate"}])
    paths = save_experiment_artifacts(
        aggregate_metrics=metric,
        occurrence_metrics=metric,
        segment_metrics=metric,
        product_metrics=metric,
        demand_segments=segments,
        model_comparison={"champion": "m"},
        promotion_decision={"decision": "rejected"},
        forecasts=forecasts,
        inventory_risk_comparison=risk,
        models={"m": {"kind": "test"}},
        artifacts_dir=tmp_path,
    )
    assert paths.metrics.is_file()
    assert paths.segment_metrics.is_file()
    assert paths.models["m"].is_file()
    assert pd.read_parquet(paths.demand_segments).loc[0, "product_id"] == "P1"


def test_save_production_artifacts_creates_retryable_bundle(
    production_bundle: ProductionBundle,
    tmp_path: object,
) -> None:
    paths = save_production_artifacts(
        manifest=production_bundle.manifest,
        forecasts=production_bundle.forecasts,
        inventory_risk=production_bundle.inventory_risk,
        model_metrics=production_bundle.model_metrics,
        model_registry=production_bundle.model_registry,
        pipeline_run=production_bundle.pipeline_run,
        artifacts_dir=tmp_path,
    )
    assert all(path.is_file() for path in paths.__dict__.values())
    assert json.loads(paths.manifest.read_text(encoding="utf-8"))["champion_model"] == (
        "moving_average_28"
    )
    assert pd.read_parquet(paths.forecasts)["horizon_days"].unique().tolist() == [
        7,
        14,
        30,
    ]
