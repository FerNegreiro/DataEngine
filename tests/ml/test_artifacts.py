from __future__ import annotations

import json

import pandas as pd

from src.ml.artifacts import save_ml_artifacts


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
