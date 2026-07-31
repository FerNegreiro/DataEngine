from __future__ import annotations

from typing import Any

import pandas as pd

from src.ml.production import ProductionBundle
from src.validation.validate_ml_bigquery_load import (
    validate_ml_bigquery_load,
    validate_ml_publication,
)


def _validate(bundle: ProductionBundle) -> dict[str, Any]:
    return validate_ml_publication(
        manifest=bundle.manifest,
        forecasts=bundle.forecasts,
        inventory_risk=bundle.inventory_risk,
        model_metrics=bundle.model_metrics,
        model_registry=bundle.model_registry,
        pipeline_run=bundle.pipeline_run,
    )


def test_preload_validation_accepts_complete_official_bundle(
    production_bundle: ProductionBundle,
) -> None:
    report = _validate(production_bundle)
    assert report["is_valid"] is True
    assert report["forecast_horizons"] == [7, 14, 30]
    assert report["products_processed"] == 1


def test_preload_validation_rejects_negative_and_duplicate_forecasts(
    production_bundle: ProductionBundle,
) -> None:
    production_bundle.forecasts.loc[0, "predicted_quantity"] = -1
    production_bundle.forecasts = pd.concat(
        [production_bundle.forecasts, production_bundle.forecasts.iloc[[0]]],
        ignore_index=True,
    )
    production_bundle.manifest["forecast_rows"] = len(production_bundle.forecasts)
    production_bundle.pipeline_run["forecast_rows"] = len(production_bundle.forecasts)
    report = _validate(production_bundle)
    assert report["is_valid"] is False
    assert any("não negativa" in error for error in report["errors"])
    assert any("duplicidade" in error for error in report["errors"])


def test_preload_validation_rejects_invalid_risk_and_challenger_forecast(
    production_bundle: ProductionBundle,
) -> None:
    production_bundle.inventory_risk.loc[0, "risk_level"] = "unknown"
    production_bundle.forecasts["model_name"] = "croston_sba"
    report = _validate(production_bundle)
    assert report["is_valid"] is False
    assert any("risk_level inválido" in error for error in report["errors"])
    assert any("somente o champion" in error for error in report["errors"])


class FakeResultJob:
    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row

    def result(self) -> list[dict[str, Any]]:
        return [self.row]


class FakeValidationClient:
    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        self.queries: list[str] = []

    def get_dataset(self, _: str) -> Any:
        return type("Dataset", (), {"location": "southamerica-east1"})()

    def get_table(self, table_id: str) -> Any:
        return type("Table", (), {"table_id": table_id})()

    def query(self, query: str, **_: Any) -> FakeResultJob:
        self.queries.append(query)
        return FakeResultJob(self.row)


def test_postload_validation_confirms_counts_horizons_and_single_champion(
    production_bundle: ProductionBundle,
) -> None:
    row = {
        "forecast_rows": len(production_bundle.forecasts),
        "forecast_duplicate_rows": 0,
        "forecast_horizons": [7, 14, 30],
        "forecast_products": 1,
        "invalid_forecast_model_rows": 0,
        "risk_rows": len(production_bundle.inventory_risk),
        "risk_duplicate_rows": 0,
        "invalid_risk_rows": 0,
        "metric_rows": len(production_bundle.model_metrics),
        "active_champion_rows": 1,
        "pipeline_run_rows": 1,
        "pipeline_status": "success",
    }
    client = FakeValidationClient(row)
    report = validate_ml_bigquery_load(
        client,
        run_id=production_bundle.manifest["run_id"],
        expected_forecast_rows=len(production_bundle.forecasts),
        expected_risk_rows=len(production_bundle.inventory_risk),
        expected_metric_rows=len(production_bundle.model_metrics),
        expected_products=1,
    )
    assert report["is_valid"] is True
    assert set(report["tables"]) == {
        "sales_forecast",
        "inventory_risk",
        "model_metrics",
        "model_registry",
        "pipeline_runs",
    }


def test_postload_validation_detects_count_mismatch(
    production_bundle: ProductionBundle,
) -> None:
    row = {
        "forecast_rows": 0,
        "forecast_duplicate_rows": 0,
        "forecast_horizons": [],
        "forecast_products": 0,
        "invalid_forecast_model_rows": 0,
        "risk_rows": 0,
        "risk_duplicate_rows": 0,
        "invalid_risk_rows": 0,
        "metric_rows": 0,
        "active_champion_rows": 0,
        "pipeline_run_rows": 0,
        "pipeline_status": "failed",
    }
    report = validate_ml_bigquery_load(
        FakeValidationClient(row),
        run_id=production_bundle.manifest["run_id"],
        expected_forecast_rows=len(production_bundle.forecasts),
        expected_risk_rows=len(production_bundle.inventory_risk),
        expected_metric_rows=len(production_bundle.model_metrics),
        expected_products=1,
    )
    assert report["is_valid"] is False
    assert any("forecast_rows divergente" in error for error in report["errors"])
