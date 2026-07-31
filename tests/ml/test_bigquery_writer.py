from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from google.api_core.exceptions import BadRequest, Forbidden, NotFound
from google.auth.exceptions import DefaultCredentialsError

from src.ml import bigquery_writer
from src.ml.bigquery_writer import (
    ML_TABLE_SCHEMAS,
    MLBigQueryPublishError,
    create_bigquery_client,
    ensure_ml_dataset,
    ensure_ml_tables,
    merge_dataframe,
    publish_ml_dataframes,
)
from src.ml.production import ProductionBundle


class FakeJob:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def result(self) -> list[dict[str, Any]]:
        return []


class FakeDataset:
    def __init__(self, location: str) -> None:
        self.location = location


class FakeBigQueryClient:
    def __init__(
        self,
        *,
        dataset_exists: bool = False,
        location: str = "southamerica-east1",
        write_error: Exception | None = None,
    ) -> None:
        self.dataset_exists = dataset_exists
        self.location = location
        self.write_error = write_error
        self.tables: dict[str, Any] = {}
        self.loads: list[tuple[pd.DataFrame, str, Any]] = []
        self.queries: list[str] = []
        self.deleted: list[str] = []
        self.updated: list[tuple[Any, list[str]]] = []

    def get_dataset(self, _: str) -> FakeDataset:
        if not self.dataset_exists:
            raise NotFound("missing")
        return FakeDataset(self.location)

    def create_dataset(self, dataset: Any) -> Any:
        self.dataset_exists = True
        return dataset

    def get_table(self, table_id: str) -> Any:
        if table_id not in self.tables:
            raise NotFound("missing")
        return self.tables[table_id]

    def create_table(self, table: Any) -> Any:
        table_id = f"{table.project}.{table.dataset_id}.{table.table_id}"
        self.tables[table_id] = table
        return table

    def update_table(self, table: Any, fields: list[str]) -> Any:
        self.updated.append((table, fields))
        return table

    def load_table_from_dataframe(
        self, dataframe: pd.DataFrame, table_id: str, **kwargs: Any
    ) -> FakeJob:
        if self.write_error:
            raise self.write_error
        self.loads.append((dataframe.copy(), table_id, kwargs["job_config"]))
        return FakeJob("load-job")

    def query(self, query: str, **_: Any) -> FakeJob:
        if self.write_error:
            raise self.write_error
        self.queries.append(query)
        return FakeJob("merge-job")

    def delete_table(self, table_id: str, **_: Any) -> None:
        self.deleted.append(table_id)


def test_creates_dataset_and_all_five_tables_with_exact_schemas() -> None:
    client = FakeBigQueryClient()
    dataset = ensure_ml_dataset(client)
    tables = ensure_ml_tables(client)
    assert dataset["created"] is True
    assert set(tables) == set(ML_TABLE_SCHEMAS)
    assert all(report["created"] for report in tables.values())
    expected_fields = {
        "sales_forecast": {
            "run_id",
            "generated_at",
            "product_id",
            "forecast_date",
            "horizon_day",
            "horizon_days",
            "predicted_quantity",
            "model_name",
            "model_version",
            "champion_status",
            "data_max_date",
            "source_project",
            "source_dataset",
        },
        "inventory_risk": {
            "run_id",
            "generated_at",
            "product_id",
            "product_name",
            "category",
            "stock_quantity",
            "minimum_stock",
            "forecast_demand",
            "projected_stock",
            "average_daily_demand",
            "estimated_coverage_days",
            "risk_level",
            "model_name",
            "model_version",
            "horizon_days",
        },
        "model_metrics": {
            "run_id",
            "model_name",
            "model_version",
            "champion_status",
            "evaluation_period",
            "forecast_horizon",
            "metric_name",
            "metric_value",
            "generated_at",
        },
        "model_registry": {
            "model_name",
            "model_version",
            "registered_at",
            "status",
            "is_champion",
            "promotion_decision",
            "rejection_reason",
            "primary_metric",
            "primary_metric_value",
            "bias",
            "training_data_min_date",
            "training_data_max_date",
            "code_version",
            "metadata_json",
        },
        "pipeline_runs": {
            "run_id",
            "started_at",
            "finished_at",
            "status",
            "source_data_min_date",
            "source_data_max_date",
            "products_processed",
            "forecast_rows",
            "risk_rows",
            "champion_model",
            "champion_version",
            "error_message",
            "duration_seconds",
        },
    }
    assert {
        table_name: {field.name for field in schema}
        for table_name, schema in ML_TABLE_SCHEMAS.items()
    } == expected_fields


def test_rejects_existing_dataset_in_the_wrong_region() -> None:
    client = FakeBigQueryClient(dataset_exists=True, location="US")
    with pytest.raises(ValueError, match="região incompatível"):
        ensure_ml_dataset(client)


def test_existing_historical_table_expiration_is_cleared() -> None:
    client = FakeBigQueryClient(dataset_exists=True)
    ensure_ml_tables(client)
    table_id = "dataengine-fernando-2026.dataengine_ml.sales_forecast"
    table = next(
        table
        for key, table in client.tables.items()
        if key.endswith("dataengine_ml.sales_forecast")
    )
    table.expires = pd.Timestamp("2026-09-29T00:00:00Z").to_pydatetime()
    client.tables[table_id] = table
    reports = ensure_ml_tables(client)
    assert reports["sales_forecast"]["expiration_cleared"] is True
    assert table.expires is None
    assert client.updated[-1][1] == ["expires"]


def test_merge_uses_staging_and_never_truncates_the_historical_target() -> None:
    client = FakeBigQueryClient(dataset_exists=True)
    ensure_ml_tables(client)
    frame = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "started_at": pd.Timestamp("2026-07-31T12:00:00Z"),
                "finished_at": None,
                "status": "running",
                "source_data_min_date": pd.Timestamp("2023-01-06").date(),
                "source_data_max_date": pd.Timestamp("2026-07-28").date(),
                "products_processed": 1,
                "forecast_rows": 51,
                "risk_rows": 1,
                "champion_model": "moving_average_28",
                "champion_version": "1.0.0",
                "error_message": None,
                "duration_seconds": None,
            }
        ]
    )
    first = merge_dataframe(
        client, table_name="pipeline_runs", dataframe=frame, run_id="run-1"
    )
    second = merge_dataframe(
        client, table_name="pipeline_runs", dataframe=frame, run_id="run-1"
    )
    assert first["write_strategy"] == "staging_write_truncate_then_target_merge"
    assert second["merge_keys"] == ["run_id"]
    assert len(client.queries) == 2
    assert all(
        "MERGE `dataengine-fernando-2026.dataengine_ml.pipeline_runs`" in sql
        for sql in client.queries
    )
    assert all("WRITE_TRUNCATE" not in sql for sql in client.queries)
    assert len(client.deleted) == 2


def test_publish_writes_forecast_risk_metrics_and_registry_with_merge(
    production_bundle: ProductionBundle,
) -> None:
    client = FakeBigQueryClient(dataset_exists=True)
    ensure_ml_tables(client)
    reports = publish_ml_dataframes(
        client,
        forecasts=production_bundle.forecasts,
        inventory_risk=production_bundle.inventory_risk,
        model_metrics=production_bundle.model_metrics,
        model_registry=production_bundle.model_registry,
        run_id=production_bundle.manifest["run_id"],
    )
    assert set(reports) == {
        "sales_forecast",
        "inventory_risk",
        "model_metrics",
        "model_registry",
    }
    assert all(report["input_rows"] > 0 for report in reports.values())
    assert len(client.queries) == 4


@pytest.mark.parametrize("error", [Forbidden("denied"), BadRequest("bad request")])
def test_merge_translates_permission_and_bad_request_errors(error: Exception) -> None:
    client = FakeBigQueryClient(dataset_exists=True, write_error=error)
    frame = pd.DataFrame(columns=[field.name for field in ML_TABLE_SCHEMAS["pipeline_runs"]])
    frame.loc[0] = ["run-1", None, None, "running", None, None, 1, 1, 1, "m", "v", None, None]
    with pytest.raises(MLBigQueryPublishError, match="MERGE"):
        merge_dataframe(
            client, table_name="pipeline_runs", dataframe=frame, run_id="run-1"
        )


def test_client_translates_adc_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_client(**_: Any) -> None:
        raise DefaultCredentialsError("no ADC")

    monkeypatch.setattr(bigquery_writer.bigquery, "Client", fail_client)
    with pytest.raises(MLBigQueryPublishError, match="Application Default Credentials"):
        create_bigquery_client()
