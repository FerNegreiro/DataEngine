from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from google.cloud import bigquery

from src.ml.config import (
    BIGQUERY_LOCATION,
    CHAMPION_MODEL,
    CHAMPION_MODEL_VERSION,
    CHAMPION_STATUS,
    FORECAST_HORIZONS,
    GCP_PROJECT_ID,
    ML_DATASET_ID,
    PRIMARY_FORECAST_HORIZON,
    REJECTED_CHALLENGER_STATUS,
)
from src.ml.inventory_risk import RISK_ORDER
from src.ml.production import FORECAST_COLUMNS, METRIC_COLUMNS, RISK_COLUMNS
from src.ml.registry import REGISTERED_MODELS, validate_official_promotion_decision

ML_TABLE_NAMES = (
    "sales_forecast",
    "inventory_risk",
    "model_metrics",
    "model_registry",
    "pipeline_runs",
)
ALLOWED_METRICS = {"wape", "mae", "rmse", "smape", "mase", "bias"}


class MLPublicationValidationError(ValueError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        details = "; ".join(report.get("errors", [])) or "erro desconhecido"
        super().__init__(f"Validação da publicação ML falhou: {details}")


class MLBigQueryLoadValidationError(ValueError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        details = "; ".join(report.get("errors", [])) or "erro desconhecido"
        super().__init__(f"Validação pós-carga ML falhou: {details}")


def _missing_columns(dataframe: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    return sorted(set(required).difference(dataframe.columns))


def _essential_nulls(dataframe: pd.DataFrame, columns: list[str]) -> int:
    available = [column for column in columns if column in dataframe]
    return int(dataframe[available].isna().any(axis=1).sum()) if available else 0


def _single_value(dataframe: pd.DataFrame, column: str) -> Any | None:
    if column not in dataframe or dataframe.empty:
        return None
    values = dataframe[column].drop_duplicates()
    return values.iloc[0] if len(values) == 1 else None


def validate_ml_publication(
    *,
    manifest: Mapping[str, Any],
    forecasts: pd.DataFrame,
    inventory_risk: pd.DataFrame,
    model_metrics: pd.DataFrame,
    model_registry: pd.DataFrame,
    pipeline_run: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    for name, dataframe, columns in (
        ("sales_forecast", forecasts, FORECAST_COLUMNS),
        ("inventory_risk", inventory_risk, RISK_COLUMNS),
        ("model_metrics", model_metrics, METRIC_COLUMNS),
    ):
        missing = _missing_columns(dataframe, columns)
        if missing:
            errors.append(f"{name}: colunas ausentes: {', '.join(missing)}")
        if dataframe.empty:
            errors.append(f"{name}: não pode estar vazia")

    expected_registry_columns = {
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
    }
    missing_registry = sorted(expected_registry_columns.difference(model_registry.columns))
    if missing_registry:
        errors.append(
            "model_registry: colunas ausentes: " + ", ".join(missing_registry)
        )
    if model_registry.empty:
        errors.append("model_registry: não pode estar vazio")

    run_id = str(manifest.get("run_id", "")).strip()
    if not run_id:
        errors.append("manifest: run_id ausente")
    if manifest.get("champion_model") != CHAMPION_MODEL:
        errors.append(f"manifest: champion deve ser {CHAMPION_MODEL}")
    if manifest.get("champion_version") != CHAMPION_MODEL_VERSION:
        errors.append("manifest: champion_version incompatível")
    if tuple(manifest.get("forecast_horizons", [])) != FORECAST_HORIZONS:
        errors.append("manifest: horizontes devem ser 7, 14 e 30")
    try:
        validate_official_promotion_decision(
            dict(manifest.get("promotion_decision", {})),
            dict(manifest.get("model_comparison", {})),
        )
    except ValueError as error:
        errors.append(f"manifest: {error}")

    if not forecasts.empty and not _missing_columns(forecasts, FORECAST_COLUMNS):
        essential = [column for column in FORECAST_COLUMNS if column]
        null_rows = _essential_nulls(forecasts, essential)
        if null_rows:
            errors.append(f"sales_forecast: {null_rows} linha(s) com nulo essencial")
        if _single_value(forecasts, "run_id") != run_id:
            errors.append("sales_forecast: run_id divergente do manifest")
        if set(forecasts["model_name"].astype(str)) != {CHAMPION_MODEL}:
            errors.append("sales_forecast: somente o champion pode ser publicado")
        if set(forecasts["model_version"].astype(str)) != {CHAMPION_MODEL_VERSION}:
            errors.append("sales_forecast: model_version do champion é incompatível")
        if set(forecasts["champion_status"].astype(str)) != {CHAMPION_STATUS}:
            errors.append("sales_forecast: champion_status deve ser champion")
        numeric_predictions = pd.to_numeric(
            forecasts["predicted_quantity"], errors="coerce"
        )
        if (
            numeric_predictions.isna().any()
            or not np.isfinite(numeric_predictions).all()
            or numeric_predictions.lt(0).any()
        ):
            errors.append("sales_forecast: predicted_quantity deve ser finita e não negativa")
        duplicate_key = [
            "run_id",
            "generated_at",
            "product_id",
            "forecast_date",
            "horizon_days",
            "model_version",
        ]
        if forecasts.duplicated(duplicate_key).any():
            errors.append("sales_forecast: duplicidade na granularidade oficial")
        horizons = tuple(sorted(forecasts["horizon_days"].unique()))
        if horizons != FORECAST_HORIZONS:
            errors.append("sales_forecast: horizontes 7, 14 e 30 são obrigatórios")
        for (product_id, horizon), group in forecasts.groupby(
            ["product_id", "horizon_days"], observed=True
        ):
            expected_days = set(range(1, int(horizon) + 1))
            actual_days = set(group["horizon_day"].astype(int))
            if len(group) != int(horizon) or actual_days != expected_days:
                errors.append(
                    "sales_forecast: horizonte incompleto para "
                    f"product_id={product_id}, horizon_days={horizon}"
                )
                break
        maximum_data_date = pd.Timestamp(manifest.get("data_max_date"))
        forecast_dates = pd.to_datetime(forecasts["forecast_date"], errors="coerce")
        if forecast_dates.isna().any() or forecast_dates.le(maximum_data_date).any():
            errors.append("sales_forecast: forecast_date deve ser posterior a data_max_date")

    if not inventory_risk.empty and not _missing_columns(inventory_risk, RISK_COLUMNS):
        essential = [
            column for column in RISK_COLUMNS if column != "estimated_coverage_days"
        ]
        null_rows = _essential_nulls(inventory_risk, essential)
        if null_rows:
            errors.append(f"inventory_risk: {null_rows} linha(s) com nulo essencial")
        if _single_value(inventory_risk, "run_id") != run_id:
            errors.append("inventory_risk: run_id divergente do manifest")
        if set(inventory_risk["horizon_days"].astype(int)) != {
            PRIMARY_FORECAST_HORIZON
        }:
            errors.append("inventory_risk: deve usar somente o horizonte principal de 14 dias")
        if not set(inventory_risk["risk_level"].astype(str)).issubset(set(RISK_ORDER)):
            errors.append("inventory_risk: risk_level inválido")
        if inventory_risk.duplicated(["run_id", "product_id", "horizon_days"]).any():
            errors.append("inventory_risk: duplicidade por run, produto e horizonte")
        for column in (
            "stock_quantity",
            "minimum_stock",
            "forecast_demand",
            "average_daily_demand",
        ):
            values = pd.to_numeric(inventory_risk[column], errors="coerce")
            if values.isna().any() or not np.isfinite(values).all() or values.lt(0).any():
                errors.append(f"inventory_risk: {column} deve ser finita e não negativa")
        projected = inventory_risk["stock_quantity"] - inventory_risk["forecast_demand"]
        if not np.allclose(projected, inventory_risk["projected_stock"], equal_nan=False):
            errors.append("inventory_risk: projected_stock incoerente")
        average = inventory_risk["forecast_demand"] / PRIMARY_FORECAST_HORIZON
        if not np.allclose(
            average, inventory_risk["average_daily_demand"], equal_nan=False
        ):
            errors.append("inventory_risk: average_daily_demand incoerente")
        positive = inventory_risk["forecast_demand"].gt(0)
        expected_coverage = (
            inventory_risk.loc[positive, "stock_quantity"]
            / inventory_risk.loc[positive, "average_daily_demand"]
        )
        if not np.allclose(
            expected_coverage,
            inventory_risk.loc[positive, "estimated_coverage_days"].astype(float),
        ):
            errors.append("inventory_risk: estimated_coverage_days incoerente")
        if inventory_risk.loc[~positive, "estimated_coverage_days"].notna().any():
            errors.append("inventory_risk: cobertura deve ser nula quando a demanda é zero")

    if not model_metrics.empty and not _missing_columns(model_metrics, METRIC_COLUMNS):
        essential = list(METRIC_COLUMNS)
        null_rows = _essential_nulls(model_metrics, essential)
        if null_rows:
            errors.append(f"model_metrics: {null_rows} linha(s) com nulo essencial")
        if _single_value(model_metrics, "run_id") != run_id:
            errors.append("model_metrics: run_id divergente do manifest")
        values = pd.to_numeric(model_metrics["metric_value"], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            errors.append("model_metrics: metric_value deve ser finito")
        if not set(model_metrics["metric_name"].astype(str)).issubset(ALLOWED_METRICS):
            errors.append("model_metrics: metric_name inválido")
        if model_metrics.duplicated(
            [
                "run_id",
                "model_name",
                "model_version",
                "evaluation_period",
                "forecast_horizon",
                "metric_name",
            ]
        ).any():
            errors.append("model_metrics: chave composta duplicada")
        official = model_metrics.loc[model_metrics["model_name"].eq(CHAMPION_MODEL)]
        if official.empty or set(official["champion_status"]) != {CHAMPION_STATUS}:
            errors.append("model_metrics: métricas oficiais do champion ausentes")
        challengers = model_metrics.loc[model_metrics["model_name"].ne(CHAMPION_MODEL)]
        if not challengers.empty and set(challengers["champion_status"]) != {
            REJECTED_CHALLENGER_STATUS
        }:
            errors.append("model_metrics: challengers devem estar identificados como rejeitados")

    if not model_registry.empty and not missing_registry:
        if model_registry.duplicated(["model_name", "model_version"]).any():
            errors.append("model_registry: model_name + model_version deve ser único")
        if set(model_registry["model_name"].astype(str)) != set(REGISTERED_MODELS):
            errors.append("model_registry: conjunto de modelos avaliados incompleto")
        champions = model_registry.loc[model_registry["is_champion"].astype(bool)]
        if len(champions) != 1 or champions.iloc[0]["model_name"] != CHAMPION_MODEL:
            errors.append("model_registry: deve existir exatamente um champion oficial")
        for payload in model_registry["metadata_json"]:
            try:
                json.loads(str(payload))
            except (TypeError, json.JSONDecodeError):
                errors.append("model_registry: metadata_json inválido")
                break

    required_run_fields = {
        "run_id",
        "started_at",
        "status",
        "source_data_min_date",
        "source_data_max_date",
        "products_processed",
        "forecast_rows",
        "risk_rows",
        "champion_model",
        "champion_version",
    }
    missing_run_fields = sorted(required_run_fields.difference(pipeline_run))
    if missing_run_fields:
        errors.append("pipeline_run: campos ausentes: " + ", ".join(missing_run_fields))
    else:
        if pipeline_run["run_id"] != run_id:
            errors.append("pipeline_run: run_id divergente")
        if pipeline_run["status"] != "running":
            errors.append("pipeline_run: pacote local deve iniciar com status running")
        if int(pipeline_run["forecast_rows"]) != len(forecasts):
            errors.append("pipeline_run: forecast_rows divergente")
        if int(pipeline_run["risk_rows"]) != len(inventory_risk):
            errors.append("pipeline_run: risk_rows divergente")
        if int(pipeline_run["products_processed"]) != forecasts["product_id"].nunique():
            errors.append("pipeline_run: products_processed divergente")

    count_expectations = {
        "forecast_rows": len(forecasts),
        "risk_rows": len(inventory_risk),
        "metric_rows": len(model_metrics),
        "registry_rows": len(model_registry),
    }
    for field, actual in count_expectations.items():
        if int(manifest.get(field, -1)) != actual:
            errors.append(f"manifest: {field} divergente")

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "run_id": run_id,
        "counts": count_expectations,
        "products_processed": int(forecasts["product_id"].nunique()) if not forecasts.empty else 0,
        "forecast_horizons": sorted(forecasts["horizon_days"].unique().tolist())
        if "horizon_days" in forecasts
        else [],
    }


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "items"):
        return dict(row.items())
    raise TypeError("Resultado de validação BigQuery incompatível")


def validate_ml_bigquery_load(
    client: Any,
    *,
    run_id: str,
    expected_forecast_rows: int,
    expected_risk_rows: int,
    expected_metric_rows: int,
    expected_products: int,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = ML_DATASET_ID,
    location: str = BIGQUERY_LOCATION,
) -> dict[str, Any]:
    errors: list[str] = []
    full_dataset_id = f"{project_id}.{dataset_id}"
    dataset = client.get_dataset(full_dataset_id)
    actual_location = str(dataset.location or "")
    if actual_location.lower() != location.lower():
        errors.append(
            f"Dataset em região incorreta: esperada={location}, encontrada={actual_location}"
        )
    table_ids: dict[str, str] = {}
    for table_name in ML_TABLE_NAMES:
        full_table_id = f"{full_dataset_id}.{table_name}"
        client.get_table(full_table_id)
        table_ids[table_name] = full_table_id

    query = f"""
        WITH forecast AS (
            SELECT * FROM `{table_ids['sales_forecast']}` WHERE run_id = @run_id
        ), risk AS (
            SELECT * FROM `{table_ids['inventory_risk']}` WHERE run_id = @run_id
        ), metrics AS (
            SELECT * FROM `{table_ids['model_metrics']}` WHERE run_id = @run_id
        ), run AS (
            SELECT * FROM `{table_ids['pipeline_runs']}` WHERE run_id = @run_id
        )
        SELECT
            (SELECT COUNT(*) FROM forecast) AS forecast_rows,
            (SELECT COUNT(*) - COUNT(DISTINCT TO_JSON_STRING(STRUCT(
                product_id, forecast_date, horizon_days, model_version
            ))) FROM forecast) AS forecast_duplicate_rows,
            (SELECT ARRAY_AGG(DISTINCT horizon_days ORDER BY horizon_days) FROM forecast)
                AS forecast_horizons,
            (SELECT COUNT(DISTINCT product_id) FROM forecast) AS forecast_products,
            (SELECT COUNTIF(
                model_name != @champion_model OR champion_status != @champion_status
            ) FROM forecast) AS invalid_forecast_model_rows,
            (SELECT COUNT(*) FROM risk) AS risk_rows,
            (SELECT COUNT(*) - COUNT(DISTINCT TO_JSON_STRING(STRUCT(
                product_id, horizon_days
            ))) FROM risk) AS risk_duplicate_rows,
            (SELECT COUNTIF(risk_level NOT IN UNNEST(@risk_levels)) FROM risk)
                AS invalid_risk_rows,
            (SELECT COUNT(*) FROM metrics) AS metric_rows,
            (SELECT COUNT(*) FROM `{table_ids['model_registry']}` WHERE is_champion)
                AS active_champion_rows,
            (SELECT COUNT(*) FROM run) AS pipeline_run_rows,
            (SELECT ANY_VALUE(status) FROM run) AS pipeline_status
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            bigquery.ScalarQueryParameter(
                "champion_model", "STRING", CHAMPION_MODEL
            ),
            bigquery.ScalarQueryParameter(
                "champion_status", "STRING", CHAMPION_STATUS
            ),
            bigquery.ArrayQueryParameter("risk_levels", "STRING", list(RISK_ORDER)),
        ]
    )
    rows = list(client.query(query, job_config=job_config, location=location).result())
    if len(rows) != 1:
        errors.append("Consulta pós-carga não retornou exatamente uma linha")
        result: dict[str, Any] = {}
    else:
        result = _row_to_dict(rows[0])

    expectations = {
        "forecast_rows": expected_forecast_rows,
        "risk_rows": expected_risk_rows,
        "metric_rows": expected_metric_rows,
        "forecast_products": expected_products,
        "forecast_duplicate_rows": 0,
        "invalid_forecast_model_rows": 0,
        "risk_duplicate_rows": 0,
        "invalid_risk_rows": 0,
        "active_champion_rows": 1,
        "pipeline_run_rows": 1,
    }
    for field, expected in expectations.items():
        if int(result.get(field, -1)) != expected:
            errors.append(
                f"{field} divergente: esperado={expected}, encontrado={result.get(field)}"
            )
    horizons = tuple(result.get("forecast_horizons") or [])
    if horizons != FORECAST_HORIZONS:
        errors.append(
            f"forecast_horizons divergente: esperado={FORECAST_HORIZONS}, encontrado={horizons}"
        )
    if result.get("pipeline_status") != "success":
        errors.append(
            f"pipeline_runs sem status success: encontrado={result.get('pipeline_status')}"
        )
    return {
        "is_valid": not errors,
        "errors": errors,
        "run_id": run_id,
        "location": actual_location,
        "tables": table_ids,
        "counts": result,
    }
