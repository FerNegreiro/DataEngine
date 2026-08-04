from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from google.cloud import bigquery

from src.ml.config import (
    BIGQUERY_LOCATION,
    CHAMPION_MODEL,
    CHAMPION_MODEL_VERSION,
    FORECAST_HORIZONS,
    GCP_PROJECT_ID,
)

SOURCE_DATASET = "dataengine"
DBT_DATASET = "dataengine_dbt"
ML_DATASET = "dataengine_ml"
SOURCE_TABLES = ("customers", "orders", "order_items", "products")
DBT_RELATIONS = (
    "stg_customers",
    "stg_orders",
    "stg_order_items",
    "stg_products",
    "int_order_items_enriched",
    "int_orders_enriched",
    "dim_customers",
    "dim_products",
    "fct_sales",
    "mart_daily_sales",
    "mart_product_performance",
    "mart_customer_metrics",
)
ML_TABLES = (
    "sales_forecast",
    "inventory_risk",
    "model_metrics",
    "model_registry",
    "pipeline_runs",
)


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "items"):
        return dict(row.items())
    raise TypeError("Resultado de consulta BigQuery incompatível")


def _query_rows(
    client: Any,
    query: str,
    *,
    job_config: bigquery.QueryJobConfig | None = None,
) -> list[dict[str, Any]]:
    result = client.query(
        query,
        job_config=job_config,
        location=BIGQUERY_LOCATION,
    ).result()
    return [_row_to_dict(row) for row in result]


def _error_context(label: str, error: Exception) -> str:
    return f"{label}: {type(error).__name__}"


def validate_final_outputs(
    *,
    bigquery_client: Any | None = None,
    project_id: str = GCP_PROJECT_ID,
) -> dict[str, Any]:
    """Confirma somente por leitura os resultados finais do pipeline."""
    errors: list[str] = []
    client = bigquery_client or bigquery.Client(
        project=project_id,
        location=BIGQUERY_LOCATION,
    )
    dataset_reports: dict[str, dict[str, Any]] = {}
    for dataset_id in (SOURCE_DATASET, DBT_DATASET, ML_DATASET):
        full_dataset_id = f"{project_id}.{dataset_id}"
        report = {"exists": False, "location": None}
        dataset_reports[dataset_id] = report
        try:
            dataset = client.get_dataset(full_dataset_id)
            report["exists"] = True
            report["location"] = str(dataset.location or "")
            if report["location"].lower() != BIGQUERY_LOCATION.lower():
                errors.append(
                    f"Dataset {dataset_id} fora de {BIGQUERY_LOCATION}: "
                    f"{report['location'] or 'localização ausente'}"
                )
        except Exception as error:
            errors.append(_error_context(f"Dataset {dataset_id} indisponível", error))

    relation_groups = {
        SOURCE_DATASET: SOURCE_TABLES,
        DBT_DATASET: DBT_RELATIONS,
        ML_DATASET: ML_TABLES,
    }
    relation_reports: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_id, relation_names in relation_groups.items():
        relation_reports[dataset_id] = {}
        for relation_name in relation_names:
            full_table_id = f"{project_id}.{dataset_id}.{relation_name}"
            relation_report = {"exists": False, "rows": None}
            relation_reports[dataset_id][relation_name] = relation_report
            try:
                table = client.get_table(full_table_id)
                relation_report["exists"] = True
                num_rows = getattr(table, "num_rows", None)
                relation_report["rows"] = int(num_rows) if num_rows is not None else None
                if dataset_id == SOURCE_DATASET and int(num_rows or 0) <= 0:
                    errors.append(f"Tabela de origem vazia: {full_table_id}")
            except Exception as error:
                errors.append(_error_context(f"Relação {full_table_id} indisponível", error))

    latest_query = f"""
        SELECT
            run_id,
            products_processed,
            forecast_rows,
            risk_rows,
            champion_model,
            champion_version
        FROM `{project_id}.{ML_DATASET}.pipeline_runs`
        WHERE status = 'success'
        ORDER BY COALESCE(finished_at, started_at) DESC, started_at DESC, run_id DESC
        LIMIT 1
    """
    try:
        latest_rows = _query_rows(client, latest_query)
    except Exception as error:
        errors.append(
            _error_context(
                "Consulta da execução bem-sucedida mais recente falhou",
                error,
            )
        )
        latest_rows = []

    latest_run = latest_rows[0] if len(latest_rows) == 1 else {}
    if len(latest_rows) != 1:
        errors.append("Nenhuma execução bem-sucedida única foi encontrada")

    counts: dict[str, Any] = {}
    run_id = str(latest_run.get("run_id", "")).strip()
    if run_id:
        validation_query = f"""
            WITH forecast AS (
                SELECT * FROM `{project_id}.{ML_DATASET}.sales_forecast`
                WHERE run_id = @run_id
            ), risk AS (
                SELECT * FROM `{project_id}.{ML_DATASET}.inventory_risk`
                WHERE run_id = @run_id
            ), metrics AS (
                SELECT * FROM `{project_id}.{ML_DATASET}.model_metrics`
                WHERE run_id = @run_id
            ), run AS (
                SELECT * FROM `{project_id}.{ML_DATASET}.pipeline_runs`
                WHERE run_id = @run_id AND status = 'success'
            )
            SELECT
                (SELECT COUNT(*) FROM forecast) AS forecast_rows,
                (SELECT COUNT(DISTINCT product_id) FROM forecast) AS forecast_products,
                (SELECT ARRAY_AGG(DISTINCT horizon_days ORDER BY horizon_days) FROM forecast)
                    AS forecast_horizons,
                (SELECT COUNT(*) - COUNT(DISTINCT TO_JSON_STRING(STRUCT(
                    product_id, forecast_date, horizon_days, model_version
                ))) FROM forecast) AS forecast_duplicate_rows,
                (SELECT COUNT(*) FROM (
                    SELECT product_id, horizon_days
                    FROM forecast
                    GROUP BY product_id, horizon_days
                    HAVING COUNT(*) != horizon_days
                        OR COUNT(DISTINCT horizon_day) != horizon_days
                        OR MIN(horizon_day) != 1
                        OR MAX(horizon_day) != horizon_days
                )) AS incomplete_forecast_groups,
                (SELECT COUNTIF(
                    model_name != @champion_model
                    OR model_version != @champion_version
                    OR champion_status != 'champion'
                ) FROM forecast) AS invalid_forecast_model_rows,
                (SELECT COUNT(*) FROM risk) AS risk_rows,
                (SELECT COUNT(*) - COUNT(DISTINCT TO_JSON_STRING(STRUCT(
                    product_id, horizon_days
                ))) FROM risk) AS risk_duplicate_rows,
                (SELECT COUNT(*) FROM metrics) AS metric_rows,
                (SELECT COUNT(*) - COUNT(DISTINCT TO_JSON_STRING(STRUCT(
                    model_name, model_version, evaluation_period,
                    forecast_horizon, metric_name
                ))) FROM metrics) AS metric_duplicate_rows,
                (SELECT COUNT(*) FROM `{project_id}.{ML_DATASET}.model_registry`
                    WHERE is_champion) AS active_champion_rows,
                (SELECT COUNT(*) FROM `{project_id}.{ML_DATASET}.model_registry`
                    WHERE is_champion
                    AND model_name = @champion_model
                    AND model_version = @champion_version) AS official_champion_rows,
                (SELECT COUNT(*) - COUNT(DISTINCT TO_JSON_STRING(STRUCT(
                    model_name, model_version
                ))) FROM `{project_id}.{ML_DATASET}.model_registry`)
                    AS registry_duplicate_rows,
                (SELECT COUNT(*) FROM run) AS pipeline_run_rows
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
                bigquery.ScalarQueryParameter(
                    "champion_model", "STRING", CHAMPION_MODEL
                ),
                bigquery.ScalarQueryParameter(
                    "champion_version", "STRING", CHAMPION_MODEL_VERSION
                ),
            ]
        )
        try:
            validation_rows = _query_rows(
                client,
                validation_query,
                job_config=job_config,
            )
        except Exception as error:
            errors.append(_error_context("Consulta de coerência final falhou", error))
            validation_rows = []
        if len(validation_rows) == 1:
            counts = validation_rows[0]
        else:
            errors.append("Consulta de coerência final não retornou exatamente uma linha")

    if latest_run:
        if latest_run.get("champion_model") != CHAMPION_MODEL:
            errors.append("A execução mais recente não preserva moving_average_28 como champion")
        if latest_run.get("champion_version") != CHAMPION_MODEL_VERSION:
            errors.append("A versão do champion da execução mais recente é inválida")

    if counts and latest_run:
        expected_products = int(latest_run.get("products_processed") or 0)
        expected_forecasts = int(latest_run.get("forecast_rows") or 0)
        expected_risks = int(latest_run.get("risk_rows") or 0)
        expected_formula = expected_products * sum(FORECAST_HORIZONS)
        expectations = {
            "forecast_rows": expected_forecasts,
            "forecast_products": expected_products,
            "risk_rows": expected_risks,
            "forecast_duplicate_rows": 0,
            "incomplete_forecast_groups": 0,
            "invalid_forecast_model_rows": 0,
            "risk_duplicate_rows": 0,
            "metric_duplicate_rows": 0,
            "active_champion_rows": 1,
            "official_champion_rows": 1,
            "registry_duplicate_rows": 0,
            "pipeline_run_rows": 1,
        }
        for field, expected in expectations.items():
            if int(counts.get(field, -1)) != expected:
                errors.append(
                    f"{field} divergente: esperado={expected}, "
                    f"encontrado={counts.get(field)}"
                )
        if int(counts.get("forecast_rows", -1)) != expected_formula:
            errors.append(
                "forecast_rows incoerente com produtos e horizontes: "
                f"esperado={expected_formula}, encontrado={counts.get('forecast_rows')}"
            )
        if expected_risks != expected_products:
            errors.append(
                "risk_rows deve igualar products_processed: "
                f"{expected_risks} != {expected_products}"
            )
        if int(counts.get("metric_rows", 0)) <= 0:
            errors.append("A execução mais recente não possui métricas")
        if tuple(counts.get("forecast_horizons") or []) != FORECAST_HORIZONS:
            errors.append(
                "Horizontes oficiais divergentes: "
                f"esperado={FORECAST_HORIZONS}, "
                f"encontrado={counts.get('forecast_horizons')}"
            )

    return {
        "is_valid": not errors,
        "errors": errors,
        "project_id": project_id,
        "location": BIGQUERY_LOCATION,
        "datasets": dataset_reports,
        "relations": relation_reports,
        "latest_successful_run": latest_run,
        "counts": counts,
    }


def main() -> int:
    try:
        report = validate_final_outputs()
    except Exception as error:
        report = {
            "is_valid": False,
            "errors": [_error_context("Validação final não pôde iniciar", error)],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
