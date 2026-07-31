from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd
from google.api_core.exceptions import (
    BadRequest,
    Forbidden,
    GoogleAPICallError,
    NotFound,
)
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery

from src.ml.config import BIGQUERY_LOCATION, GCP_PROJECT_ID, ML_DATASET_ID


class MLBigQueryPublishError(RuntimeError):
    """Falha operacional ao publicar resultados de ML no BigQuery."""


def _field(name: str, field_type: str, mode: str = "REQUIRED") -> bigquery.SchemaField:
    return bigquery.SchemaField(name, field_type, mode=mode)


ML_TABLE_SCHEMAS: dict[str, tuple[bigquery.SchemaField, ...]] = {
    "sales_forecast": (
        _field("run_id", "STRING"),
        _field("generated_at", "TIMESTAMP"),
        _field("product_id", "STRING"),
        _field("forecast_date", "DATE"),
        _field("horizon_day", "INTEGER"),
        _field("horizon_days", "INTEGER"),
        _field("predicted_quantity", "FLOAT"),
        _field("model_name", "STRING"),
        _field("model_version", "STRING"),
        _field("champion_status", "STRING"),
        _field("data_max_date", "DATE"),
        _field("source_project", "STRING"),
        _field("source_dataset", "STRING"),
    ),
    "inventory_risk": (
        _field("run_id", "STRING"),
        _field("generated_at", "TIMESTAMP"),
        _field("product_id", "STRING"),
        _field("product_name", "STRING"),
        _field("category", "STRING"),
        _field("stock_quantity", "FLOAT"),
        _field("minimum_stock", "FLOAT"),
        _field("forecast_demand", "FLOAT"),
        _field("projected_stock", "FLOAT"),
        _field("average_daily_demand", "FLOAT"),
        _field("estimated_coverage_days", "FLOAT", mode="NULLABLE"),
        _field("risk_level", "STRING"),
        _field("model_name", "STRING"),
        _field("model_version", "STRING"),
        _field("horizon_days", "INTEGER"),
    ),
    "model_metrics": (
        _field("run_id", "STRING"),
        _field("model_name", "STRING"),
        _field("model_version", "STRING"),
        _field("champion_status", "STRING"),
        _field("evaluation_period", "STRING"),
        _field("forecast_horizon", "INTEGER"),
        _field("metric_name", "STRING"),
        _field("metric_value", "FLOAT"),
        _field("generated_at", "TIMESTAMP"),
    ),
    "model_registry": (
        _field("model_name", "STRING"),
        _field("model_version", "STRING"),
        _field("registered_at", "TIMESTAMP"),
        _field("status", "STRING"),
        _field("is_champion", "BOOLEAN"),
        _field("promotion_decision", "STRING"),
        _field("rejection_reason", "STRING", mode="NULLABLE"),
        _field("primary_metric", "STRING"),
        _field("primary_metric_value", "FLOAT", mode="NULLABLE"),
        _field("bias", "FLOAT", mode="NULLABLE"),
        _field("training_data_min_date", "DATE"),
        _field("training_data_max_date", "DATE"),
        _field("code_version", "STRING", mode="NULLABLE"),
        _field("metadata_json", "STRING"),
    ),
    "pipeline_runs": (
        _field("run_id", "STRING"),
        _field("started_at", "TIMESTAMP"),
        _field("finished_at", "TIMESTAMP", mode="NULLABLE"),
        _field("status", "STRING"),
        _field("source_data_min_date", "DATE"),
        _field("source_data_max_date", "DATE"),
        _field("products_processed", "INTEGER"),
        _field("forecast_rows", "INTEGER"),
        _field("risk_rows", "INTEGER"),
        _field("champion_model", "STRING"),
        _field("champion_version", "STRING"),
        _field("error_message", "STRING", mode="NULLABLE"),
        _field("duration_seconds", "FLOAT", mode="NULLABLE"),
    ),
}

MERGE_KEYS = {
    "sales_forecast": (
        "run_id",
        "product_id",
        "forecast_date",
        "horizon_days",
        "model_version",
    ),
    "inventory_risk": ("run_id", "product_id", "horizon_days"),
    "model_metrics": (
        "run_id",
        "model_name",
        "model_version",
        "evaluation_period",
        "forecast_horizon",
        "metric_name",
    ),
    "model_registry": ("model_name", "model_version"),
    "pipeline_runs": ("run_id",),
}


def create_bigquery_client(
    project_id: str = GCP_PROJECT_ID,
    location: str = BIGQUERY_LOCATION,
) -> bigquery.Client:
    try:
        return bigquery.Client(project=project_id, location=location)
    except DefaultCredentialsError as error:
        raise MLBigQueryPublishError(
            "Falha de autenticação com Application Default Credentials para publicação ML"
        ) from error


def ensure_ml_dataset(
    client: Any,
    *,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = ML_DATASET_ID,
    location: str = BIGQUERY_LOCATION,
) -> dict[str, Any]:
    full_dataset_id = f"{project_id}.{dataset_id}"
    try:
        dataset = client.get_dataset(full_dataset_id)
        created = False
    except NotFound:
        dataset = bigquery.Dataset(full_dataset_id)
        dataset.location = location
        try:
            dataset = client.create_dataset(dataset)
        except (
            DefaultCredentialsError,
            BadRequest,
            Forbidden,
            GoogleAPICallError,
        ) as error:
            raise MLBigQueryPublishError(
                f"Falha ao criar dataset ML {full_dataset_id}: {error}"
            ) from error
        created = True
    except (DefaultCredentialsError, BadRequest, Forbidden, GoogleAPICallError) as error:
        raise MLBigQueryPublishError(
            f"Falha ao verificar dataset ML {full_dataset_id}: {error}"
        ) from error
    actual_location = str(dataset.location or "")
    if actual_location.lower() != location.lower():
        raise ValueError(
            "Dataset ML em região incompatível: "
            f"esperada={location}, encontrada={actual_location or 'ausente'}"
        )
    return {
        "dataset_id": dataset_id,
        "full_dataset_id": full_dataset_id,
        "location": actual_location,
        "created": created,
    }


def _canonical_type(field_type: str) -> str:
    aliases = {
        "INT64": "INTEGER",
        "FLOAT64": "FLOAT",
        "BOOL": "BOOLEAN",
    }
    normalized = field_type.upper()
    return aliases.get(normalized, normalized)


def _schema_signature(
    schema: tuple[bigquery.SchemaField, ...] | list[Any],
) -> list[tuple[str, str, str]]:
    return [
        (field.name, _canonical_type(field.field_type), field.mode.upper())
        for field in schema
    ]


def ensure_ml_tables(
    client: Any,
    *,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = ML_DATASET_ID,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for table_name, schema in ML_TABLE_SCHEMAS.items():
        full_table_id = f"{project_id}.{dataset_id}.{table_name}"
        try:
            table = client.get_table(full_table_id)
            created = False
        except NotFound:
            table = bigquery.Table(full_table_id, schema=list(schema))
            try:
                table = client.create_table(table)
            except (
                DefaultCredentialsError,
                BadRequest,
                Forbidden,
                GoogleAPICallError,
            ) as error:
                raise MLBigQueryPublishError(
                    f"Falha ao criar tabela ML {full_table_id}: {error}"
                ) from error
            created = True
        except (DefaultCredentialsError, BadRequest, Forbidden, GoogleAPICallError) as error:
            raise MLBigQueryPublishError(
                f"Falha ao verificar tabela ML {full_table_id}: {error}"
            ) from error
        expected_signature = _schema_signature(schema)
        actual_signature = _schema_signature(table.schema)
        if actual_signature != expected_signature:
            raise ValueError(
                f"Schema incompatível em {full_table_id}: "
                f"esperado={expected_signature}, encontrado={actual_signature}"
            )
        expiration_cleared = False
        if getattr(table, "expires", None) is not None:
            table.expires = None
            try:
                table = client.update_table(table, ["expires"])
            except (
                DefaultCredentialsError,
                BadRequest,
                Forbidden,
                GoogleAPICallError,
            ) as error:
                raise MLBigQueryPublishError(
                    f"Falha ao remover expiração histórica de {full_table_id}: {error}"
                ) from error
            expiration_cleared = True
        reports[table_name] = {
            "full_table_id": full_table_id,
            "created": created,
            "schema": expected_signature,
            "expiration_cleared": expiration_cleared,
        }
    return reports


def _staging_suffix(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", value)
    return normalized[:80] or "run"


def _merge_sql(
    target_table_id: str,
    staging_table_id: str,
    table_name: str,
) -> str:
    schema = ML_TABLE_SCHEMAS[table_name]
    columns = [field.name for field in schema]
    keys = MERGE_KEYS[table_name]
    condition = " AND ".join(f"T.{key} = S.{key}" for key in keys)
    immutable = {"registered_at"} if table_name == "model_registry" else set()
    update_columns = [
        column for column in columns if column not in keys and column not in immutable
    ]
    update_clause = ",\n            ".join(
        f"T.{column} = S.{column}" for column in update_columns
    )
    insert_columns = ", ".join(columns)
    insert_values = ", ".join(f"S.{column}" for column in columns)
    return f"""
        MERGE `{target_table_id}` AS T
        USING `{staging_table_id}` AS S
        ON {condition}
        WHEN MATCHED THEN
          UPDATE SET
            {update_clause}
        WHEN NOT MATCHED THEN
          INSERT ({insert_columns})
          VALUES ({insert_values})
    """


def merge_dataframe(
    client: Any,
    *,
    table_name: str,
    dataframe: pd.DataFrame,
    run_id: str,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = ML_DATASET_ID,
    location: str = BIGQUERY_LOCATION,
) -> dict[str, Any]:
    if table_name not in ML_TABLE_SCHEMAS:
        raise ValueError(f"Tabela ML desconhecida: {table_name}")
    if dataframe.empty:
        raise ValueError(f"{table_name} não pode ser publicada vazia")
    schema = ML_TABLE_SCHEMAS[table_name]
    columns = [field.name for field in schema]
    missing = sorted(set(columns).difference(dataframe.columns))
    if missing:
        raise ValueError(f"{table_name} sem colunas: {', '.join(missing)}")
    keys = list(MERGE_KEYS[table_name])
    if dataframe.duplicated(keys).any():
        raise ValueError(f"{table_name} possui chave de MERGE duplicada")

    target_table_id = f"{project_id}.{dataset_id}.{table_name}"
    staging_table_id = (
        f"{project_id}.{dataset_id}._staging_{table_name}_{_staging_suffix(run_id)}"
    )
    job_config = bigquery.LoadJobConfig(
        schema=list(schema),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    load_job: Any | None = None
    merge_job: Any | None = None
    try:
        load_job = client.load_table_from_dataframe(
            dataframe.loc[:, columns],
            staging_table_id,
            job_config=job_config,
            location=location,
        )
        load_job.result()
        merge_job = client.query(
            _merge_sql(target_table_id, staging_table_id, table_name),
            location=location,
        )
        merge_job.result()
    except DefaultCredentialsError as error:
        raise MLBigQueryPublishError(
            f"Falha de autenticação ao publicar {target_table_id}"
        ) from error
    except (BadRequest, Forbidden, GoogleAPICallError) as error:
        raise MLBigQueryPublishError(
            f"Falha no MERGE de {target_table_id}: {error}"
        ) from error
    finally:
        try:
            client.delete_table(staging_table_id, not_found_ok=True)
        except (
            DefaultCredentialsError,
            BadRequest,
            Forbidden,
            GoogleAPICallError,
        ):
            pass
    return {
        "table_name": table_name,
        "full_table_id": target_table_id,
        "input_rows": int(len(dataframe)),
        "load_job_id": str(getattr(load_job, "job_id", "")),
        "merge_job_id": str(getattr(merge_job, "job_id", "")),
        "merge_keys": keys,
        "write_strategy": "staging_write_truncate_then_target_merge",
    }


def publish_ml_dataframes(
    client: Any,
    *,
    forecasts: pd.DataFrame,
    inventory_risk: pd.DataFrame,
    model_metrics: pd.DataFrame,
    model_registry: pd.DataFrame,
    run_id: str,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = ML_DATASET_ID,
    location: str = BIGQUERY_LOCATION,
) -> dict[str, dict[str, Any]]:
    frames = {
        "sales_forecast": forecasts,
        "inventory_risk": inventory_risk,
        "model_metrics": model_metrics,
        "model_registry": model_registry,
    }
    return {
        table_name: merge_dataframe(
            client,
            table_name=table_name,
            dataframe=dataframe,
            run_id=run_id,
            project_id=project_id,
            dataset_id=dataset_id,
            location=location,
        )
        for table_name, dataframe in frames.items()
    }


def upsert_pipeline_run(
    client: Any,
    pipeline_run: Mapping[str, Any],
    *,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = ML_DATASET_ID,
    location: str = BIGQUERY_LOCATION,
) -> dict[str, Any]:
    dataframe = pd.DataFrame.from_records([dict(pipeline_run)])
    return merge_dataframe(
        client,
        table_name="pipeline_runs",
        dataframe=dataframe,
        run_id=str(pipeline_run["run_id"]),
        project_id=project_id,
        dataset_id=dataset_id,
        location=location,
    )
