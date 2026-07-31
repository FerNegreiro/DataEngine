from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from google.api_core.exceptions import (
    BadRequest,
    Forbidden,
    GoogleAPICallError,
    NotFound,
)
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery

from pipelines.loading.read_from_s3 import (
    DEFAULT_SILVER_STAGING_DIR,
    download_silver_files,
)
from pipelines.loading.upload_to_s3 import (
    BUCKET_NAME,
    build_partition,
    normalize_execution_date,
)
from src.validation.validate_bigquery_load import (
    REQUIRED_DATASETS,
    BigQueryInputValidationError,
    BigQueryLoadValidationError,
    validate_bigquery_inputs,
    validate_bigquery_load,
)

GCP_PROJECT_ID = "dataengine-fernando-2026"
BIGQUERY_DATASET_ID = "dataengine"
BIGQUERY_LOCATION = "southamerica-east1"
WRITE_DISPOSITION = bigquery.WriteDisposition.WRITE_TRUNCATE
SOURCE_FORMAT = bigquery.SourceFormat.PARQUET

logger = logging.getLogger(__name__)


def build_full_table_id(
    dataset_name: str,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = BIGQUERY_DATASET_ID,
) -> str:
    return f"{project_id}.{dataset_id}.{dataset_name}"


def validate_bigquery_configuration(
    project_id: str,
    dataset_id: str,
    location: str,
) -> dict[str, object]:
    errors: list[str] = []
    if not project_id.strip():
        errors.append("project_id não pode ser vazio")
    if not dataset_id.strip():
        errors.append("dataset_id não pode ser vazio")
    if not location.strip():
        errors.append("location não pode ser vazia")

    return {
        "is_valid": not errors,
        "errors": errors,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "location": location,
    }


def ensure_bigquery_dataset(
    client: Any,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = BIGQUERY_DATASET_ID,
    location: str = BIGQUERY_LOCATION,
) -> dict[str, object]:
    full_dataset_id = f"{project_id}.{dataset_id}"
    try:
        dataset = client.get_dataset(full_dataset_id)
    except NotFound:
        dataset = bigquery.Dataset(full_dataset_id)
        dataset.location = location
        try:
            dataset = client.create_dataset(dataset)
        except (BadRequest, Forbidden, GoogleAPICallError) as error:
            raise RuntimeError(
                "Falha ao criar dataset no BigQuery: "
                f"projeto={project_id}, dataset={dataset_id}, "
                f"localização={location}: {error}"
            ) from error
        created = True
    except (BadRequest, Forbidden, GoogleAPICallError) as error:
        raise RuntimeError(
            "Falha ao verificar dataset no BigQuery: "
            f"projeto={project_id}, dataset={dataset_id}, "
            f"localização={location}: {error}"
        ) from error
    else:
        created = False

    actual_location = str(dataset.location or "")
    if actual_location.lower() != location.lower():
        raise ValueError(
            "Dataset BigQuery existente em localização incompatível: "
            f"projeto={project_id}, dataset={dataset_id}, "
            f"esperada={location}, encontrada={actual_location or 'ausente'}"
        )

    return {
        "dataset_id": dataset_id,
        "full_dataset_id": full_dataset_id,
        "location": actual_location,
        "created": created,
    }


def _create_bigquery_client(
    project_id: str,
    location: str,
) -> bigquery.Client:
    try:
        return bigquery.Client(project=project_id, location=location)
    except DefaultCredentialsError as error:
        raise RuntimeError(
            "Falha de autenticação com Application Default Credentials: "
            f"projeto={project_id}, localização={location}"
        ) from error


def _files_from_source_report(
    source_report: Mapping[str, object],
) -> dict[str, Path]:
    files = source_report.get("files", [])
    return {
        str(file_report["dataset"]): Path(str(file_report["local_path"]))
        for file_report in files
        if isinstance(file_report, Mapping)
        and "dataset" in file_report
        and "local_path" in file_report
    }


def _load_table(
    client: Any,
    dataset_name: str,
    local_path: Path,
    input_rows: int,
    project_id: str,
    dataset_id: str,
    location: str,
) -> dict[str, object]:
    full_table_id = build_full_table_id(
        dataset_name,
        project_id=project_id,
        dataset_id=dataset_id,
    )
    job_config = bigquery.LoadJobConfig(
        source_format=SOURCE_FORMAT,
        write_disposition=WRITE_DISPOSITION,
    )

    logger.info(
        "Carregando %s no BigQuery %s com WRITE_TRUNCATE",
        local_path,
        full_table_id,
    )
    try:
        with local_path.open("rb") as parquet_file:
            job = client.load_table_from_file(
                parquet_file,
                full_table_id,
                job_config=job_config,
                location=location,
            )
            job.result()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Falha na carga BigQuery: "
            f"projeto={project_id}, dataset={dataset_id}, "
            f"tabela={dataset_name}, arquivo={local_path}, etapa=abertura do Parquet"
        ) from error
    except DefaultCredentialsError as error:
        raise RuntimeError(
            "Falha de autenticação na carga BigQuery: "
            f"projeto={project_id}, dataset={dataset_id}, "
            f"tabela={dataset_name}, arquivo={local_path}, etapa=job de carga"
        ) from error
    except (BadRequest, Forbidden, GoogleAPICallError) as error:
        raise RuntimeError(
            "Falha no job de carga BigQuery: "
            f"projeto={project_id}, dataset={dataset_id}, "
            f"tabela={dataset_name}, arquivo={local_path}, etapa=job de carga: {error}"
        ) from error

    try:
        table = client.get_table(full_table_id)
    except (NotFound, BadRequest, Forbidden, GoogleAPICallError) as error:
        raise RuntimeError(
            "Falha na validação da tabela carregada no BigQuery: "
            f"projeto={project_id}, dataset={dataset_id}, "
            f"tabela={dataset_name}, arquivo={local_path}, etapa=get_table: {error}"
        ) from error

    loaded_rows = int(table.num_rows)
    logger.info(
        "Carga BigQuery concluída: tabela=%s, job_id=%s, linhas=%s",
        full_table_id,
        job.job_id,
        loaded_rows,
    )
    return {
        "dataset": dataset_name,
        "local_path": str(local_path),
        "table_id": dataset_name,
        "full_table_id": full_table_id,
        "job_id": str(job.job_id),
        "input_rows": input_rows,
        "loaded_rows": loaded_rows,
        "schema_field_count": len(table.schema),
        "write_disposition": WRITE_DISPOSITION,
        "success": loaded_rows == input_rows and loaded_rows > 0,
    }


def load_silver_to_bigquery(
    silver_files: Mapping[str, Path | str] | None = None,
    silver_report: Mapping[str, object] | None = None,
    execution_date: datetime | None = None,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = BIGQUERY_DATASET_ID,
    location: str = BIGQUERY_LOCATION,
    bucket_name: str = BUCKET_NAME,
    staging_dir: Path | str = DEFAULT_SILVER_STAGING_DIR,
    s3_client: Any | None = None,
    bigquery_client: Any | None = None,
) -> dict[str, object]:
    started_at = perf_counter()
    configuration = validate_bigquery_configuration(
        project_id,
        dataset_id,
        location,
    )
    if not configuration["is_valid"]:
        details = "; ".join(configuration["errors"])
        raise ValueError(f"Configuração BigQuery inválida: {details}")

    utc_execution_date = normalize_execution_date(execution_date)
    partition = build_partition(utc_execution_date)

    if silver_files is None:
        source_report = download_silver_files(
            staging_dir=staging_dir,
            bucket_name=bucket_name,
            execution_date=utc_execution_date,
            s3_client=s3_client,
        )
        local_files = _files_from_source_report(source_report)
    else:
        if silver_report is None:
            raise ValueError(
                "silver_report é obrigatório ao reutilizar arquivos Silver locais"
            )
        source_report = silver_report
        local_files = {
            dataset_name: Path(path)
            for dataset_name, path in silver_files.items()
        }

    input_validation = validate_bigquery_inputs(
        local_files,
        expected_partition=partition,
        silver_report=source_report,
    )
    if not input_validation["is_valid"]:
        raise BigQueryInputValidationError(input_validation)

    client = (
        bigquery_client
        if bigquery_client is not None
        else _create_bigquery_client(project_id, location)
    )
    ensure_bigquery_dataset(
        client,
        project_id=project_id,
        dataset_id=dataset_id,
        location=location,
    )

    input_rows = {
        dataset_name: int(
            input_validation["datasets"][dataset_name]["row_count"]
        )
        for dataset_name in REQUIRED_DATASETS
    }
    table_reports: dict[str, dict[str, object]] = {}
    for dataset_name in REQUIRED_DATASETS:
        table_reports[dataset_name] = _load_table(
            client=client,
            dataset_name=dataset_name,
            local_path=local_files[dataset_name],
            input_rows=input_rows[dataset_name],
            project_id=project_id,
            dataset_id=dataset_id,
            location=location,
        )

    output_rows = {
        dataset_name: int(table_report["loaded_rows"])
        for dataset_name, table_report in table_reports.items()
    }
    load_validation = validate_bigquery_load(
        table_reports,
        expected_rows=input_rows,
    )
    if not load_validation["is_valid"]:
        raise BigQueryLoadValidationError(load_validation)

    return {
        "success": True,
        "execution_date": utc_execution_date.isoformat(),
        "partition": partition,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "location": location,
        "write_disposition": WRITE_DISPOSITION,
        "loaded_count": len(table_reports),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "validation": load_validation,
        "tables": table_reports,
        "duration_seconds": round(perf_counter() - started_at, 6),
    }


def _parse_execution_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use uma data ISO 8601, por exemplo 2026-07-30T00:00:00+00:00"
        ) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Carrega no BigQuery a partição Silver correspondente à data UTC informada"
        )
    )
    parser.add_argument(
        "--execution-date",
        type=_parse_execution_date,
        default=None,
        help=(
            "Data ISO 8601 da partição Silver. "
            "Quando omitida, usa a data UTC atual; não procura a última partição."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        report = load_silver_to_bigquery(
            execution_date=arguments.execution_date,
        )
    except Exception as error:
        logger.error("Carga da camada Silver no BigQuery falhou: %s", error)
        return 1

    print("Carga da camada Silver no BigQuery concluída")
    print(f"- Projeto: {report['project_id']}")
    print(f"- Dataset: {report['dataset_id']}")
    print(f"- Partição: {report['partition']}")
    for dataset_name, rows in report["output_rows"].items():
        print(f"- {dataset_name}: {rows} linha(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
