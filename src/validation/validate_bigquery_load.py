from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pyarrow.lib import ArrowException
from pyarrow.parquet import ParquetFile

from src.transformation.transform_silver_data import SILVER_SCHEMAS

REQUIRED_DATASETS = tuple(SILVER_SCHEMAS)


class BigQueryInputValidationError(ValueError):
    def __init__(self, validation_report: dict[str, object]) -> None:
        self.validation_report = validation_report
        errors = validation_report["errors"]
        details = "; ".join(errors) if errors else "erro desconhecido"
        super().__init__(f"Validação pré-carga do BigQuery falhou: {details}")


class BigQueryLoadValidationError(ValueError):
    def __init__(self, validation_report: dict[str, object]) -> None:
        self.validation_report = validation_report
        errors = validation_report["errors"]
        details = "; ".join(errors) if errors else "erro desconhecido"
        super().__init__(f"Validação pós-carga do BigQuery falhou: {details}")


def _append_error(
    errors: list[str],
    dataset_errors: list[str],
    dataset_name: str,
    message: str,
) -> None:
    detail = f"{dataset_name}: {message}"
    if detail not in errors:
        errors.append(detail)
    if message not in dataset_errors:
        dataset_errors.append(message)


def _source_files_by_dataset(
    silver_report: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    files = silver_report.get("files", [])
    return {
        str(file_report["dataset"]): file_report
        for file_report in files
        if isinstance(file_report, Mapping) and "dataset" in file_report
    }


def validate_bigquery_inputs(
    silver_files: Mapping[str, Path | str],
    expected_partition: str,
    silver_report: Mapping[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    dataset_reports: dict[str, dict[str, object]] = {}
    source_files = _source_files_by_dataset(silver_report)

    report_partition = str(silver_report.get("partition", ""))
    if report_partition != expected_partition:
        errors.append(
            "Partição Silver divergente: "
            f"esperada={expected_partition}, recebida={report_partition or 'ausente'}"
        )

    silver_validation = silver_report.get("validation")
    if isinstance(silver_validation, Mapping) and not silver_validation.get(
        "is_valid",
        False,
    ):
        errors.append("O relatório Silver fornecido está marcado como inválido")

    missing_datasets = sorted(set(REQUIRED_DATASETS) - set(silver_files))
    unexpected_datasets = sorted(set(silver_files) - set(REQUIRED_DATASETS))
    if missing_datasets:
        errors.append(
            "Arquivos Silver obrigatórios não informados: "
            f"{', '.join(missing_datasets)}"
        )
    if unexpected_datasets:
        warnings.append(
            "Datasets adicionais ignorados na carga BigQuery: "
            f"{', '.join(unexpected_datasets)}"
        )

    for dataset_name in REQUIRED_DATASETS:
        dataset_report: dict[str, object] = {
            "exists": False,
            "local_path": None,
            "file_size_bytes": 0,
            "row_count": 0,
            "schema_field_count": 0,
            "partition": report_partition,
            "errors": [],
            "warnings": [],
        }
        dataset_reports[dataset_name] = dataset_report
        if dataset_name not in silver_files:
            _append_error(
                errors,
                dataset_report["errors"],
                dataset_name,
                "arquivo Silver não informado",
            )
            continue

        local_path = Path(silver_files[dataset_name])
        dataset_report["local_path"] = str(local_path)
        if local_path.name != f"{dataset_name}.parquet":
            _append_error(
                errors,
                dataset_report["errors"],
                dataset_name,
                f"arquivo não corresponde à tabela esperada: {local_path.name}",
            )
        if not local_path.is_file():
            _append_error(
                errors,
                dataset_report["errors"],
                dataset_name,
                f"arquivo Silver não encontrado: {local_path}",
            )
            continue

        dataset_report["exists"] = True
        file_size = local_path.stat().st_size
        dataset_report["file_size_bytes"] = file_size
        if file_size <= 0:
            _append_error(
                errors,
                dataset_report["errors"],
                dataset_name,
                f"arquivo Silver está vazio: {local_path}",
            )
            continue

        try:
            parquet_file = ParquetFile(local_path)
        except (OSError, ValueError, ArrowException) as error:
            _append_error(
                errors,
                dataset_report["errors"],
                dataset_name,
                f"falha ao ler Parquet Silver em {local_path}: {error}",
            )
            continue

        row_count = parquet_file.metadata.num_rows
        schema_field_count = len(parquet_file.schema_arrow)
        dataset_report["row_count"] = row_count
        dataset_report["schema_field_count"] = schema_field_count
        if row_count <= 0:
            _append_error(
                errors,
                dataset_report["errors"],
                dataset_name,
                "arquivo Parquet Silver não possui linhas",
            )

        expected_object_key = (
            f"silver/{dataset_name}/{expected_partition}/{dataset_name}.parquet"
        )
        source_file = source_files.get(dataset_name)
        if source_file is None:
            _append_error(
                errors,
                dataset_report["errors"],
                dataset_name,
                "metadados da partição Silver não foram informados",
            )
            continue

        object_key = str(source_file.get("object_key", ""))
        dataset_report["object_key"] = object_key
        if object_key != expected_object_key:
            _append_error(
                errors,
                dataset_report["errors"],
                dataset_name,
                "chave Silver não corresponde à partição esperada: "
                f"{object_key or 'ausente'}",
            )

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "partition": expected_partition,
        "datasets": dataset_reports,
    }


def validate_bigquery_load(
    tables: Mapping[str, Mapping[str, object]],
    expected_rows: Mapping[str, int],
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    table_reports: dict[str, dict[str, object]] = {}

    missing_tables = sorted(set(REQUIRED_DATASETS) - set(tables))
    if missing_tables:
        errors.append(
            "Tabelas obrigatórias ausentes no BigQuery: "
            f"{', '.join(missing_tables)}"
        )

    for dataset_name in REQUIRED_DATASETS:
        source_report = tables.get(dataset_name)
        table_report = {
            "exists": source_report is not None,
            "input_rows": expected_rows.get(dataset_name, 0),
            "loaded_rows": 0,
            "table_id": dataset_name,
            "full_table_id": None,
            "errors": [],
            "warnings": [],
        }
        table_reports[dataset_name] = table_report
        if source_report is None:
            _append_error(
                errors,
                table_report["errors"],
                dataset_name,
                "tabela não foi carregada ou não existe",
            )
            continue

        table_report["loaded_rows"] = int(source_report.get("loaded_rows", 0))
        table_report["full_table_id"] = source_report.get("full_table_id")
        input_count = int(source_report.get("input_rows", 0))
        loaded_count = int(source_report.get("loaded_rows", 0))
        table_report["input_rows"] = input_count

        if not source_report.get("success", False):
            _append_error(
                errors,
                table_report["errors"],
                dataset_name,
                "carga não foi marcada como bem-sucedida",
            )
        if loaded_count <= 0:
            _append_error(
                errors,
                table_report["errors"],
                dataset_name,
                "tabela BigQuery está vazia",
            )
        if input_count != loaded_count:
            _append_error(
                errors,
                table_report["errors"],
                dataset_name,
                f"divergência de linhas: entrada={input_count}, BigQuery={loaded_count}",
            )

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "tables": table_reports,
    }
