from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validation.validate_bigquery_load import (
    validate_bigquery_inputs,
    validate_bigquery_load,
)

PARTITION = "year=2026/month=02/day=03"


def _write_silver_files(
    directory: Path,
    datasets: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for dataset_name, dataframe in datasets.items():
        path = directory / f"{dataset_name}.parquet"
        dataframe.to_parquet(path, index=False, compression="snappy")
        files[dataset_name] = path
    return files


def _silver_report(
    files: dict[str, Path],
    *,
    partition: str = PARTITION,
    is_valid: bool = True,
) -> dict[str, object]:
    return {
        "partition": partition,
        "validation": {
            "is_valid": is_valid,
            "errors": [] if is_valid else ["erro Silver simulado"],
            "warnings": [],
        },
        "files": [
            {
                "dataset": dataset_name,
                "local_path": str(path),
                "object_key": (
                    f"silver/{dataset_name}/{partition}/{dataset_name}.parquet"
                ),
            }
            for dataset_name, path in files.items()
        ],
    }


def test_validates_bigquery_inputs_before_loading(
    tmp_path: Path,
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    files = _write_silver_files(tmp_path, valid_silver_dataframes)

    report = validate_bigquery_inputs(
        files,
        expected_partition=PARTITION,
        silver_report=_silver_report(files),
    )

    assert report["is_valid"] is True
    assert report["errors"] == []
    assert all(
        dataset_report["exists"]
        and dataset_report["file_size_bytes"] > 0
        and dataset_report["row_count"] > 0
        and dataset_report["schema_field_count"] > 0
        for dataset_report in report["datasets"].values()
    )


def test_rejects_missing_silver_file_before_loading(
    tmp_path: Path,
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    files = _write_silver_files(tmp_path, valid_silver_dataframes)
    files["customers"] = tmp_path / "missing" / "customers.parquet"

    report = validate_bigquery_inputs(
        files,
        expected_partition=PARTITION,
        silver_report=_silver_report(files),
    )

    assert report["is_valid"] is False
    assert any("arquivo Silver não encontrado" in error for error in report["errors"])


def test_rejects_empty_silver_file(
    tmp_path: Path,
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    files = _write_silver_files(tmp_path, valid_silver_dataframes)
    files["orders"].write_bytes(b"")

    report = validate_bigquery_inputs(
        files,
        expected_partition=PARTITION,
        silver_report=_silver_report(files),
    )

    assert report["is_valid"] is False
    assert any("arquivo Silver está vazio" in error for error in report["errors"])


def test_rejects_invalid_silver_report_or_partition(
    tmp_path: Path,
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    files = _write_silver_files(tmp_path, valid_silver_dataframes)

    report = validate_bigquery_inputs(
        files,
        expected_partition=PARTITION,
        silver_report=_silver_report(
            files,
            partition="year=2026/month=02/day=04",
            is_valid=False,
        ),
    )

    assert report["is_valid"] is False
    assert any("Partição Silver divergente" in error for error in report["errors"])
    assert any("relatório Silver" in error for error in report["errors"])


def test_validates_loaded_bigquery_tables() -> None:
    expected_rows = {
        "customers": 2,
        "orders": 2,
        "order_items": 2,
        "products": 2,
    }
    tables = {
        dataset_name: {
            "full_table_id": f"project.dataset.{dataset_name}",
            "input_rows": rows,
            "loaded_rows": rows,
            "success": True,
        }
        for dataset_name, rows in expected_rows.items()
    }

    report = validate_bigquery_load(tables, expected_rows)

    assert report["is_valid"] is True
    assert report["errors"] == []
    assert set(report["tables"]) == set(expected_rows)


def test_detects_bigquery_row_count_divergence() -> None:
    expected_rows = {
        "customers": 2,
        "orders": 2,
        "order_items": 2,
        "products": 2,
    }
    tables = {
        dataset_name: {
            "full_table_id": f"project.dataset.{dataset_name}",
            "input_rows": rows,
            "loaded_rows": 1 if dataset_name == "orders" else rows,
            "success": dataset_name != "orders",
        }
        for dataset_name, rows in expected_rows.items()
    }

    report = validate_bigquery_load(tables, expected_rows)

    assert report["is_valid"] is False
    assert any("divergência de linhas" in error for error in report["errors"])


def test_detects_missing_or_empty_bigquery_table() -> None:
    expected_rows = {
        "customers": 2,
        "orders": 2,
        "order_items": 2,
        "products": 2,
    }
    tables = {
        "customers": {
            "full_table_id": "project.dataset.customers",
            "input_rows": 2,
            "loaded_rows": 0,
            "success": False,
        }
    }

    report = validate_bigquery_load(tables, expected_rows)

    assert report["is_valid"] is False
    assert any("tabela BigQuery está vazia" in error for error in report["errors"])
    assert any("Tabelas obrigatórias ausentes" in error for error in report["errors"])
