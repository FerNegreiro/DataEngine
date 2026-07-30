from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validation.validate_silver_data import validate_silver_data


def _input_rows(datasets: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {name: len(dataframe) for name, dataframe in datasets.items()}


def test_valid_silver_data_returns_structured_report(
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    report = validate_silver_data(
        valid_silver_dataframes,
        input_rows=_input_rows(valid_silver_dataframes),
    )

    assert report["is_valid"] is True
    assert report["errors"] == []
    assert report["warnings"] == []
    assert set(report["datasets"]) == {
        "customers",
        "orders",
        "order_items",
        "products",
    }
    assert all(
        dataset_report["input_rows"] == dataset_report["output_rows"]
        for dataset_report in report["datasets"].values()
    )


def test_detects_missing_silver_file(
    tmp_path: Path,
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    sources: dict[str, pd.DataFrame | Path] = dict(valid_silver_dataframes)
    sources["customers"] = tmp_path / "customers.parquet"

    report = validate_silver_data(sources)

    assert report["is_valid"] is False
    assert any("arquivo Silver não encontrado" in error for error in report["errors"])


def test_detects_duplicate_primary_key(
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    datasets = {
        name: dataframe.copy()
        for name, dataframe in valid_silver_dataframes.items()
    }
    datasets["customers"] = pd.concat(
        [datasets["customers"], datasets["customers"].iloc[[0]]],
        ignore_index=True,
    )

    report = validate_silver_data(datasets)

    assert report["is_valid"] is False
    assert any("chave primária duplicada" in error for error in report["errors"])


def test_detects_invalid_foreign_key(
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    datasets = {
        name: dataframe.copy()
        for name, dataframe in valid_silver_dataframes.items()
    }
    datasets["orders"].loc[0, "customer_id"] = "CUST-999999"

    report = validate_silver_data(datasets)

    assert report["is_valid"] is False
    assert any("referências inexistentes" in error for error in report["errors"])


def test_detects_negative_monetary_value(
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    datasets = {
        name: dataframe.copy()
        for name, dataframe in valid_silver_dataframes.items()
    }
    datasets["orders"].loc[0, "shipping_cost"] = -1.0

    report = validate_silver_data(datasets)

    assert report["is_valid"] is False
    assert any("valores monetários negativos" in error for error in report["errors"])


def test_detects_invalid_status_email_state_and_financial_reconciliation(
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    datasets = {
        name: dataframe.copy()
        for name, dataframe in valid_silver_dataframes.items()
    }
    datasets["customers"].loc[0, "email"] = "EMAIL-INVALIDO"
    datasets["customers"].loc[0, "state"] = "XX"
    datasets["orders"].loc[0, "order_status"] = "Desconhecido"
    datasets["orders"].loc[0, "order_total"] += 10

    report = validate_silver_data(datasets)

    assert report["is_valid"] is False
    assert any("email possui formato inválido" in error for error in report["errors"])
    assert any("UFs inválidas" in error for error in report["errors"])
    assert any("valores não permitidos" in error for error in report["errors"])
    assert any("não reconcilia" in error for error in report["errors"])


def test_detects_row_count_change(
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    input_rows = _input_rows(valid_silver_dataframes)
    input_rows["products"] += 1

    report = validate_silver_data(
        valid_silver_dataframes,
        input_rows=input_rows,
    )

    assert report["is_valid"] is False
    assert any("quantidade de linhas mudou" in error for error in report["errors"])
