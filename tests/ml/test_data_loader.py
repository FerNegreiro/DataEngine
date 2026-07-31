from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from google.api_core.exceptions import NotFound
from google.auth.exceptions import DefaultCredentialsError

from src.ml.data_loader import (
    MLAuthenticationError,
    MLEmptyDataError,
    MLTableNotFoundError,
    load_analytical_data,
    load_local_analytical_data,
)


class FakeQueryJob:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def result(self) -> list[dict[str, Any]]:
        return self.rows


class FakeBigQueryClient:
    def __init__(
        self,
        sales_rows: list[dict[str, Any]],
        product_rows: list[dict[str, Any]],
        error: Exception | None = None,
    ) -> None:
        self.sales_rows = sales_rows
        self.product_rows = product_rows
        self.error = error
        self.queries: list[str] = []

    def query(self, query: str, **_: Any) -> FakeQueryJob:
        self.queries.append(query)
        if self.error:
            raise self.error
        rows = self.sales_rows if "fct_sales" in query else self.product_rows
        return FakeQueryJob(rows)


def _rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sales = [
        {
            "product_id": "P1",
            "date": pd.Timestamp("2026-01-01"),
            "quantity_sold": 2,
            "revenue": 40,
        }
    ]
    products = [
        {
            "product_id": "P1",
            "category": "cat-a",
            "brand": "brand-a",
            "unit_price": 20,
            "unit_cost": 10,
            "stock_quantity": 30,
            "minimum_stock": 5,
            "is_active": True,
        }
    ]
    return sales, products


def test_load_analytical_data_uses_only_read_queries() -> None:
    sales, products = _rows()
    client = FakeBigQueryClient(sales, products)

    loaded = load_analytical_data(client)

    assert len(loaded.sales) == 1
    assert len(loaded.products) == 1
    assert len(client.queries) == 2
    assert "WHERE is_realized_sale" in client.queries[0]
    assert all("INSERT" not in query.upper() for query in client.queries)


def test_load_analytical_data_translates_authentication_error() -> None:
    client = FakeBigQueryClient([], [], DefaultCredentialsError("no ADC"))
    with pytest.raises(MLAuthenticationError, match="ADC"):
        load_analytical_data(client)


def test_load_analytical_data_translates_missing_table() -> None:
    client = FakeBigQueryClient([], [], NotFound("missing"))
    with pytest.raises(MLTableNotFoundError, match="não encontrada"):
        load_analytical_data(client)


def test_load_analytical_data_rejects_empty_sales() -> None:
    _, products = _rows()
    client = FakeBigQueryClient([], products)
    with pytest.raises(MLEmptyDataError, match="zero linhas"):
        load_analytical_data(client)


def test_load_local_analytical_data_requires_both_files(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError, match="fct_sales.parquet"):
        load_local_analytical_data(tmp_path)


def test_load_local_analytical_data_reads_both_parquets(tmp_path: Any) -> None:
    sales, products = _rows()
    pd.DataFrame(sales).to_parquet(tmp_path / "fct_sales.parquet", index=False)
    pd.DataFrame(products).to_parquet(tmp_path / "dim_products.parquet", index=False)

    loaded = load_local_analytical_data(tmp_path)

    assert loaded.sales["quantity_sold"].tolist() == [2]
    assert loaded.products["is_active"].tolist() == [True]
