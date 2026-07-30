from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.transformation.transform_silver_data import (
    SILVER_SCHEMAS,
    transform_silver_data,
    transform_silver_dataset,
)


@pytest.mark.parametrize("dataset_name", list(SILVER_SCHEMAS))
def test_transforms_each_dataset_and_preserves_rows_and_primary_keys(
    bronze_dataframes: dict[str, pd.DataFrame],
    dataset_name: str,
) -> None:
    source = bronze_dataframes[dataset_name]
    primary_key = str(SILVER_SCHEMAS[dataset_name]["primary_key"])

    transformed = transform_silver_dataset(source, dataset_name)

    assert len(transformed) == len(source)
    assert set(transformed[primary_key]) == set(source[primary_key].str.strip())
    assert transformed[primary_key].is_monotonic_increasing


def test_standardizes_columns_texts_and_customer_fields(
    bronze_dataframes: dict[str, pd.DataFrame],
) -> None:
    customers = bronze_dataframes["customers"].rename(
        columns={"full_name": "Full Name"}
    )

    transformed = transform_silver_dataset(customers, "customers")

    assert "full_name" in transformed
    assert transformed.loc[0, "full_name"] == "Ana Silva"
    assert transformed.loc[0, "email"] == "ana@example.com"
    assert transformed.loc[0, "state"] == "SP"


def test_normalizes_dates_money_quantities_and_booleans(
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    customers = valid_silver_dataframes["customers"]
    orders = valid_silver_dataframes["orders"]
    order_items = valid_silver_dataframes["order_items"]
    products = valid_silver_dataframes["products"]

    assert pd.api.types.is_datetime64_any_dtype(customers["birth_date"])
    assert pd.api.types.is_datetime64_any_dtype(orders["order_date"])
    assert pd.api.types.is_datetime64_any_dtype(products["created_at"])
    assert orders["order_total"].dtype == "float64"
    assert order_items["line_total"].dtype == "float64"
    assert order_items["quantity"].dtype == "int64"
    assert products["stock_quantity"].dtype == "int64"
    assert customers["is_active"].dtype == "bool"
    assert products["is_active"].dtype == "bool"


def test_preserves_foreign_key_relationships(
    valid_silver_dataframes: dict[str, pd.DataFrame],
) -> None:
    customers = valid_silver_dataframes["customers"]
    orders = valid_silver_dataframes["orders"]
    order_items = valid_silver_dataframes["order_items"]
    products = valid_silver_dataframes["products"]

    assert set(orders["customer_id"]).issubset(set(customers["customer_id"]))
    assert set(order_items["order_id"]).issubset(set(orders["order_id"]))
    assert set(order_items["product_id"]).issubset(set(products["product_id"]))


def test_transformation_is_deterministic(
    bronze_dataframes: dict[str, pd.DataFrame],
) -> None:
    first = transform_silver_data(bronze_dataframes)
    second = transform_silver_data(bronze_dataframes)

    for dataset_name in SILVER_SCHEMAS:
        assert_frame_equal(first[dataset_name], second[dataset_name])
