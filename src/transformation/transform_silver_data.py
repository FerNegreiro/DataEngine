from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd

from src.extraction.generate_customers import CUSTOMER_COLUMNS
from src.extraction.generate_orders import ORDER_COLUMNS, ORDER_ITEM_COLUMNS
from src.validation.validate_raw_data import PRODUCT_COLUMNS

SILVER_SCHEMAS = {
    "customers": {
        "columns": CUSTOMER_COLUMNS,
        "primary_key": "customer_id",
        "strings": (
            "customer_id",
            "full_name",
            "email",
            "gender",
            "city",
            "state",
            "region",
            "acquisition_channel",
            "customer_segment",
        ),
        "datetimes": ("birth_date", "registration_date"),
        "floats": (),
        "money": (),
        "integers": (),
        "booleans": ("is_active",),
    },
    "orders": {
        "columns": ORDER_COLUMNS,
        "primary_key": "order_id",
        "strings": (
            "order_id",
            "customer_id",
            "order_status",
            "payment_method",
            "sales_channel",
        ),
        "datetimes": ("order_date", "delivery_date"),
        "floats": ("shipping_cost", "discount_amount", "order_total"),
        "money": ("shipping_cost", "discount_amount", "order_total"),
        "integers": (),
        "booleans": (),
    },
    "order_items": {
        "columns": ORDER_ITEM_COLUMNS,
        "primary_key": "order_item_id",
        "strings": ("order_item_id", "order_id", "product_id"),
        "datetimes": (),
        "floats": (
            "unit_price",
            "unit_cost",
            "discount_percentage",
            "line_total",
        ),
        "money": ("unit_price", "unit_cost", "line_total"),
        "integers": ("quantity",),
        "booleans": (),
    },
    "products": {
        "columns": PRODUCT_COLUMNS,
        "primary_key": "product_id",
        "strings": (
            "product_id",
            "product_name",
            "category",
            "brand",
            "supplier",
        ),
        "datetimes": ("created_at",),
        "floats": ("unit_price", "unit_cost"),
        "money": ("unit_price", "unit_cost"),
        "integers": ("stock_quantity", "minimum_stock"),
        "booleans": ("is_active",),
    },
}


def _to_snake_case(column: object) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(column))
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    normalized.columns = [_to_snake_case(column) for column in dataframe.columns]
    if normalized.columns.duplicated().any():
        duplicated = sorted(set(normalized.columns[normalized.columns.duplicated()]))
        raise ValueError(
            "A padronização para snake_case gerou colunas duplicadas: "
            f"{', '.join(duplicated)}"
        )
    return normalized


def _normalize_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def _normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("bool")

    normalized = series.astype("string").str.strip().str.lower()
    converted = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )
    if converted.isna().any():
        raise ValueError(f"Valores booleanos inválidos na coluna {series.name}")
    return converted.astype("bool")


def transform_silver_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    if dataset_name not in SILVER_SCHEMAS:
        raise ValueError(f"Dataset Silver desconhecido: {dataset_name}")

    transformed = _normalize_columns(dataframe)
    schema = SILVER_SCHEMAS[dataset_name]
    expected_columns = tuple(schema["columns"])
    missing_columns = [
        column for column in expected_columns if column not in transformed.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Colunas obrigatórias ausentes em {dataset_name}: "
            f"{', '.join(missing_columns)}"
        )

    for column in schema["strings"]:
        transformed[column] = _normalize_text(transformed[column])
    for column in schema["datetimes"]:
        transformed[column] = pd.to_datetime(transformed[column], errors="raise")
    for column in schema["floats"]:
        transformed[column] = pd.to_numeric(
            transformed[column],
            errors="raise",
        ).astype("float64")
    for column in schema["money"]:
        transformed[column] = transformed[column].round(2)
    for column in schema["integers"]:
        transformed[column] = pd.to_numeric(
            transformed[column],
            errors="raise",
        ).astype("int64")
    for column in schema["booleans"]:
        transformed[column] = _normalize_boolean(transformed[column])

    if dataset_name == "customers":
        transformed["email"] = transformed["email"].str.lower()
        transformed["state"] = transformed["state"].str.upper()

    additional_columns = [
        column for column in transformed.columns if column not in expected_columns
    ]
    transformed = transformed.loc[:, [*expected_columns, *additional_columns]]
    return transformed.sort_values(
        str(schema["primary_key"]),
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def transform_silver_data(
    datasets: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    expected_datasets = set(SILVER_SCHEMAS)
    received_datasets = set(datasets)
    missing_datasets = sorted(expected_datasets - received_datasets)
    unexpected_datasets = sorted(received_datasets - expected_datasets)

    if missing_datasets:
        raise ValueError(
            f"Datasets obrigatórios ausentes para a Silver: {', '.join(missing_datasets)}"
        )
    if unexpected_datasets:
        raise ValueError(
            f"Datasets desconhecidos recebidos para a Silver: "
            f"{', '.join(unexpected_datasets)}"
        )

    return {
        dataset_name: transform_silver_dataset(
            datasets[dataset_name],
            dataset_name,
        )
        for dataset_name in SILVER_SCHEMAS
    }
