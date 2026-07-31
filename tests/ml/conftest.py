from __future__ import annotations

import pandas as pd
import pytest

from src.ml.feature_engineering import build_product_day_grid


@pytest.fixture
def ml_products() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "P1",
                "category": "cat-a",
                "brand": "brand-a",
                "unit_price": 20.0,
                "unit_cost": 10.0,
                "stock_quantity": 30,
                "minimum_stock": 5,
                "is_active": True,
            },
            {
                "product_id": "P2",
                "category": "cat-b",
                "brand": "brand-b",
                "unit_price": 15.0,
                "unit_cost": 7.0,
                "stock_quantity": 50,
                "minimum_stock": 8,
                "is_active": False,
            },
        ]
    )


@pytest.fixture
def ml_sales() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    dates = pd.date_range("2026-01-01", periods=35, freq="D")
    for index, current_date in enumerate(dates):
        if index % 7 == 0:
            records.append(
                {
                    "product_id": "P1",
                    "date": current_date,
                    "quantity_sold": 2.0,
                    "revenue": 40.0,
                }
            )
        if index % 5 == 0:
            records.append(
                {
                    "product_id": "P2",
                    "date": current_date,
                    "quantity_sold": 1.0,
                    "revenue": 15.0,
                }
            )
    return pd.DataFrame.from_records(records)


@pytest.fixture
def ml_grid(ml_sales: pd.DataFrame, ml_products: pd.DataFrame) -> pd.DataFrame:
    return build_product_day_grid(
        ml_sales,
        ml_products,
        start_date="2026-01-01",
        end_date="2026-02-04",
    )
