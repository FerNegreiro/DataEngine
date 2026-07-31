from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ml.feature_engineering import build_product_day_grid
from src.ml.production import ProductionBundle, build_production_bundle
from src.ml.registry import REGISTERED_MODELS


@pytest.fixture
def ml_products() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "P1",
                "product_name": "Product 1",
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
                "product_name": "Product 2",
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


@pytest.fixture
def approved_experiment_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "iteration_02"
    directory.mkdir()
    records = [
        {
            "split": split,
            "model_name": model_name,
            "horizon": horizon,
            "wape": 1.0,
            "mae": 0.5,
            "rmse": 0.75,
            "smape": 1.2,
            "mase": 2.0,
            "bias": -0.1 if model_name == "croston_sba" else 0.05,
        }
        for split in (
            "validation_fold_1",
            "validation_fold_2",
            "validation_fold_3",
            "final_test",
        )
        for model_name in REGISTERED_MODELS
        for horizon in (7, 14, 30)
    ]
    (directory / "metrics.json").write_text(
        json.dumps({"aggregate_metrics": records}), encoding="utf-8"
    )
    promotion = {
        "decision": "rejected",
        "challenger": "croston_sba",
        "final_champion": "moving_average_28",
        "reason": "Critérios não satisfeitos: acceptable_aggregate_bias",
    }
    comparison = {"final_champion": "moving_average_28"}
    (directory / "promotion_decision.json").write_text(
        json.dumps(promotion), encoding="utf-8"
    )
    (directory / "model_comparison.json").write_text(
        json.dumps(comparison), encoding="utf-8"
    )
    return directory


@pytest.fixture
def production_bundle(
    ml_products: pd.DataFrame,
    approved_experiment_dir: Path,
    tmp_path: Path,
) -> ProductionBundle:
    dates = list(pd.date_range("2023-01-06", "2026-07-28", freq="30D"))
    if dates[-1] != pd.Timestamp("2026-07-28"):
        dates.append(pd.Timestamp("2026-07-28"))
    sales = pd.DataFrame(
        [
            {
                "product_id": product_id,
                "date": current_date,
                "quantity_sold": quantity,
                "revenue": quantity * 20,
            }
            for current_date in dates
            for product_id, quantity in (("P1", 2.0), ("P2", 1.0))
        ]
    )
    grid = build_product_day_grid(
        sales,
        ml_products,
        start_date="2023-01-06",
        end_date="2026-07-28",
    )
    return build_production_bundle(
        grid,
        ml_products,
        experiment_dir=approved_experiment_dir,
        generated_at=pd.Timestamp("2026-07-31T12:00:00Z"),
        repository=tmp_path,
    )
