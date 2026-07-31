from __future__ import annotations

import pandas as pd

from src.ml.inventory_risk import classify_inventory_risk


def test_inventory_risk_assigns_all_deterministic_classes() -> None:
    definitions = {
        "critical": (5, 1, 10),
        "high_risk": (20, 15, 10),
        "attention": (50, 5, 30),
        "adequate": (60, 5, 25),
        "potential_overstock": (100, 5, 10),
    }
    products = pd.DataFrame(
        [
            {
                "product_id": product_id,
                "stock_quantity": stock,
                "minimum_stock": minimum,
                "is_active": True,
            }
            for product_id, (stock, minimum, _) in definitions.items()
        ]
    )
    forecasts = pd.DataFrame(
        [
            {
                "product_id": product_id,
                "forecast_date": pd.Timestamp("2026-08-01") + pd.Timedelta(days=day - 1),
                "horizon_day": day,
                "predicted_quantity": demand / 14,
            }
            for product_id, (_, _, demand) in definitions.items()
            for day in range(1, 15)
        ]
    )

    risk = classify_inventory_risk(forecasts, products).set_index("product_id")

    assert {product_id: str(risk.loc[product_id, "risk_class"]) for product_id in definitions} == {
        product_id: product_id for product_id in definitions
    }


def test_inventory_risk_excludes_inactive_products() -> None:
    products = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "stock_quantity": 10,
                "minimum_stock": 2,
                "is_active": False,
            }
        ]
    )
    forecasts = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "forecast_date": pd.Timestamp("2026-08-01") + pd.Timedelta(days=day - 1),
                "horizon_day": day,
                "predicted_quantity": 0,
            }
            for day in range(1, 15)
        ]
    )
    assert classify_inventory_risk(forecasts, products).empty
