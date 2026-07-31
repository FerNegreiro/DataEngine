from __future__ import annotations

import pandas as pd
import pytest

from pipelines.machine_learning.run_ml_pipeline import run_ml_pipeline


def test_pipeline_rejects_unsupported_forecast_horizon() -> None:
    with pytest.raises(ValueError, match="forecast_horizon"):
        run_ml_pipeline(forecast_horizon=21)


def test_pipeline_rejects_truncated_source_before_training(
    ml_products: pd.DataFrame,
) -> None:
    truncated_sales = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "date": pd.Timestamp("2026-07-28"),
                "quantity_sold": 1.0,
                "revenue": 20.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="não cobre o período"):
        run_ml_pipeline(sales=truncated_sales, products=ml_products)
