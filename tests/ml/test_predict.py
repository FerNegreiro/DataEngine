from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.predict import recursive_forecast


def test_recursive_forecast_feeds_predictions_back_without_future_actuals(
    ml_grid: pd.DataFrame,
) -> None:
    single_product_history = ml_grid.loc[ml_grid["product_id"] == "P1"].copy()

    def next_value(features: pd.DataFrame) -> np.ndarray:
        return features["lag_1"].to_numpy(dtype=float) + 1.0

    last_actual = float(single_product_history["quantity_sold"].iloc[-1])
    forecasts = recursive_forecast(
        single_product_history,
        next_value,
        3,
        active_only=False,
        generated_at=pd.Timestamp("2026-07-31", tz="UTC"),
    )

    assert forecasts["predicted_quantity"].tolist() == [
        last_actual + 1,
        last_actual + 2,
        last_actual + 3,
    ]
    assert forecasts["horizon_day"].tolist() == [1, 2, 3]
    assert forecasts["forecast_date"].min() == pd.Timestamp("2026-02-05")


def test_recursive_forecast_scores_active_products_only_by_default(
    ml_grid: pd.DataFrame,
) -> None:
    forecasts = recursive_forecast(
        ml_grid,
        lambda features: np.zeros(len(features)),
        2,
    )
    assert forecasts["product_id"].unique().tolist() == ["P1"]
    assert len(forecasts) == 2
