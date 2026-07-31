from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.intermittent_demand import (
    croston_level,
    forecast_intermittent_grid,
    intermittent_level,
    tsb_level,
)


def test_croston_classic_and_sba_are_deterministic() -> None:
    demand = [0, 0, 2, 0, 0]
    classic = croston_level(demand, alpha=0.1, variant="classic")
    sba = croston_level(demand, alpha=0.1, variant="sba")
    assert classic == 2 / 3
    assert sba == classic * 0.95


def test_tsb_updates_occurrence_probability_without_negative_forecast() -> None:
    forecast = tsb_level([0, 2, 0, 0, 4, 0], alpha=0.2, beta=0.3)
    assert forecast >= 0
    assert np.isfinite(forecast)


def test_intermittent_methods_handle_all_zero_and_single_demand() -> None:
    for method in ("croston", "croston_sba", "tsb"):
        assert intermittent_level([0, 0, 0], method) == 0
        assert intermittent_level([0, 5, 0], method) >= 0


def test_intermittent_grid_forecast_has_full_nonnegative_horizon(
    ml_grid: pd.DataFrame,
) -> None:
    forecast = forecast_intermittent_grid(
        ml_grid,
        horizon=3,
        method="croston_sba",
        active_only=False,
        generated_at=pd.Timestamp("2026-07-31", tz="UTC"),
    )
    assert len(forecast) == 6
    assert forecast["predicted_quantity"].ge(0).all()
    assert forecast.groupby("product_id")["horizon_day"].nunique().eq(3).all()
