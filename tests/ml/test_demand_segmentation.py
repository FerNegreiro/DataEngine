from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.demand_segmentation import (
    classify_demand_pattern,
    demand_statistics,
    segment_product_demand,
)


def test_adi_and_cv_squared_follow_the_documented_contract() -> None:
    statistics = demand_statistics(pd.Series([0, 2, 0, 4, 0, 0], dtype=float))
    assert statistics["non_zero_demand_days"] == 2
    assert statistics["adi"] == 3.0
    assert statistics["cv_squared"] == 1 / 9
    assert statistics["demand_pattern"] == "intermittent"


def test_demand_pattern_quadrants_use_fixed_thresholds() -> None:
    assert classify_demand_pattern(1.0, 0.1) == "smooth"
    assert classify_demand_pattern(2.0, 0.1) == "intermittent"
    assert classify_demand_pattern(1.0, 0.6) == "erratic"
    assert classify_demand_pattern(2.0, 0.6) == "lumpy"


def test_zero_and_single_demand_series_are_deterministic() -> None:
    zero = demand_statistics(pd.Series([0, 0, 0], dtype=float))
    single = demand_statistics(pd.Series([0, 5, 0, 0], dtype=float))
    assert np.isinf(zero["adi"])
    assert zero["cv_squared"] == 0.0
    assert zero["demand_pattern"] == "intermittent"
    assert single["adi"] == 4.0
    assert single["cv_squared"] == 0.0


def test_segmentation_returns_one_row_per_product(ml_grid: pd.DataFrame) -> None:
    segments = segment_product_demand(ml_grid)
    assert len(segments) == 2
    assert set(segments.columns) == {
        "product_id",
        "non_zero_demand_days",
        "adi",
        "cv_squared",
        "demand_pattern",
    }
