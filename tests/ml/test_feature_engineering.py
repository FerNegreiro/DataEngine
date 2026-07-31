from __future__ import annotations

import pandas as pd

from src.ml.feature_engineering import add_temporal_features, build_product_day_grid


def test_product_day_grid_includes_zero_sales_days(
    ml_sales: pd.DataFrame,
    ml_products: pd.DataFrame,
) -> None:
    grid = build_product_day_grid(
        ml_sales,
        ml_products,
        start_date="2026-01-01",
        end_date="2026-01-10",
    )

    assert len(grid) == 20
    assert not grid.duplicated(["product_id", "date"]).any()
    assert grid["quantity_sold"].eq(0).any()
    assert grid.loc[grid["quantity_sold"].eq(0), "revenue"].eq(0).all()
    assert grid.groupby("product_id")["category"].nunique().eq(1).all()


def test_temporal_features_are_sorted_and_causal(ml_grid: pd.DataFrame) -> None:
    featured = add_temporal_features(ml_grid.sample(frac=1.0, random_state=42))
    target_row = featured.loc[
        (featured["product_id"] == "P1")
        & (featured["date"] == pd.Timestamp("2026-01-08"))
    ].iloc[0]

    assert target_row["lag_7"] == 2.0
    assert target_row["sales_last_7_days"] == 2.0
    assert featured.equals(featured.sort_values(["product_id", "date"]).reset_index(drop=True))


def test_current_target_does_not_change_current_rolling_features(ml_grid: pd.DataFrame) -> None:
    original = add_temporal_features(ml_grid)
    changed_grid = ml_grid.copy()
    mask = (changed_grid["product_id"] == "P1") & (
        changed_grid["date"] == pd.Timestamp("2026-01-15")
    )
    changed_grid.loc[mask, "quantity_sold"] = 999
    changed = add_temporal_features(changed_grid)
    feature_columns = [
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_28",
        "sales_last_14_days",
    ]

    pd.testing.assert_series_equal(
        original.loc[mask, feature_columns].iloc[0],
        changed.loc[mask, feature_columns].iloc[0],
    )


def test_intermittent_features_use_only_prior_observations(ml_grid: pd.DataFrame) -> None:
    original = add_temporal_features(ml_grid)
    changed_grid = ml_grid.copy()
    mask = (changed_grid["product_id"] == "P1") & (
        changed_grid["date"] == pd.Timestamp("2026-01-15")
    )
    changed_grid.loc[mask, "quantity_sold"] = 500
    changed = add_temporal_features(changed_grid)
    columns = [
        "days_since_last_sale",
        "sale_days_last_7",
        "sale_days_last_14",
        "sale_days_last_28",
        "sale_days_last_30",
        "historical_sale_probability",
        "mean_positive_demand",
        "causal_adi",
        "current_zero_streak",
    ]
    pd.testing.assert_series_equal(
        original.loc[mask, columns].iloc[0],
        changed.loc[mask, columns].iloc[0],
    )


def test_intermittent_feature_values_have_causal_meaning(ml_grid: pd.DataFrame) -> None:
    featured = add_temporal_features(ml_grid)
    row = featured.loc[
        (featured["product_id"] == "P1")
        & (featured["date"] == pd.Timestamp("2026-01-03"))
    ].iloc[0]
    assert row["days_since_last_sale"] == 2
    assert row["current_zero_streak"] == 1
    assert row["historical_sale_probability"] == 0.5
    assert row["mean_positive_demand"] == 2
    assert row["causal_adi"] == 2
