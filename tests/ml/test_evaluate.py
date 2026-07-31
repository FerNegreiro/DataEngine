from __future__ import annotations

import pandas as pd

from src.ml.config import MODEL_NAME
from src.ml.evaluate import (
    calculate_forecast_metrics,
    evaluate_temporal_folds,
    primary_baseline_comparison,
    seasonal_naive_scale,
)
from src.ml.temporal_split import TemporalFold


def test_forecast_metrics_handle_zero_zero_and_bias() -> None:
    metrics = calculate_forecast_metrics(
        [0, 2, 0, 2],
        [0, 1, 1, 2],
        mase_denominator=0.5,
    )

    assert metrics["wape"] == 0.5
    assert metrics["mae"] == 0.5
    assert metrics["rmse"] == 2**0.5 / 2
    assert metrics["smape"] == (0 + 2 / 3 + 2 + 0) / 4
    assert metrics["bias"] == 0.0
    assert metrics["mase"] == 1.0


def test_mase_is_not_applicable_when_training_scale_is_zero() -> None:
    training = pd.DataFrame(
        {
            "product_id": ["P1"] * 14,
            "date": pd.date_range("2026-01-01", periods=14),
            "quantity_sold": [0.0] * 14,
        }
    )
    assert seasonal_naive_scale(training) is None
    assert calculate_forecast_metrics([0], [0])["mase"] is None


def test_primary_comparison_counts_validation_wins() -> None:
    records = []
    for split, candidate, baseline in [
        ("validation_fold_1", 0.8, 1.0),
        ("validation_fold_2", 1.1, 1.0),
        ("validation_fold_3", 0.9, 1.0),
        ("final_test", 0.7, 0.8),
    ]:
        records.extend(
            [
                {"split": split, "model_name": MODEL_NAME, "horizon": 14, "wape": candidate},
                {
                    "split": split,
                    "model_name": "moving_average_28",
                    "horizon": 14,
                    "wape": baseline,
                },
            ]
        )
    comparison = primary_baseline_comparison(pd.DataFrame(records))
    assert comparison["validation_fold_wins"] == 2
    assert comparison["candidate_passes_two_of_three"] is True


def test_evaluate_temporal_fold_scores_candidate_and_baselines(
    ml_grid: pd.DataFrame,
) -> None:
    fold = TemporalFold(
        name="small_fold",
        train_start_date=pd.Timestamp("2026-01-01").date(),
        train_end_date=pd.Timestamp("2026-01-31").date(),
        validation_start_date=pd.Timestamp("2026-02-01").date(),
        validation_end_date=pd.Timestamp("2026-02-04").date(),
    )
    metrics = evaluate_temporal_folds(
        ml_grid,
        [fold],
        model_parameters={"max_iter": 3, "min_samples_leaf": 2},
        horizons=(2, 4),
    )

    assert set(metrics["model_name"]) == {
        MODEL_NAME,
        "last_observation",
        "seasonal_lag_7",
        "moving_average_28",
    }
    assert set(metrics["horizon"]) == {2, 4}
    assert len(metrics) == 8
