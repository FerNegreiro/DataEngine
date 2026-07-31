from __future__ import annotations

import pandas as pd

from src.ml.model_selection import (
    CHAMPION_MODEL,
    _inventory_risk_scenarios,
    _metric_rows,
    _run_fold,
    decide_promotion,
)
from src.ml.temporal_split import TemporalFold


def _decision_inputs(
    candidate_wapes: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = ["validation_fold_1", "validation_fold_2", "validation_fold_3"]
    aggregate_records = []
    for split, candidate_wape in zip(splits, candidate_wapes, strict=True):
        aggregate_records.extend(
            [
                {
                    "split": split,
                    "model_name": "challenger",
                    "horizon": 14,
                    "wape": candidate_wape,
                    "bias": 0.0,
                },
                {
                    "split": split,
                    "model_name": CHAMPION_MODEL,
                    "horizon": 14,
                    "wape": 1.0,
                    "bias": 0.0,
                },
            ]
        )
    aggregate_records.extend(
        [
            {
                "split": "final_test",
                "model_name": "challenger",
                "horizon": 14,
                "wape": candidate_wapes[-1],
                "bias": 0.1,
            },
            {
                "split": "final_test",
                "model_name": CHAMPION_MODEL,
                "horizon": 14,
                "wape": 1.0,
                "bias": 0.0,
            },
        ]
    )
    segments = pd.DataFrame(
        [
            {
                "split": "final_test",
                "model_name": model,
                "horizon": 14,
                "demand_pattern": pattern,
                "wape": value,
            }
            for pattern in ("intermittent", "lumpy")
            for model, value in [("challenger", 0.9), (CHAMPION_MODEL, 1.0)]
        ]
    )
    comparisons = pd.DataFrame(
        {
            "split": splits + ["final_test"],
            "product_id": ["P1"] * 4,
            "forecast_date": pd.date_range("2026-04-01", periods=4),
            "model_name": ["challenger"] * 4,
            "predicted_quantity": [1.0] * 4,
        }
    )
    return pd.DataFrame(aggregate_records), segments, comparisons


def test_promotion_rejects_candidate_that_loses_validation_folds() -> None:
    aggregate, segments, comparisons = _decision_inputs([1.2, 1.1, 1.3])
    decision = decide_promotion(aggregate, segments, comparisons, "challenger")
    assert decision["decision"] == "rejected"
    assert decision["validation_fold_wins"] == 0
    assert decision["final_champion"] == CHAMPION_MODEL


def test_promotion_accepts_candidate_only_when_all_gates_pass() -> None:
    aggregate, segments, comparisons = _decision_inputs([0.9, 0.95, 0.92])
    decision = decide_promotion(aggregate, segments, comparisons, "challenger")
    assert decision["decision"] == "promoted"
    assert decision["final_champion"] == "challenger"


def test_fold_evaluation_produces_aggregate_segment_and_product_metrics(
    ml_grid: pd.DataFrame,
) -> None:
    fold = TemporalFold(
        name="small_fold",
        train_start_date=pd.Timestamp("2026-01-01").date(),
        train_end_date=pd.Timestamp("2026-01-31").date(),
        validation_start_date=pd.Timestamp("2026-02-01").date(),
        validation_end_date=pd.Timestamp("2026-02-04").date(),
    )
    output = _run_fold(
        ml_grid,
        fold,
        occurrence_threshold=0.2,
        model_parameters={"max_iter": 3, "min_samples_leaf": 2},
        hurdle_classifier_parameters={"max_iter": 3, "min_samples_leaf": 2},
        hurdle_regressor_parameters={"max_iter": 3, "min_samples_leaf": 2},
        croston_alpha=0.1,
        tsb_beta=0.1,
    )
    aggregate, segments, products = _metric_rows(
        output.comparison, {fold.name: output}, horizons=(2, 4)
    )
    assert aggregate["model_name"].nunique() == 9
    assert set(aggregate["horizon"]) == {2, 4}
    assert not segments.empty
    assert set(products["product_id"]) == {"P1", "P2"}


def test_inventory_comparison_preserves_champion_and_counts_changes(
    ml_products: pd.DataFrame,
) -> None:
    active_products = ml_products.loc[ml_products["is_active"]].copy()
    forecasts = pd.DataFrame(
        [
            {
                "split": "future",
                "product_id": "P1",
                "forecast_date": pd.Timestamp("2026-08-01")
                + pd.Timedelta(days=day - 1),
                "horizon_day": day,
                "predicted_quantity": quantity,
                "model_name": model,
            }
            for model, quantity in [(CHAMPION_MODEL, 0.0), ("challenger", 3.0)]
            for day in range(1, 15)
        ]
    )
    comparison, changes = _inventory_risk_scenarios(
        forecasts,
        active_products,
        final_champion=CHAMPION_MODEL,
        challenger="challenger",
    )
    assert comparison.loc[comparison["is_champion"], "model_name"].eq(CHAMPION_MODEL).all()
    assert changes["challenger"] == 1
