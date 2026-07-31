from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.machine_learning.run_ml_pipeline import run_ml_pipeline


def test_pipeline_rejects_unsupported_forecast_horizon() -> None:
    with pytest.raises(ValueError, match="forecast_horizon"):
        run_ml_pipeline(forecast_horizon=21)


def test_pipeline_rejects_unknown_experiment_before_loading_sources() -> None:
    with pytest.raises(ValueError, match="iteration_02"):
        run_ml_pipeline(experiment="iteration_99")


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


def test_iteration_02_runs_end_to_end_without_bigquery(
    ml_products: pd.DataFrame,
    tmp_path: Path,
) -> None:
    dates = list(pd.date_range("2023-01-06", "2026-07-28", freq="10D"))
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
            for product_id, quantity in [("P1", 2.0), ("P2", 1.0)]
        ]
    )
    result = run_ml_pipeline(
        sales=sales,
        products=ml_products,
        experiment="iteration_02",
        artifacts_dir=tmp_path,
        model_parameters={"max_iter": 2, "min_samples_leaf": 2},
        hurdle_classifier_parameters={"max_iter": 2, "min_samples_leaf": 2},
        hurdle_regressor_parameters={"max_iter": 2, "min_samples_leaf": 2},
    )

    assert result["experiment"] == "iteration_02"
    assert result["bigquery_write_performed"] is False
    assert result["final_champion"]
    assert Path(result["artifact_paths"]["promotion_decision"]).is_file()
    comparison = json.loads(
        Path(result["artifact_paths"]["model_comparison"]).read_text(encoding="utf-8")
    )
    assert (
        comparison["product_win_count_method"]
        == "tie_inclusive_minimum_valid_product_wape"
    )
