from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.machine_learning import run_ml_pipeline as pipeline_module
from pipelines.machine_learning.run_ml_pipeline import _parse_args, run_ml_pipeline


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


def test_cli_keeps_local_mode_and_exposes_explicit_publication_mode() -> None:
    assert _parse_args([]).publish_bigquery is False
    assert _parse_args(["--publish-bigquery"]).publish_bigquery is True
    with pytest.raises(SystemExit):
        _parse_args(["--publish-bigquery", "--experiment", "iteration_02"])


def test_publish_mode_builds_only_official_champion_and_calls_publisher(
    ml_products: pd.DataFrame,
    approved_experiment_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def fake_publish(bundle: object, **_: object) -> dict[str, object]:
        assert bundle.forecasts["model_name"].unique().tolist() == [
            "moving_average_28"
        ]
        return {
            "run_id": bundle.manifest["run_id"],
            "champion_model": "moving_average_28",
            "champion_version": "1.0.0",
        }

    monkeypatch.setattr(pipeline_module, "publish_ml_results", fake_publish)
    result = run_ml_pipeline(
        sales=sales,
        products=ml_products,
        publish_bigquery=True,
        production_artifacts_dir=tmp_path / "production",
        approved_experiment_dir=approved_experiment_dir,
    )
    assert result["bigquery_write_performed"] is True
    assert result["forecast_horizons"] == [7, 14, 30]
    assert result["forecast_rows"] == 51
    assert result["risk_rows"] == 1
    assert result["registry_rows"] == 5
