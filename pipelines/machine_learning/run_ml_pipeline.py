from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from src.ml.artifacts import save_ml_artifacts
from src.ml.config import (
    ARTIFACTS_DIR,
    BIGQUERY_LOCATION,
    DBT_DATASET_ID,
    DEFAULT_MODEL_PARAMETERS,
    EXPECTED_DATA_START_DATE,
    FINAL_TEST_END_DATE,
    FORECAST_HORIZONS,
    GCP_PROJECT_ID,
    ML_STAGING_DIR,
    MODEL_FEATURE_COLUMNS,
    MODEL_NAME,
    MODEL_VERSION,
    PRIMARY_FORECAST_HORIZON,
)
from src.ml.data_loader import AnalyticalData, load_analytical_data, load_local_analytical_data
from src.ml.evaluate import (
    evaluate_temporal_folds,
    primary_baseline_comparison,
)
from src.ml.feature_engineering import add_temporal_features, build_product_day_grid
from src.ml.inventory_risk import classify_inventory_risk
from src.ml.predict import model_predictor, recursive_forecast
from src.ml.temporal_split import (
    build_expanding_window_folds,
    build_final_test_fold,
    validate_required_date_range,
)
from src.ml.train import train_forecasting_model


def _metric_records(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    sanitized = metrics.astype(object).where(pd.notna(metrics), None)
    return sanitized.to_dict(orient="records")


def _load_inputs(
    *,
    sales: pd.DataFrame | None,
    products: pd.DataFrame | None,
    skip_bigquery: bool,
    staging_dir: Path | str,
    bigquery_client: Any | None,
) -> tuple[AnalyticalData, str]:
    if (sales is None) != (products is None):
        raise ValueError("sales e products devem ser informados juntos")
    if sales is not None and products is not None:
        return AnalyticalData(sales=sales.copy(), products=products.copy()), "injected"
    if skip_bigquery:
        return load_local_analytical_data(staging_dir), "local_parquet"
    return load_analytical_data(bigquery_client), "bigquery_read_only"


def run_ml_pipeline(
    *,
    sales: pd.DataFrame | None = None,
    products: pd.DataFrame | None = None,
    skip_bigquery: bool = False,
    forecast_horizon: int = PRIMARY_FORECAST_HORIZON,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
    staging_dir: Path | str = ML_STAGING_DIR,
    bigquery_client: Any | None = None,
    model_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if forecast_horizon not in FORECAST_HORIZONS:
        raise ValueError(f"forecast_horizon deve ser um de {FORECAST_HORIZONS}")
    analytical_data, source_mode = _load_inputs(
        sales=sales,
        products=products,
        skip_bigquery=skip_bigquery,
        staging_dir=staging_dir,
        bigquery_client=bigquery_client,
    )
    source_dates = pd.to_datetime(analytical_data.sales["date"], errors="raise").dt.date
    if source_dates.min() > EXPECTED_DATA_START_DATE or source_dates.max() < FINAL_TEST_END_DATE:
        raise ValueError(
            "A fonte de vendas não cobre o período produtivo obrigatório: "
            f"fonte={source_dates.min()}..{source_dates.max()}, "
            f"obrigatório={EXPECTED_DATA_START_DATE}..{FINAL_TEST_END_DATE}"
        )
    grid = build_product_day_grid(
        analytical_data.sales,
        analytical_data.products,
        start_date=EXPECTED_DATA_START_DATE,
        end_date=FINAL_TEST_END_DATE,
    )
    validate_required_date_range(
        grid,
        required_start=EXPECTED_DATA_START_DATE,
        required_end=FINAL_TEST_END_DATE,
    )

    validation_folds = build_expanding_window_folds(grid)
    final_test_fold = build_final_test_fold(grid)
    metrics_frame = evaluate_temporal_folds(
        grid,
        [*validation_folds, final_test_fold],
        model_parameters=model_parameters,
    )
    comparison = primary_baseline_comparison(metrics_frame)

    featured_grid = add_temporal_features(grid)
    final_model = train_forecasting_model(featured_grid, model_parameters)
    generation_horizon = max(forecast_horizon, PRIMARY_FORECAST_HORIZON)
    all_forecasts = recursive_forecast(
        grid,
        model_predictor(final_model),
        generation_horizon,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        active_only=True,
    )
    inventory_risk = classify_inventory_risk(
        all_forecasts,
        analytical_data.products,
        horizon=PRIMARY_FORECAST_HORIZON,
    )
    forecasts = all_forecasts.loc[
        all_forecasts["horizon_day"] <= forecast_horizon
    ].reset_index(drop=True)

    merged_parameters = {**DEFAULT_MODEL_PARAMETERS, **(model_parameters or {})}
    metric_payload = {
        "records": _metric_records(metrics_frame),
        "primary_baseline_comparison": comparison,
    }
    date_values = pd.to_datetime(grid["date"])
    active_count = int(analytical_data.products["is_active"].astype(bool).sum())
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at_utc": all_forecasts["generated_at"].iloc[0],
        "source": {
            "mode": source_mode,
            "gcp_project_id": GCP_PROJECT_ID,
            "dataset_id": DBT_DATASET_ID,
            "location": BIGQUERY_LOCATION,
            "tables": ["fct_sales", "dim_products"],
            "bigquery_write_performed": False,
        },
        "training": {
            "target": "quantity_sold",
            "granularity": "product_day",
            "start_date": date_values.min().date(),
            "end_date": date_values.max().date(),
            "row_count": int(len(grid)),
            "product_count": int(grid["product_id"].nunique()),
            "active_product_count": active_count,
            "day_count": int(date_values.nunique()),
            "zero_quantity_rate": float(grid["quantity_sold"].eq(0).mean()),
            "features": list(MODEL_FEATURE_COLUMNS),
            "model_parameters": merged_parameters,
        },
        "validation": {
            "strategy": "expanding_window_date_based",
            "random_split_used": False,
            "validation_fold_count": len(validation_folds),
            "final_test_start_date": final_test_fold.validation_start_date,
            "final_test_end_date": final_test_fold.validation_end_date,
            "horizons": list(FORECAST_HORIZONS),
            "metrics": metric_payload,
        },
        "forecast": {
            "requested_horizon": forecast_horizon,
            "primary_horizon": PRIMARY_FORECAST_HORIZON,
            "origin_date": date_values.max().date(),
            "start_date": pd.to_datetime(forecasts["forecast_date"]).min().date(),
            "end_date": pd.to_datetime(forecasts["forecast_date"]).max().date(),
            "row_count": int(len(forecasts)),
            "active_products_only": True,
            "recursive": True,
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    paths = save_ml_artifacts(
        model=final_model,
        metadata=metadata,
        metrics=metric_payload,
        feature_columns=MODEL_FEATURE_COLUMNS,
        forecasts=forecasts,
        inventory_risk=inventory_risk,
        artifacts_dir=artifacts_dir,
    )
    return {
        "grid_rows": int(len(grid)),
        "product_count": int(grid["product_id"].nunique()),
        "active_product_count": active_count,
        "day_count": int(date_values.nunique()),
        "forecast_rows": int(len(forecasts)),
        "risk_distribution": {
            str(label): int(count)
            for label, count in inventory_risk["risk_class"]
            .value_counts(sort=False)
            .items()
        },
        "primary_baseline_comparison": comparison,
        "artifact_paths": paths.to_dict(),
        "bigquery_write_performed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina previsão de demanda e classifica risco de estoque."
    )
    parser.add_argument(
        "--skip-bigquery",
        action="store_true",
        help="Usa Parquets locais em data/ml_staging e não consulta o BigQuery.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        choices=FORECAST_HORIZONS,
        default=PRIMARY_FORECAST_HORIZON,
    )
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--staging-dir", type=Path, default=ML_STAGING_DIR)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = run_ml_pipeline(
        skip_bigquery=arguments.skip_bigquery,
        forecast_horizon=arguments.forecast_horizon,
        artifacts_dir=arguments.artifacts_dir,
        staging_dir=arguments.staging_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
