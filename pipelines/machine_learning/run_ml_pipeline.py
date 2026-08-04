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

from pipelines.machine_learning.publish_ml_results import publish_ml_results
from src.ml.artifacts import (
    save_experiment_artifacts,
    save_ml_artifacts,
    save_production_artifacts,
)
from src.ml.config import (
    ARTIFACTS_DIR,
    BIGQUERY_LOCATION,
    DBT_DATASET_ID,
    DEFAULT_MODEL_PARAMETERS,
    EXPECTED_DATA_START_DATE,
    EXPERIMENTS_DIR,
    FINAL_TEST_END_DATE,
    FORECAST_HORIZONS,
    GCP_PROJECT_ID,
    ML_STAGING_DIR,
    MODEL_FEATURE_COLUMNS,
    MODEL_NAME,
    MODEL_VERSION,
    PRIMARY_FORECAST_HORIZON,
    PRODUCTION_ARTIFACTS_DIR,
)
from src.ml.data_loader import AnalyticalData, load_analytical_data, load_local_analytical_data
from src.ml.evaluate import (
    evaluate_temporal_folds,
    primary_baseline_comparison,
)
from src.ml.feature_engineering import add_temporal_features, build_product_day_grid
from src.ml.inventory_risk import classify_inventory_risk
from src.ml.model_selection import evaluate_iteration_02
from src.ml.predict import model_predictor, recursive_forecast
from src.ml.production import build_production_bundle
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
    experiment: str | None = None,
    hurdle_classifier_parameters: Mapping[str, Any] | None = None,
    hurdle_regressor_parameters: Mapping[str, Any] | None = None,
    prepare_publication: bool = False,
    publish_bigquery: bool = False,
    production_artifacts_dir: Path | str = PRODUCTION_ARTIFACTS_DIR,
    approved_experiment_dir: Path | str = EXPERIMENTS_DIR / "iteration_02",
) -> dict[str, Any]:
    if forecast_horizon not in FORECAST_HORIZONS:
        raise ValueError(f"forecast_horizon deve ser um de {FORECAST_HORIZONS}")
    if experiment is not None and experiment != "iteration_02":
        raise ValueError("experiment deve ser iteration_02")
    if (prepare_publication or publish_bigquery) and experiment is not None:
        raise ValueError(
            "Modos de publicação não podem executar uma iteração experimental"
        )
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
    publication_requested = prepare_publication or publish_bigquery
    grid_end_date = source_dates.max() if publication_requested else FINAL_TEST_END_DATE
    grid = build_product_day_grid(
        analytical_data.sales,
        analytical_data.products,
        start_date=EXPECTED_DATA_START_DATE,
        end_date=grid_end_date,
    )
    validate_required_date_range(
        grid,
        required_start=EXPECTED_DATA_START_DATE,
        required_end=FINAL_TEST_END_DATE,
    )

    if publication_requested:
        bundle = build_production_bundle(
            grid,
            analytical_data.products,
            experiment_dir=approved_experiment_dir,
        )
        production_paths = save_production_artifacts(
            manifest=bundle.manifest,
            forecasts=bundle.forecasts,
            inventory_risk=bundle.inventory_risk,
            model_metrics=bundle.model_metrics,
            model_registry=bundle.model_registry,
            pipeline_run=bundle.pipeline_run,
            artifacts_dir=production_artifacts_dir,
        )
        result = {
            "prepare_publication": prepare_publication,
            "publish_bigquery": publish_bigquery,
            "run_id": bundle.manifest["run_id"],
            "champion_model": bundle.manifest["champion_model"],
            "champion_version": bundle.manifest["champion_version"],
            "forecast_horizons": list(FORECAST_HORIZONS),
            "forecast_rows": int(len(bundle.forecasts)),
            "risk_rows": int(len(bundle.inventory_risk)),
            "metric_rows": int(len(bundle.model_metrics)),
            "registry_rows": int(len(bundle.model_registry)),
            "active_product_count": int(bundle.manifest["products_processed"]),
            "artifact_paths": production_paths.to_dict(),
            "bigquery_write_performed": False,
        }
        if prepare_publication:
            return result
        publication = publish_ml_results(
            bundle,
            artifacts_dir=production_artifacts_dir,
            bigquery_client=bigquery_client,
        )
        result.update(
            {
                "run_id": publication["run_id"],
                "champion_model": publication["champion_model"],
                "champion_version": publication["champion_version"],
                "publication": publication,
                "bigquery_write_performed": True,
            }
        )
        return result

    if experiment is not None:
        experiment_result = evaluate_iteration_02(
            grid,
            analytical_data.products,
            forecast_horizon=forecast_horizon,
            model_parameters=model_parameters,
            hurdle_classifier_parameters=hurdle_classifier_parameters,
            hurdle_regressor_parameters=hurdle_regressor_parameters,
        )
        experiment_directory = Path(artifacts_dir) / "experiments" / experiment
        experiment_paths = save_experiment_artifacts(
            aggregate_metrics=experiment_result.aggregate_metrics,
            occurrence_metrics=experiment_result.occurrence_metrics,
            segment_metrics=experiment_result.segment_metrics,
            product_metrics=experiment_result.product_metrics,
            demand_segments=experiment_result.demand_segments,
            model_comparison=experiment_result.model_comparison,
            promotion_decision=experiment_result.promotion_decision,
            forecasts=experiment_result.forecasts,
            inventory_risk_comparison=experiment_result.inventory_risk_comparison,
            models=experiment_result.models,
            artifacts_dir=experiment_directory,
        )
        final_segments = experiment_result.demand_segments.loc[
            experiment_result.demand_segments["split"].eq("final_test")
        ]
        champion = experiment_result.promotion_decision["final_champion"]
        champion_risk = experiment_result.inventory_risk_comparison.loc[
            experiment_result.inventory_risk_comparison["model_name"].eq(champion)
        ]
        return {
            "experiment": experiment,
            "grid_rows": int(len(grid)),
            "product_count": int(grid["product_id"].nunique()),
            "demand_segment_distribution": {
                str(pattern): int(count)
                for pattern, count in final_segments["demand_pattern"]
                .value_counts()
                .sort_index()
                .items()
            },
            "occurrence_threshold": experiment_result.model_comparison[
                "occurrence_threshold_selection"
            ],
            "promotion_decision": experiment_result.promotion_decision,
            "final_champion": champion,
            "champion_risk_distribution": {
                str(risk_class): int(count)
                for risk_class, count in champion_risk["risk_class"]
                .value_counts()
                .items()
            },
            "risk_changes_vs_final_champion": experiment_result.model_comparison[
                "risk_changes_vs_final_champion"
            ],
            "artifact_paths": experiment_paths.to_dict(),
            "bigquery_write_performed": False,
        }

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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    publication_mode = parser.add_mutually_exclusive_group()
    publication_mode.add_argument(
        "--experiment",
        choices=("iteration_02",),
        help="Executa uma iteração experimental sem substituir os artefatos anteriores.",
    )
    publication_mode.add_argument(
        "--prepare-publication",
        action="store_true",
        help=(
            "Prepara e valida o pacote produtivo sem gravar no BigQuery, para "
            "publicação posterior idempotente."
        ),
    )
    publication_mode.add_argument(
        "--publish-bigquery",
        action="store_true",
        help=(
            "Publica somente o champion moving_average_28 no dataset dataengine_ml, "
            "com MERGE idempotente."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = _parse_args(argv)
    result = run_ml_pipeline(
        skip_bigquery=arguments.skip_bigquery,
        forecast_horizon=arguments.forecast_horizon,
        artifacts_dir=arguments.artifacts_dir,
        staging_dir=arguments.staging_dir,
        experiment=arguments.experiment,
        prepare_publication=arguments.prepare_publication,
        publish_bigquery=arguments.publish_bigquery,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
