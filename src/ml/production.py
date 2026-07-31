from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.baselines import predict_moving_average_28
from src.ml.config import (
    CHAMPION_MODEL,
    CHAMPION_MODEL_VERSION,
    CHAMPION_STATUS,
    DBT_DATASET_ID,
    EXPERIMENTS_DIR,
    FORECAST_HORIZONS,
    GCP_PROJECT_ID,
    PRIMARY_FORECAST_HORIZON,
    REJECTED_CHALLENGER_STATUS,
)
from src.ml.inventory_risk import classify_inventory_risk
from src.ml.predict import recursive_forecast
from src.ml.registry import (
    REGISTERED_MODELS,
    build_model_registry,
    model_version_for,
    validate_official_promotion_decision,
)
from src.ml.run_metadata import (
    build_pipeline_run,
    generate_run_id,
    resolve_code_version,
    utc_timestamp,
)

FORECAST_COLUMNS = (
    "run_id",
    "generated_at",
    "product_id",
    "forecast_date",
    "horizon_day",
    "horizon_days",
    "predicted_quantity",
    "model_name",
    "model_version",
    "champion_status",
    "data_max_date",
    "source_project",
    "source_dataset",
)
RISK_COLUMNS = (
    "run_id",
    "generated_at",
    "product_id",
    "product_name",
    "category",
    "stock_quantity",
    "minimum_stock",
    "forecast_demand",
    "projected_stock",
    "average_daily_demand",
    "estimated_coverage_days",
    "risk_level",
    "model_name",
    "model_version",
    "horizon_days",
)
METRIC_COLUMNS = (
    "run_id",
    "model_name",
    "model_version",
    "champion_status",
    "evaluation_period",
    "forecast_horizon",
    "metric_name",
    "metric_value",
    "generated_at",
)


@dataclass
class ProductionBundle:
    manifest: dict[str, Any]
    forecasts: pd.DataFrame
    inventory_risk: pd.DataFrame
    model_metrics: pd.DataFrame
    model_registry: pd.DataFrame
    pipeline_run: dict[str, Any]


def load_approved_experiment(
    experiment_dir: Path | str = EXPERIMENTS_DIR / "iteration_02",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    directory = Path(experiment_dir)
    paths = {
        "metrics": directory / "metrics.json",
        "promotion": directory / "promotion_decision.json",
        "comparison": directory / "model_comparison.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Artefatos oficiais da iteration_02 ausentes: " + ", ".join(missing)
        )
    metrics_payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    promotion = json.loads(paths["promotion"].read_text(encoding="utf-8"))
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    validate_official_promotion_decision(promotion, comparison)
    records = metrics_payload.get("aggregate_metrics")
    if not isinstance(records, list) or not records:
        raise ValueError("metrics.json não possui aggregate_metrics completos")
    return pd.DataFrame.from_records(records), promotion, comparison


def _build_official_forecasts(
    grid: pd.DataFrame,
    *,
    run_id: str,
    generated_at: pd.Timestamp,
    data_max_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    maximum_horizon = max(FORECAST_HORIZONS)
    base = recursive_forecast(
        grid,
        predict_moving_average_28,
        maximum_horizon,
        model_name=CHAMPION_MODEL,
        model_version=CHAMPION_MODEL_VERSION,
        active_only=True,
        generated_at=generated_at,
    )
    expanded: list[pd.DataFrame] = []
    for horizon in FORECAST_HORIZONS:
        horizon_forecast = base.loc[base["horizon_day"] <= horizon].copy()
        horizon_forecast["horizon_days"] = horizon
        expanded.append(horizon_forecast)
    forecasts = pd.concat(expanded, ignore_index=True)
    forecasts.insert(0, "run_id", run_id)
    forecasts["champion_status"] = CHAMPION_STATUS
    forecasts["data_max_date"] = data_max_date.date()
    forecasts["source_project"] = GCP_PROJECT_ID
    forecasts["source_dataset"] = DBT_DATASET_ID
    forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"]).dt.date
    return forecasts.loc[:, list(FORECAST_COLUMNS)], base


def _build_official_risk(
    base_forecasts: pd.DataFrame,
    products: pd.DataFrame,
    *,
    run_id: str,
    generated_at: pd.Timestamp,
) -> pd.DataFrame:
    risk = classify_inventory_risk(
        base_forecasts,
        products,
        horizon=PRIMARY_FORECAST_HORIZON,
    )
    names = products.loc[:, ["product_id", "product_name", "category"]]
    risk = risk.merge(names, on="product_id", how="left", validate="one_to_one")
    risk.insert(0, "run_id", run_id)
    risk.insert(1, "generated_at", generated_at)
    risk["average_daily_demand"] = (
        risk["forecast_demand"] / PRIMARY_FORECAST_HORIZON
    )
    risk["estimated_coverage_days"] = risk["coverage_days"].where(
        np.isfinite(risk["coverage_days"]), None
    )
    risk["risk_level"] = risk["risk_class"].astype(str)
    risk["model_name"] = CHAMPION_MODEL
    risk["model_version"] = CHAMPION_MODEL_VERSION
    risk["horizon_days"] = PRIMARY_FORECAST_HORIZON
    return risk.loc[:, list(RISK_COLUMNS)]


def _build_metric_records(
    aggregate_metrics: pd.DataFrame,
    *,
    run_id: str,
    generated_at: pd.Timestamp,
) -> pd.DataFrame:
    required = {
        "split",
        "model_name",
        "horizon",
        "wape",
        "mae",
        "rmse",
        "smape",
        "mase",
        "bias",
    }
    missing = required.difference(aggregate_metrics.columns)
    if missing:
        raise ValueError(
            "Métricas experimentais incompletas: " + ", ".join(sorted(missing))
        )
    selected = aggregate_metrics.loc[
        aggregate_metrics["model_name"].isin(REGISTERED_MODELS)
        & aggregate_metrics["horizon"].isin(FORECAST_HORIZONS)
    ]
    records: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        model_name = str(row["model_name"])
        for metric_name in ("wape", "mae", "rmse", "smape", "mase", "bias"):
            metric_value = row.get(metric_name)
            if metric_value is None or pd.isna(metric_value):
                continue
            records.append(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "model_version": model_version_for(model_name),
                    "champion_status": (
                        CHAMPION_STATUS
                        if model_name == CHAMPION_MODEL
                        else REJECTED_CHALLENGER_STATUS
                    ),
                    "evaluation_period": str(row["split"]),
                    "forecast_horizon": int(row["horizon"]),
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "generated_at": generated_at,
                }
            )
    metrics = pd.DataFrame.from_records(records)
    if metrics.empty:
        raise ValueError("Nenhuma métrica oficial foi preparada para publicação")
    return metrics.loc[:, list(METRIC_COLUMNS)]


def build_production_bundle(
    grid: pd.DataFrame,
    products: pd.DataFrame,
    *,
    experiment_dir: Path | str = EXPERIMENTS_DIR / "iteration_02",
    generated_at: pd.Timestamp | str | None = None,
    repository: Path | str | None = None,
) -> ProductionBundle:
    timestamp = utc_timestamp(generated_at)
    dates = pd.to_datetime(grid["date"], errors="raise")
    data_min_date = dates.min().date()
    data_max_timestamp = dates.max().normalize()
    data_max_date = data_max_timestamp.date()
    run_id = generate_run_id(timestamp, data_max_date)
    aggregate_metrics, promotion, comparison = load_approved_experiment(experiment_dir)

    forecasts, base_forecasts = _build_official_forecasts(
        grid,
        run_id=run_id,
        generated_at=timestamp,
        data_max_date=data_max_timestamp,
    )
    inventory_risk = _build_official_risk(
        base_forecasts,
        products,
        run_id=run_id,
        generated_at=timestamp,
    )
    metrics = _build_metric_records(
        aggregate_metrics,
        run_id=run_id,
        generated_at=timestamp,
    )
    code_version = resolve_code_version(repository)
    registry = build_model_registry(
        aggregate_metrics,
        promotion,
        comparison,
        registered_at=timestamp,
        training_data_min_date=data_min_date,
        training_data_max_date=data_max_date,
        code_version=code_version,
    )
    products_processed = int(forecasts["product_id"].nunique())
    pipeline_run = build_pipeline_run(
        run_id=run_id,
        started_at=timestamp,
        source_data_min_date=data_min_date,
        source_data_max_date=data_max_date,
        products_processed=products_processed,
        forecast_rows=len(forecasts),
        risk_rows=len(inventory_risk),
    )
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "generated_at": timestamp,
        "champion_model": CHAMPION_MODEL,
        "champion_version": CHAMPION_MODEL_VERSION,
        "champion_status": CHAMPION_STATUS,
        "source_project": GCP_PROJECT_ID,
        "source_dataset": DBT_DATASET_ID,
        "data_min_date": data_min_date,
        "data_max_date": data_max_date,
        "forecast_horizons": list(FORECAST_HORIZONS),
        "primary_forecast_horizon": PRIMARY_FORECAST_HORIZON,
        "products_processed": products_processed,
        "forecast_rows": int(len(forecasts)),
        "risk_rows": int(len(inventory_risk)),
        "metric_rows": int(len(metrics)),
        "registry_rows": int(len(registry)),
        "code_version": code_version,
        "promotion_decision": promotion,
        "model_comparison": comparison,
    }
    return ProductionBundle(
        manifest=manifest,
        forecasts=forecasts,
        inventory_risk=inventory_risk,
        model_metrics=metrics,
        model_registry=registry,
        pipeline_run=pipeline_run,
    )
