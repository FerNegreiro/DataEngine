from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.ml.baselines import BASELINE_PREDICTORS
from src.ml.config import FORECAST_HORIZONS, MODEL_NAME, MODEL_VERSION
from src.ml.feature_engineering import add_temporal_features
from src.ml.predict import model_predictor, recursive_forecast
from src.ml.temporal_split import TemporalFold
from src.ml.train import train_forecasting_model


def seasonal_naive_scale(training_grid: pd.DataFrame, seasonality: int = 7) -> float | None:
    if seasonality <= 0:
        raise ValueError("seasonality deve ser positiva")
    differences = (
        training_grid.sort_values(["product_id", "date"])
        .groupby("product_id", observed=True)["quantity_sold"]
        .diff(seasonality)
        .abs()
        .dropna()
    )
    if differences.empty:
        return None
    denominator = float(differences.mean())
    return denominator if denominator > 0 else None


def calculate_forecast_metrics(
    actual: Iterable[float],
    predicted: Iterable[float],
    *,
    mase_denominator: float | None = None,
) -> dict[str, float | None]:
    actual_values = np.asarray(list(actual), dtype=float)
    predicted_values = np.asarray(list(predicted), dtype=float)
    if actual_values.shape != predicted_values.shape or actual_values.size == 0:
        raise ValueError("Actual e predicted devem ter o mesmo tamanho positivo")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Actual e predicted devem conter somente valores finitos")
    if (actual_values < 0).any() or (predicted_values < 0).any():
        raise ValueError("Métricas de demanda exigem valores não negativos")

    errors = predicted_values - actual_values
    absolute_errors = np.abs(errors)
    actual_sum = float(actual_values.sum())
    symmetric_denominator = np.abs(actual_values) + np.abs(predicted_values)
    smape_terms = np.divide(
        2.0 * absolute_errors,
        symmetric_denominator,
        out=np.zeros_like(absolute_errors),
        where=symmetric_denominator != 0,
    )
    mae = float(absolute_errors.mean())
    return {
        "wape": float(absolute_errors.sum() / actual_sum) if actual_sum > 0 else None,
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "smape": float(smape_terms.mean()),
        "bias": float(errors.sum() / actual_sum) if actual_sum > 0 else None,
        "mase": (
            float(mae / mase_denominator)
            if mase_denominator is not None and mase_denominator > 0
            else None
        ),
    }


def calculate_occurrence_metrics(
    actual_occurrence: Iterable[int | bool],
    probabilities: Iterable[float],
    *,
    threshold: float,
) -> dict[str, float | int | None]:
    actual = np.asarray(list(actual_occurrence), dtype="int8")
    predicted_probability = np.asarray(list(probabilities), dtype=float)
    if actual.shape != predicted_probability.shape or actual.size == 0:
        raise ValueError("Ocorrências e probabilidades devem ter o mesmo tamanho positivo")
    if not np.isin(actual, [0, 1]).all():
        raise ValueError("A ocorrência real deve ser binária")
    if not np.isfinite(predicted_probability).all():
        raise ValueError("As probabilidades devem ser finitas")
    if ((predicted_probability < 0) | (predicted_probability > 1)).any():
        raise ValueError("As probabilidades devem estar entre zero e um")
    if not 0 < threshold < 1:
        raise ValueError("threshold deve estar entre zero e um")

    predicted = (predicted_probability >= threshold).astype("int8")
    return {
        "threshold": threshold,
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "pr_auc": (
            float(average_precision_score(actual, predicted_probability))
            if actual.sum() > 0
            else None
        ),
        "predicted_sale_day_rate": float(predicted.mean()),
        "actual_sale_day_rate": float(actual.mean()),
        "positive_day_count": int(actual.sum()),
        "row_count": int(actual.size),
    }


def _forecast_actuals(
    grid: pd.DataFrame,
    forecast: pd.DataFrame,
    fold: TemporalFold,
) -> pd.DataFrame:
    dates = pd.to_datetime(grid["date"]).dt.date
    actuals = grid.loc[
        (dates >= fold.validation_start_date) & (dates <= fold.validation_end_date),
        ["product_id", "date", "quantity_sold"],
    ].rename(columns={"date": "forecast_date", "quantity_sold": "actual_quantity"})
    comparison = actuals.merge(
        forecast[["product_id", "forecast_date", "horizon_day", "predicted_quantity"]],
        on=["product_id", "forecast_date"],
        how="left",
        validate="one_to_one",
    )
    if comparison["predicted_quantity"].isna().any():
        raise ValueError(f"Previsões ausentes no fold {fold.name}")
    return comparison


def _records_for_forecast(
    comparison: pd.DataFrame,
    fold: TemporalFold,
    model_name: str,
    mase_denominator: float | None,
    horizons: Iterable[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    available_days = int(comparison["horizon_day"].max())
    for horizon in horizons:
        if horizon > available_days:
            continue
        subset = comparison.loc[comparison["horizon_day"] <= horizon]
        metrics = calculate_forecast_metrics(
            subset["actual_quantity"],
            subset["predicted_quantity"],
            mase_denominator=mase_denominator,
        )
        records.append(
            {
                "split": fold.name,
                "model_name": model_name,
                "horizon": horizon,
                "train_start_date": fold.train_start_date.isoformat(),
                "train_end_date": fold.train_end_date.isoformat(),
                "evaluation_start_date": fold.validation_start_date.isoformat(),
                "evaluation_end_date": fold.validation_end_date.isoformat(),
                **metrics,
            }
        )
    return records


def evaluate_temporal_folds(
    grid: pd.DataFrame,
    folds: Iterable[TemporalFold],
    *,
    model_parameters: Mapping[str, Any] | None = None,
    horizons: Iterable[int] = FORECAST_HORIZONS,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    normalized_horizons = tuple(sorted(set(horizons)))
    for fold in folds:
        dates = pd.to_datetime(grid["date"]).dt.date
        training_grid = grid.loc[dates <= fold.train_end_date].copy()
        evaluation_days = (fold.validation_end_date - fold.validation_start_date).days + 1
        featured_training = add_temporal_features(training_grid)
        model = train_forecasting_model(featured_training, model_parameters)
        scale = seasonal_naive_scale(training_grid)

        candidate_forecast = recursive_forecast(
            training_grid,
            model_predictor(model),
            evaluation_days,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            active_only=False,
        )
        comparison = _forecast_actuals(grid, candidate_forecast, fold)
        records.extend(
            _records_for_forecast(
                comparison,
                fold,
                MODEL_NAME,
                scale,
                normalized_horizons,
            )
        )

        for baseline_name, baseline_predictor in BASELINE_PREDICTORS.items():
            baseline_forecast = recursive_forecast(
                training_grid,
                baseline_predictor,
                evaluation_days,
                model_name=baseline_name,
                model_version="deterministic",
                active_only=False,
            )
            comparison = _forecast_actuals(grid, baseline_forecast, fold)
            records.extend(
                _records_for_forecast(
                    comparison,
                    fold,
                    baseline_name,
                    scale,
                    normalized_horizons,
                )
            )
    return pd.DataFrame.from_records(records)


def primary_baseline_comparison(metrics: pd.DataFrame) -> dict[str, Any]:
    required = {"split", "model_name", "horizon", "wape"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Colunas ausentes nas métricas: {', '.join(sorted(missing))}")
    primary = metrics.loc[
        (metrics["horizon"] == 14)
        & metrics["model_name"].isin([MODEL_NAME, "moving_average_28"])
    ]
    pivot = primary.pivot(index="split", columns="model_name", values="wape")
    valid = pivot.dropna(subset=[MODEL_NAME, "moving_average_28"])
    validation = valid.loc[valid.index.str.startswith("validation_fold_")]
    wins = int((validation[MODEL_NAME] < validation["moving_average_28"]).sum())
    return {
        "primary_horizon": 14,
        "baseline": "moving_average_28",
        "validation_fold_wins": wins,
        "validation_fold_count": int(len(validation)),
        "candidate_passes_two_of_three": wins >= 2,
        "wape_by_split": {
            split: {
                MODEL_NAME: float(row[MODEL_NAME]),
                "moving_average_28": float(row["moving_average_28"]),
            }
            for split, row in valid.iterrows()
        },
    }
