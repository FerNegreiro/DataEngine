from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.ml.config import HURDLE_FEATURE_COLUMNS, MODEL_NAME, MODEL_VERSION
from src.ml.feature_engineering import GRID_COLUMNS

Predictor = Callable[[pd.DataFrame], np.ndarray]


def _validate_history_grid(history_grid: pd.DataFrame) -> pd.DataFrame:
    missing = set(GRID_COLUMNS).difference(history_grid.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes no histórico: {', '.join(sorted(missing))}"
        )
    if history_grid.empty:
        raise ValueError("O histórico não pode ser vazio")
    if history_grid.duplicated(["product_id", "date"]).any():
        raise ValueError("O histórico deve ser único por produto e data")

    history = history_grid.loc[:, list(GRID_COLUMNS)].copy()
    history["date"] = pd.to_datetime(history["date"], errors="raise").dt.normalize()
    history = history.sort_values(["product_id", "date"]).reset_index(drop=True)
    expected_dates = pd.date_range(history["date"].min(), history["date"].max(), freq="D")
    counts = history.groupby("product_id", observed=True)["date"].nunique()
    if not counts.eq(len(expected_dates)).all():
        raise ValueError("Cada produto deve possuir a mesma grade diária completa")
    if len(history) != counts.size * len(expected_dates):
        raise ValueError("O histórico não representa uma grade produto-dia completa")
    return history


def _lag(values: list[float], days: int) -> float:
    return values[-days] if len(values) >= days else np.nan


def _window(values: list[float], days: int) -> np.ndarray:
    return np.asarray(values[-days:], dtype=float)


def _build_step_features(
    product_ids: list[Any],
    quantities: dict[Any, list[float]],
    static_products: pd.DataFrame,
    forecast_date: pd.Timestamp,
    first_history_date: pd.Timestamp,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for product_id in product_ids:
        values = quantities[product_id]
        static = static_products.loc[product_id]
        record: dict[str, Any] = {
            "product_id": product_id,
            "category": static["category"],
            "brand": static["brand"],
            "day_of_week": forecast_date.dayofweek,
            "day_of_month": forecast_date.day,
            "month": forecast_date.month,
            "quarter": forecast_date.quarter,
            "is_weekend": int(forecast_date.dayofweek in (5, 6)),
            "time_index": (forecast_date - first_history_date).days,
            "unit_price": static["unit_price"],
            "unit_cost": static["unit_cost"],
        }
        for lag in (1, 7, 14, 28, 30):
            record[f"lag_{lag}"] = _lag(values, lag)
        for window in (7, 14, 28, 30):
            observations = _window(values, window)
            record[f"rolling_mean_{window}"] = observations.mean()
            record[f"rolling_std_{window}"] = observations.std(ddof=0)
        for window in (7, 14, 30):
            record[f"sales_last_{window}_days"] = _window(values, window).sum()
        positive_values = np.asarray(values, dtype=float) > 0
        positive_count = int(positive_values.sum())
        for window in (7, 14, 28, 30):
            record[f"sale_days_last_{window}"] = int(positive_values[-window:].sum())
        record["historical_sale_probability"] = (
            positive_count / len(values) if values else 0.0
        )
        record["mean_positive_demand"] = (
            float(np.asarray(values, dtype=float)[positive_values].mean())
            if positive_count
            else 0.0
        )
        record["causal_adi"] = len(values) / positive_count if positive_count else np.nan
        positive_indices = np.flatnonzero(positive_values)
        record["days_since_last_sale"] = (
            len(values) - int(positive_indices[-1]) if positive_indices.size else np.nan
        )
        zero_streak = 0
        for quantity in reversed(values):
            if quantity > 0:
                break
            zero_streak += 1
        record["current_zero_streak"] = zero_streak
        records.append(record)
    return pd.DataFrame.from_records(records).loc[:, list(HURDLE_FEATURE_COLUMNS)]


def recursive_forecast(
    history_grid: pd.DataFrame,
    predictor: Predictor,
    horizon: int,
    *,
    model_name: str = MODEL_NAME,
    model_version: str = MODEL_VERSION,
    active_only: bool = True,
    generated_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Prevê passo a passo, realimentando apenas previsões após a origem."""
    if horizon <= 0:
        raise ValueError("horizon deve ser positivo")
    history = _validate_history_grid(history_grid)
    first_history_date = history["date"].min()
    origin_date = history["date"].max()
    latest_products = (
        history.sort_values("date")
        .groupby("product_id", observed=True, sort=True)
        .tail(1)
        .set_index("product_id")
    )
    if active_only:
        latest_products = latest_products.loc[latest_products["is_active"].astype(bool)]
    if latest_products.empty:
        raise ValueError("Nenhum produto elegível para previsão")

    product_ids = latest_products.index.tolist()
    quantities = {
        product_id: group["quantity_sold"].astype(float).tolist()
        for product_id, group in history.groupby(
            "product_id", observed=True, sort=True
        )
        if product_id in latest_products.index
    }
    timestamp = generated_at or pd.Timestamp.now(tz="UTC")
    output: list[dict[str, Any]] = []
    for horizon_day in range(1, horizon + 1):
        forecast_date = origin_date + pd.Timedelta(days=horizon_day)
        features = _build_step_features(
            product_ids,
            quantities,
            latest_products,
            forecast_date,
            first_history_date,
        )
        predictions = np.asarray(predictor(features), dtype=float)
        if predictions.shape != (len(product_ids),):
            raise ValueError("O predictor deve retornar uma previsão por produto")
        if not np.isfinite(predictions).all():
            raise ValueError("O predictor retornou valores não finitos")
        predictions = np.clip(predictions, a_min=0.0, a_max=None)
        for product_id, prediction in zip(product_ids, predictions, strict=True):
            quantities[product_id].append(float(prediction))
            output.append(
                {
                    "product_id": product_id,
                    "forecast_date": forecast_date,
                    "horizon_day": horizon_day,
                    "predicted_quantity": float(prediction),
                    "model_name": model_name,
                    "model_version": model_version,
                    "generated_at": timestamp,
                }
            )
    return pd.DataFrame.from_records(output)


def model_predictor(model: Any) -> Predictor:
    def predict(features: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict(features), dtype=float)

    return predict


def forecast_origin(forecasts: pd.DataFrame) -> date:
    if forecasts.empty:
        raise ValueError("Forecast vazio não possui origem")
    return (pd.to_datetime(forecasts["forecast_date"]).min() - pd.Timedelta(days=1)).date()
