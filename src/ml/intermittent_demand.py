from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from src.ml.config import (
    DEFAULT_CROSTON_ALPHA,
    DEFAULT_TSB_ALPHA,
    DEFAULT_TSB_BETA,
)

INTERMITTENT_METHODS = ("croston", "croston_sba", "tsb")


def _validate_series(demand: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(demand), dtype=float)
    if values.size == 0:
        raise ValueError("A série de demanda não pode ser vazia")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("A demanda deve conter somente valores finitos e não negativos")
    return values


def _validate_smoothing(value: float, name: str) -> None:
    if not 0 < value <= 1:
        raise ValueError(f"{name} deve estar no intervalo (0, 1]")


def croston_level(
    demand: Iterable[float],
    *,
    alpha: float = DEFAULT_CROSTON_ALPHA,
    variant: str = "classic",
) -> float:
    values = _validate_series(demand)
    _validate_smoothing(alpha, "alpha")
    if variant not in {"classic", "sba"}:
        raise ValueError("variant deve ser classic ou sba")
    positive_indices = np.flatnonzero(values > 0)
    if positive_indices.size == 0:
        return 0.0

    first_index = int(positive_indices[0])
    demand_level = float(values[first_index])
    interval_level = float(first_index + 1)
    previous_index = first_index
    for index in positive_indices[1:]:
        current_index = int(index)
        interval = current_index - previous_index
        demand_level += alpha * (float(values[current_index]) - demand_level)
        interval_level += alpha * (interval - interval_level)
        previous_index = current_index
    forecast = demand_level / interval_level
    if variant == "sba":
        forecast *= 1 - alpha / 2
    return max(0.0, float(forecast))


def tsb_level(
    demand: Iterable[float],
    *,
    alpha: float = DEFAULT_TSB_ALPHA,
    beta: float = DEFAULT_TSB_BETA,
) -> float:
    values = _validate_series(demand)
    _validate_smoothing(alpha, "alpha")
    _validate_smoothing(beta, "beta")
    positive_indices = np.flatnonzero(values > 0)
    if positive_indices.size == 0:
        return 0.0

    first_index = int(positive_indices[0])
    probability = 1.0 / (first_index + 1)
    demand_level = float(values[first_index])
    for quantity in values[first_index + 1 :]:
        occurrence = float(quantity > 0)
        probability += beta * (occurrence - probability)
        if occurrence:
            demand_level += alpha * (float(quantity) - demand_level)
    return max(0.0, float(probability * demand_level))


def intermittent_level(
    demand: Iterable[float],
    method: str,
    *,
    alpha: float = DEFAULT_CROSTON_ALPHA,
    beta: float = DEFAULT_TSB_BETA,
) -> float:
    if method == "croston":
        return croston_level(demand, alpha=alpha, variant="classic")
    if method == "croston_sba":
        return croston_level(demand, alpha=alpha, variant="sba")
    if method == "tsb":
        return tsb_level(demand, alpha=alpha, beta=beta)
    raise ValueError(f"Método intermitente desconhecido: {method}")


def forecast_intermittent_grid(
    history_grid: pd.DataFrame,
    horizon: int,
    method: str,
    *,
    alpha: float = DEFAULT_CROSTON_ALPHA,
    beta: float = DEFAULT_TSB_BETA,
    active_only: bool = True,
    generated_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon deve ser positivo")
    required = {"product_id", "date", "quantity_sold", "is_active"}
    missing = required.difference(history_grid.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no histórico: {', '.join(sorted(missing))}")
    history = history_grid.sort_values(["product_id", "date"]).copy()
    origin_date = pd.to_datetime(history["date"]).max().normalize()
    timestamp = generated_at or pd.Timestamp.now(tz="UTC")
    latest = history.groupby("product_id", observed=True, sort=True).tail(1)
    eligible = set(
        latest.loc[latest["is_active"].astype(bool), "product_id"]
        if active_only
        else latest["product_id"]
    )

    output: list[dict[str, Any]] = []
    for product_id, group in history.groupby("product_id", observed=True, sort=True):
        if product_id not in eligible:
            continue
        level = intermittent_level(
            group["quantity_sold"], method, alpha=alpha, beta=beta
        )
        for horizon_day in range(1, horizon + 1):
            output.append(
                {
                    "product_id": product_id,
                    "forecast_date": origin_date + pd.Timedelta(days=horizon_day),
                    "horizon_day": horizon_day,
                    "predicted_quantity": level,
                    "model_name": method,
                    "model_version": "deterministic_v1",
                    "generated_at": timestamp,
                }
            )
    return pd.DataFrame.from_records(output)
