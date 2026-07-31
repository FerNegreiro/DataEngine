from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.config import PRIMARY_FORECAST_HORIZON

RISK_ORDER = (
    "critical",
    "high_risk",
    "attention",
    "adequate",
    "potential_overstock",
)


def classify_inventory_risk(
    forecasts: pd.DataFrame,
    products: pd.DataFrame,
    *,
    horizon: int = PRIMARY_FORECAST_HORIZON,
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon deve ser positivo")
    forecast_columns = {
        "product_id",
        "forecast_date",
        "horizon_day",
        "predicted_quantity",
    }
    product_columns = {
        "product_id",
        "stock_quantity",
        "minimum_stock",
        "is_active",
    }
    missing_forecast = forecast_columns.difference(forecasts.columns)
    missing_products = product_columns.difference(products.columns)
    if missing_forecast or missing_products:
        raise ValueError(
            "Colunas ausentes para risco de estoque: "
            + ", ".join(sorted(missing_forecast | missing_products))
        )
    if products["product_id"].duplicated().any():
        raise ValueError("products deve ser único por product_id")

    eligible_forecasts = forecasts.loc[forecasts["horizon_day"] <= horizon].copy()
    day_counts = eligible_forecasts.groupby("product_id", observed=True)[
        "horizon_day"
    ].nunique()
    if day_counts.empty or not day_counts.eq(horizon).all():
        raise ValueError(f"Cada produto deve possuir exatamente {horizon} dias previstos")
    if (eligible_forecasts["predicted_quantity"] < 0).any():
        raise ValueError("Previsões negativas não são válidas para risco de estoque")

    demand = (
        eligible_forecasts.groupby("product_id", as_index=False, observed=True)[
            "predicted_quantity"
        ]
        .sum()
        .rename(columns={"predicted_quantity": "forecast_demand"})
    )
    active_products = products.loc[
        products["is_active"].astype(bool),
        ["product_id", "stock_quantity", "minimum_stock"],
    ]
    risk = active_products.merge(demand, on="product_id", how="inner", validate="one_to_one")
    risk["projected_stock"] = risk["stock_quantity"] - risk["forecast_demand"]
    risk["coverage_days"] = np.divide(
        horizon * risk["stock_quantity"].to_numpy(dtype=float),
        risk["forecast_demand"].to_numpy(dtype=float),
        out=np.full(len(risk), np.inf, dtype=float),
        where=risk["forecast_demand"].to_numpy(dtype=float) > 0,
    )

    critical = (risk["stock_quantity"] <= 0) | (risk["projected_stock"] <= 0)
    high_risk = (
        (risk["projected_stock"] > 0)
        & (risk["projected_stock"] <= risk["minimum_stock"])
    )
    attention = (
        (risk["projected_stock"] > risk["minimum_stock"])
        & (risk["coverage_days"] < 2 * horizon)
    )
    potential_overstock = (
        (risk["projected_stock"] > risk["minimum_stock"])
        & (
            (risk["forecast_demand"] == 0)
            | (risk["coverage_days"] > 3 * horizon)
        )
    )
    risk["risk_class"] = np.select(
        [critical, high_risk, attention, potential_overstock],
        ["critical", "high_risk", "attention", "potential_overstock"],
        default="adequate",
    )
    risk["risk_class"] = pd.Categorical(
        risk["risk_class"], categories=list(RISK_ORDER), ordered=True
    )
    return risk.sort_values(["risk_class", "product_id"]).reset_index(drop=True)
