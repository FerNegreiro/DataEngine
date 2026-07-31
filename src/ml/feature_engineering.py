from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.ml.config import LAG_DAYS, ROLLING_WINDOWS
from src.ml.data_loader import PRODUCT_COLUMNS, SALES_COLUMNS

GRID_COLUMNS = (
    "product_id",
    "date",
    "quantity_sold",
    "revenue",
    *PRODUCT_COLUMNS[1:],
)


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    dataframe_name: str,
) -> None:
    missing = set(columns).difference(dataframe.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes em {dataframe_name}: {', '.join(sorted(missing))}"
        )


def build_product_day_grid(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Cria a grade completa produto-dia e representa ausência legítima de venda como zero."""
    _require_columns(sales, SALES_COLUMNS, "sales")
    _require_columns(products, PRODUCT_COLUMNS, "products")
    if sales.empty or products.empty:
        raise ValueError("Vendas e produtos devem possuir ao menos uma linha")
    if products["product_id"].duplicated().any():
        raise ValueError("products deve ter exatamente uma linha por product_id")

    daily_sales = sales.loc[:, list(SALES_COLUMNS)].copy()
    daily_sales["date"] = pd.to_datetime(daily_sales["date"], errors="raise").dt.normalize()
    daily_sales = (
        daily_sales.groupby(["product_id", "date"], as_index=False, observed=True)[
            ["quantity_sold", "revenue"]
        ]
        .sum()
        .sort_values(["product_id", "date"])
    )
    unknown_products = set(daily_sales["product_id"]).difference(products["product_id"])
    if unknown_products:
        raise ValueError(
            "Vendas possuem produtos ausentes em dim_products: "
            + ", ".join(sorted(unknown_products))
        )

    minimum_date = (
        pd.Timestamp(start_date).normalize() if start_date else daily_sales["date"].min()
    )
    maximum_date = (
        pd.Timestamp(end_date).normalize() if end_date else daily_sales["date"].max()
    )
    if minimum_date > maximum_date:
        raise ValueError("start_date deve ser anterior ou igual a end_date")

    product_ids = products["product_id"].sort_values().tolist()
    dates = pd.date_range(minimum_date, maximum_date, freq="D")
    grid = pd.MultiIndex.from_product(
        [product_ids, dates],
        names=["product_id", "date"],
    ).to_frame(index=False)
    grid = grid.merge(
        daily_sales,
        on=["product_id", "date"],
        how="left",
        validate="one_to_one",
    )
    grid[["quantity_sold", "revenue"]] = grid[
        ["quantity_sold", "revenue"]
    ].fillna(0.0)
    grid = grid.merge(
        products.loc[:, list(PRODUCT_COLUMNS)],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    if grid.duplicated(["product_id", "date"]).any():
        raise ValueError("A grade produto-dia contém chaves duplicadas")
    if grid.loc[:, list(PRODUCT_COLUMNS)].isna().any().any():
        raise ValueError("A grade produto-dia perdeu atributos obrigatórios de produto")
    if (grid[["quantity_sold", "revenue"]] < 0).any().any():
        raise ValueError("A grade produto-dia contém venda ou receita negativa")
    return (
        grid.loc[:, list(GRID_COLUMNS)]
        .sort_values(["product_id", "date"])
        .reset_index(drop=True)
    )


def add_temporal_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de calendário, lags e janelas encerradas em t-1."""
    _require_columns(grid, GRID_COLUMNS, "grid")
    if grid.empty:
        raise ValueError("A grade produto-dia não pode ser vazia")
    if grid.duplicated(["product_id", "date"]).any():
        raise ValueError("A grade produto-dia deve ser única por product_id e date")

    featured = grid.copy()
    featured["date"] = pd.to_datetime(featured["date"], errors="raise").dt.normalize()
    featured = featured.sort_values(["product_id", "date"]).reset_index(drop=True)
    dates = featured["date"]
    featured["day_of_week"] = dates.dt.dayofweek.astype("int8")
    featured["day_of_month"] = dates.dt.day.astype("int8")
    featured["month"] = dates.dt.month.astype("int8")
    featured["quarter"] = dates.dt.quarter.astype("int8")
    featured["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype("int8")
    featured["time_index"] = (dates - dates.min()).dt.days.astype("int32")

    grouped = featured.groupby("product_id", sort=False, observed=True)["quantity_sold"]
    for lag in LAG_DAYS:
        featured[f"lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        featured[f"rolling_mean_{window}"] = grouped.transform(
            lambda series, size=window: series.shift(1)
            .rolling(size, min_periods=1)
            .mean()
        )
        featured[f"rolling_std_{window}"] = grouped.transform(
            lambda series, size=window: series.shift(1)
            .rolling(size, min_periods=1)
            .std(ddof=0)
        )

    for window in (7, 14, 30):
        featured[f"sales_last_{window}_days"] = grouped.transform(
            lambda series, size=window: series.shift(1)
            .rolling(size, min_periods=1)
            .sum()
        )
    return featured
