from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from google.api_core.exceptions import (
    BadRequest,
    Forbidden,
    GoogleAPICallError,
    NotFound,
)
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery

from src.ml.config import (
    BIGQUERY_LOCATION,
    DBT_DATASET_ID,
    GCP_PROJECT_ID,
    ML_STAGING_DIR,
)

SALES_COLUMNS = ("product_id", "date", "quantity_sold", "revenue")
PRODUCT_COLUMNS = (
    "product_id",
    "product_name",
    "category",
    "brand",
    "unit_price",
    "unit_cost",
    "stock_quantity",
    "minimum_stock",
    "is_active",
)


class MLDataLoadError(RuntimeError):
    """Erro base de carregamento dos dados analíticos de ML."""


class MLAuthenticationError(MLDataLoadError):
    """Falha ao obter ou usar Application Default Credentials."""


class MLTableNotFoundError(MLDataLoadError):
    """Tabela analítica obrigatória não encontrada."""


class MLEmptyDataError(MLDataLoadError):
    """Fonte obrigatória retornou zero linhas."""


@dataclass(frozen=True)
class AnalyticalData:
    sales: pd.DataFrame
    products: pd.DataFrame


def build_full_table_id(
    table_name: str,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = DBT_DATASET_ID,
) -> str:
    return f"{project_id}.{dataset_id}.{table_name}"


def _create_bigquery_client(
    project_id: str = GCP_PROJECT_ID,
    location: str = BIGQUERY_LOCATION,
) -> bigquery.Client:
    try:
        return bigquery.Client(project=project_id, location=location)
    except DefaultCredentialsError as error:
        raise MLAuthenticationError(
            "Falha de autenticação com Application Default Credentials para o pipeline ML"
        ) from error


def _query_to_dataframe(client: Any, query: str, *, source_name: str) -> pd.DataFrame:
    try:
        rows = client.query(query, location=BIGQUERY_LOCATION).result()
        return pd.DataFrame([dict(row.items()) for row in rows])
    except DefaultCredentialsError as error:
        raise MLAuthenticationError(
            f"Falha de autenticação ADC ao consultar {source_name}"
        ) from error
    except NotFound as error:
        raise MLTableNotFoundError(
            f"Tabela BigQuery obrigatória não encontrada: {source_name}"
        ) from error
    except (BadRequest, Forbidden, GoogleAPICallError) as error:
        raise MLDataLoadError(
            f"Falha na consulta somente leitura ao BigQuery: tabela={source_name}: {error}"
        ) from error


def _validate_and_normalize_sales(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        raise MLEmptyDataError("A consulta de vendas realizadas retornou zero linhas")
    missing = set(SALES_COLUMNS).difference(dataframe.columns)
    if missing:
        raise MLDataLoadError(
            f"Colunas ausentes no resultado de vendas: {', '.join(sorted(missing))}"
        )

    sales = dataframe.loc[:, list(SALES_COLUMNS)].copy()
    sales["date"] = pd.to_datetime(sales["date"], errors="raise").dt.normalize()
    for column in ("quantity_sold", "revenue"):
        sales[column] = pd.to_numeric(sales[column], errors="raise")
    if sales[["product_id", "date", "quantity_sold", "revenue"]].isna().any().any():
        raise MLDataLoadError("Vendas contêm nulos em colunas obrigatórias")
    if (sales[["quantity_sold", "revenue"]] < 0).any().any():
        raise MLDataLoadError("Vendas contêm quantidade ou receita negativa")
    return sales.sort_values(["product_id", "date"]).reset_index(drop=True)


def _validate_and_normalize_products(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        raise MLEmptyDataError("A consulta de produtos retornou zero linhas")
    missing = set(PRODUCT_COLUMNS).difference(dataframe.columns)
    if missing:
        raise MLDataLoadError(
            f"Colunas ausentes no resultado de produtos: {', '.join(sorted(missing))}"
        )

    products = dataframe.loc[:, list(PRODUCT_COLUMNS)].copy()
    if products["product_id"].isna().any() or products["product_id"].duplicated().any():
        raise MLDataLoadError("dim_products deve ter um product_id único e não nulo")
    for column in ("unit_price", "unit_cost", "stock_quantity", "minimum_stock"):
        products[column] = pd.to_numeric(products[column], errors="raise")
    numeric_columns = ["unit_price", "unit_cost", "stock_quantity", "minimum_stock"]
    if (products[numeric_columns] < 0).any().any():
        raise MLDataLoadError("Produtos contêm preço, custo ou estoque negativo")
    active_values = products["is_active"]
    if active_values.dtype == object:
        normalized = active_values.astype(str).str.strip().str.lower()
        allowed = {"true", "false", "1", "0"}
        if not set(normalized).issubset(allowed):
            raise MLDataLoadError("is_active deve conter somente valores booleanos")
        products["is_active"] = normalized.isin({"true", "1"})
    else:
        products["is_active"] = active_values.astype(bool)
    return products.sort_values("product_id").reset_index(drop=True)


def load_analytical_data(
    bigquery_client: Any | None = None,
    project_id: str = GCP_PROJECT_ID,
    dataset_id: str = DBT_DATASET_ID,
) -> AnalyticalData:
    """Carrega somente vendas realizadas agregadas por produto-dia e o catálogo atual."""
    client = bigquery_client or _create_bigquery_client(project_id, BIGQUERY_LOCATION)
    sales_table = build_full_table_id("fct_sales", project_id, dataset_id)
    products_table = build_full_table_id("dim_products", project_id, dataset_id)

    sales_query = f"""
        SELECT
            product_id,
            DATE(order_date) AS date,
            SUM(quantity) AS quantity_sold,
            ROUND(SUM(item_total), 2) AS revenue
        FROM `{sales_table}`
        WHERE is_realized_sale
        GROUP BY product_id, date
        ORDER BY product_id, date
    """
    products_query = f"""
        SELECT
            product_id,
            product_name,
            category,
            brand,
            unit_price,
            unit_cost,
            stock_quantity,
            minimum_stock,
            is_active
        FROM `{products_table}`
        ORDER BY product_id
    """

    sales = _query_to_dataframe(client, sales_query, source_name=sales_table)
    products = _query_to_dataframe(client, products_query, source_name=products_table)
    return AnalyticalData(
        sales=_validate_and_normalize_sales(sales),
        products=_validate_and_normalize_products(products),
    )


def load_local_analytical_data(
    staging_dir: Path | str = ML_STAGING_DIR,
) -> AnalyticalData:
    """Carrega fixtures Parquet locais sem acessar o BigQuery."""
    directory = Path(staging_dir)
    sales_path = directory / "fct_sales.parquet"
    products_path = directory / "dim_products.parquet"
    missing = [str(path) for path in (sales_path, products_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Arquivos locais obrigatórios ausentes para --skip-bigquery: "
            + ", ".join(missing)
        )
    return AnalyticalData(
        sales=_validate_and_normalize_sales(pd.read_parquet(sales_path)),
        products=_validate_and_normalize_products(pd.read_parquet(products_path)),
    )
