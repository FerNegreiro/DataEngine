from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.validation.validate_raw_data import validate_raw_data

PRODUCTS_PATH = Path("data/raw/products.csv")
CUSTOMERS_PATH = Path("data/raw/customers.csv")
ORDERS_PATH = Path("data/raw/orders.csv")
ORDER_ITEMS_PATH = Path("data/raw/order_items.csv")
OUTPUT_DIR = Path("data/processed")

DATASET_TYPES = {
    "products": {
        "string": (
            "product_id",
            "product_name",
            "category",
            "brand",
            "supplier",
        ),
        "float64": ("unit_price", "unit_cost"),
        "int64": ("stock_quantity", "minimum_stock"),
        "datetime64": ("created_at",),
        "bool": ("is_active",),
    },
    "customers": {
        "string": (
            "customer_id",
            "full_name",
            "email",
            "gender",
            "city",
            "state",
            "region",
            "acquisition_channel",
            "customer_segment",
        ),
        "float64": (),
        "int64": (),
        "datetime64": ("birth_date", "registration_date"),
        "bool": ("is_active",),
    },
    "orders": {
        "string": (
            "order_id",
            "customer_id",
            "order_status",
            "payment_method",
            "sales_channel",
        ),
        "float64": ("shipping_cost", "discount_amount", "order_total"),
        "int64": (),
        "datetime64": ("order_date", "delivery_date"),
        "bool": (),
    },
    "order_items": {
        "string": ("order_item_id", "order_id", "product_id"),
        "float64": (
            "unit_price",
            "unit_cost",
            "discount_percentage",
            "line_total",
        ),
        "int64": ("quantity",),
        "datetime64": (),
        "bool": (),
    },
}


def _convert_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("bool")

    normalized = series.astype("string").str.strip().str.lower()
    converted = normalized.map({"true": True, "false": False})
    if converted.isna().any():
        raise ValueError(f"Valores booleanos inválidos na coluna {series.name}")
    return converted.astype("bool")


def _apply_types(dataframe: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    converted = dataframe.copy()
    type_config = DATASET_TYPES[dataset_name]

    for column in type_config["string"]:
        converted[column] = converted[column].astype("string")
    for column in type_config["float64"]:
        converted[column] = pd.to_numeric(converted[column], errors="raise").astype(
            "float64"
        )
    for column in type_config["int64"]:
        converted[column] = pd.to_numeric(converted[column], errors="raise").astype("int64")
    for column in type_config["datetime64"]:
        converted[column] = pd.to_datetime(converted[column], errors="raise")
    for column in type_config["bool"]:
        converted[column] = _convert_boolean(converted[column])

    return converted


def _write_parquet(dataframe: pd.DataFrame, destination: Path) -> None:
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    pq.write_table(table, destination, compression="snappy")


def process_raw_to_parquet(
    products_path: Path | str = PRODUCTS_PATH,
    customers_path: Path | str = CUSTOMERS_PATH,
    orders_path: Path | str = ORDERS_PATH,
    order_items_path: Path | str = ORDER_ITEMS_PATH,
    output_dir: Path | str = OUTPUT_DIR,
) -> dict[str, object]:
    validation_report = validate_raw_data(
        products_path=products_path,
        customers_path=customers_path,
        orders_path=orders_path,
        order_items_path=order_items_path,
    )
    if not validation_report["is_valid"]:
        validation_errors = validation_report["errors"]
        details = "; ".join(validation_errors) if validation_errors else "erro desconhecido"
        raise ValueError(f"Validação dos dados brutos falhou: {details}")

    sources = {
        "products": Path(products_path),
        "customers": Path(customers_path),
        "orders": Path(orders_path),
        "order_items": Path(order_items_path),
    }
    dataframes = {
        name: _apply_types(pd.read_csv(path), name) for name, path in sources.items()
    }

    destination_directory = Path(output_dir)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destinations = {
        name: destination_directory / f"{name}.parquet" for name in dataframes
    }
    for name, dataframe in dataframes.items():
        _write_parquet(dataframe, destinations[name])

    return {
        "success": True,
        "files": {name: str(path) for name, path in destinations.items()},
        "rows": {name: len(dataframe) for name, dataframe in dataframes.items()},
        "columns": {
            name: list(dataframe.columns) for name, dataframe in dataframes.items()
        },
    }


def main() -> int:
    try:
        report = process_raw_to_parquet()
    except Exception as error:
        print(f"Falha no processamento para Parquet: {error}")
        return 1

    print("Processamento para Parquet concluído")
    for name, path in report["files"].items():
        print(f"- {name}: {report['rows'][name]} linha(s) em {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
