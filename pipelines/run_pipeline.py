from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import pandas as pd

from pipelines.loading.load_silver_to_bigquery import load_silver_to_bigquery
from pipelines.loading.upload_to_s3 import (
    normalize_execution_date,
    upload_processed_files,
)
from pipelines.processing.process_bronze_to_silver import process_bronze_to_silver
from src.extraction.generate_customers import generate_customers
from src.extraction.generate_orders import generate_orders
from src.extraction.generate_products import generate_products
from src.transformation.process_raw_to_parquet import process_raw_to_parquet
from src.validation.validate_raw_data import validate_raw_data

StageResult = TypeVar("StageResult")


def _run_stage(name: str, action: Callable[[], StageResult]) -> StageResult:
    print(f"Iniciando etapa: {name}")
    try:
        result = action()
    except Exception as error:
        raise RuntimeError(f"Falha na etapa {name}: {error}") from error
    print(f"Etapa concluída: {name}")
    return result


def _validate_or_raise(
    products_path: Path,
    customers_path: Path,
    orders_path: Path,
    order_items_path: Path,
) -> dict[str, object]:
    report = validate_raw_data(
        products_path=products_path,
        customers_path=customers_path,
        orders_path=orders_path,
        order_items_path=order_items_path,
    )
    if not report["is_valid"]:
        errors = report["errors"]
        details = "; ".join(errors) if errors else "erro desconhecido"
        raise ValueError(f"Dados brutos inválidos: {details}")
    return report


def run_pipeline(
    products_quantity: int = 100,
    customers_quantity: int = 500,
    orders_quantity: int = 2000,
    seed: int = 42,
    raw_dir: Path | str = "data/raw",
    processed_dir: Path | str = "data/processed",
    silver_staging_dir: Path | str = "data/silver_staging",
    execution_date: datetime | None = None,
) -> dict[str, object]:
    quantities = {
        "products_quantity": products_quantity,
        "customers_quantity": customers_quantity,
        "orders_quantity": orders_quantity,
    }
    for name, value in quantities.items():
        if value <= 0:
            raise ValueError(f"{name} deve ser maior que zero")

    utc_execution_date = normalize_execution_date(execution_date)
    raw_directory = Path(raw_dir)
    processed_directory = Path(processed_dir)
    products_path = raw_directory / "products.csv"
    customers_path = raw_directory / "customers.csv"
    orders_path = raw_directory / "orders.csv"
    order_items_path = raw_directory / "order_items.csv"

    products = _run_stage(
        "geração de produtos",
        lambda: generate_products(
            quantity=products_quantity,
            seed=seed,
            output_path=products_path,
        ),
    )
    customers = _run_stage(
        "geração de clientes",
        lambda: generate_customers(
            quantity=customers_quantity,
            seed=seed + 1,
            output_path=customers_path,
        ),
    )

    def generate_order_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        return generate_orders(
            quantity=orders_quantity,
            seed=seed + 2,
            customers_path=customers_path,
            products_path=products_path,
            orders_output_path=orders_path,
            order_items_output_path=order_items_path,
        )

    orders, order_items = _run_stage("geração de pedidos e itens", generate_order_data)
    validation_report = _run_stage(
        "validação dos dados brutos",
        lambda: _validate_or_raise(
            products_path,
            customers_path,
            orders_path,
            order_items_path,
        ),
    )
    processing_report = _run_stage(
        "processamento para Parquet",
        lambda: process_raw_to_parquet(
            products_path=products_path,
            customers_path=customers_path,
            orders_path=orders_path,
            order_items_path=order_items_path,
            output_dir=processed_directory,
        ),
    )
    upload_report = _run_stage(
        "upload para AWS S3 Bronze",
        lambda: upload_processed_files(
            processed_dir=processed_directory,
            execution_date=utc_execution_date,
        ),
    )
    silver_report = _run_stage(
        "processamento da camada Silver",
        lambda: process_bronze_to_silver(
            execution_date=utc_execution_date,
            bucket_name=upload_report["bucket"],
            staging_dir=silver_staging_dir,
        ),
    )
    if silver_report["partition"] != upload_report["partition"]:
        raise RuntimeError("As camadas Bronze e Silver utilizaram partições diferentes")
    silver_files = {
        file_report["dataset"]: file_report["local_path"]
        for file_report in silver_report["files"]
    }
    bigquery_report = _run_stage(
        "carga da camada Silver no BigQuery",
        lambda: load_silver_to_bigquery(
            silver_files=silver_files,
            silver_report=silver_report,
            execution_date=utc_execution_date,
        ),
    )
    if bigquery_report["partition"] != silver_report["partition"]:
        raise RuntimeError(
            "Bronze, Silver e BigQuery utilizaram partições diferentes"
        )

    return {
        "success": True,
        "raw_files": {
            "products": str(products_path),
            "customers": str(customers_path),
            "orders": str(orders_path),
            "order_items": str(order_items_path),
        },
        "processed_files": processing_report["files"],
        "rows": {
            "products": len(products),
            "customers": len(customers),
            "orders": len(orders),
            "order_items": len(order_items),
        },
        "validation": {
            "is_valid": validation_report["is_valid"],
            "errors": validation_report["errors"],
            "warnings": validation_report["warnings"],
        },
        "s3": upload_report,
        "silver": silver_report,
        "bigquery": bigquery_report,
    }


def main() -> int:
    try:
        report = run_pipeline()
    except Exception as error:
        print(f"Pipeline finalizado com falha: {error}")
        return 1

    print("Pipeline concluído com sucesso")
    for name, rows in report["rows"].items():
        print(f"- {name}: {rows} linha(s)")
    print(
        f"- Bronze: {report['s3']['uploaded_count']} arquivo(s) enviado(s) "
        f"para o bucket {report['s3']['bucket']}"
    )
    print(
        f"- Silver: {report['silver']['uploaded_count']} arquivo(s) enviado(s) "
        f"para o bucket {report['silver']['bucket']}"
    )
    print(
        f"- BigQuery: {report['bigquery']['loaded_count']} tabela(s) carregada(s)"
    )
    print(f"- Projeto: {report['bigquery']['project_id']}")
    print(f"- Dataset: {report['bigquery']['dataset_id']}")
    print(f"- Partição: {report['s3']['partition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
