from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pandas as pd

from pipelines.loading.upload_to_s3 import upload_processed_files
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
) -> dict[str, object]:
    quantities = {
        "products_quantity": products_quantity,
        "customers_quantity": customers_quantity,
        "orders_quantity": orders_quantity,
    }
    for name, value in quantities.items():
        if value <= 0:
            raise ValueError(f"{name} deve ser maior que zero")

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
        ),
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
        f"- S3: {report['s3']['uploaded_count']} arquivo(s) enviado(s) "
        f"para o bucket {report['s3']['bucket']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
