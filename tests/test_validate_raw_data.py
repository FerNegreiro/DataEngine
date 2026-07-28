from pathlib import Path
from shutil import copyfile
from typing import NamedTuple

import pandas as pd
import pytest

import src.validation.validate_raw_data as validator_module
from src.extraction.generate_customers import generate_customers
from src.extraction.generate_orders import generate_orders
from src.extraction.generate_products import generate_products
from src.validation.validate_raw_data import validate_raw_data


class RawFiles(NamedTuple):
    products: Path
    customers: Path
    orders: Path
    order_items: Path


@pytest.fixture(scope="module")
def valid_raw_files(tmp_path_factory: pytest.TempPathFactory) -> RawFiles:
    directory = tmp_path_factory.mktemp("validation_source")
    products = directory / "products.csv"
    customers = directory / "customers.csv"
    orders = directory / "orders.csv"
    order_items = directory / "order_items.csv"
    generate_products(quantity=30, seed=20, output_path=products)
    generate_customers(quantity=60, seed=20, output_path=customers)
    generate_orders(
        quantity=150,
        seed=20,
        customers_path=customers,
        products_path=products,
        orders_output_path=orders,
        order_items_output_path=order_items,
    )
    return RawFiles(products, customers, orders, order_items)


def _copy_raw_files(directory: Path, source: RawFiles) -> RawFiles:
    directory.mkdir(parents=True, exist_ok=True)
    copied = RawFiles(
        directory / "products.csv",
        directory / "customers.csv",
        directory / "orders.csv",
        directory / "order_items.csv",
    )
    for source_path, destination_path in zip(source, copied, strict=True):
        copyfile(source_path, destination_path)
    return copied


def _validate(files: RawFiles) -> dict[str, object]:
    return validate_raw_data(
        products_path=files.products,
        customers_path=files.customers,
        orders_path=files.orders,
        order_items_path=files.order_items,
    )


def _has_error(report: dict[str, object], text: str) -> bool:
    return any(text in error for error in report["errors"])


def test_valid_files_pass_validation(valid_raw_files: RawFiles) -> None:
    report = _validate(valid_raw_files)

    assert report["is_valid"] is True
    assert report["errors"] == []


def test_missing_file_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = RawFiles(
        tmp_path / "missing_products.csv",
        valid_raw_files.customers,
        valid_raw_files.orders,
        valid_raw_files.order_items,
    )
    report = _validate(files)

    assert report["is_valid"] is False
    assert _has_error(report, "não encontrado")


def test_empty_file_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    files.products.write_text("", encoding="utf-8")
    report = _validate(files)

    assert _has_error(report, "está vazio")


def test_missing_required_column_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    products = pd.read_csv(files.products).drop(columns="unit_price")
    products.to_csv(files.products, index=False)
    report = _validate(files)

    assert _has_error(report, "Colunas obrigatórias ausentes")


def test_duplicate_primary_key_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    products = pd.read_csv(files.products)
    products.loc[1, "product_id"] = products.loc[0, "product_id"]
    products.to_csv(files.products, index=False)
    report = _validate(files)

    assert _has_error(report, "Chave primária duplicada")


def test_invalid_customer_reference_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    orders = pd.read_csv(files.orders)
    orders.loc[0, "customer_id"] = "CUST-999999"
    orders.to_csv(files.orders, index=False)
    report = _validate(files)

    assert _has_error(report, "orders.customer_id possui referências inexistentes")


def test_invalid_order_reference_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    items = pd.read_csv(files.order_items)
    items.loc[0, "order_id"] = "ORD-99999999"
    items.to_csv(files.order_items, index=False)
    report = _validate(files)

    assert _has_error(report, "order_items.order_id possui referências inexistentes")


def test_invalid_product_reference_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    items = pd.read_csv(files.order_items)
    items.loc[0, "product_id"] = "PROD-999999"
    items.to_csv(files.order_items, index=False)
    report = _validate(files)

    assert _has_error(report, "order_items.product_id possui referências inexistentes")


def test_product_cost_not_lower_than_price_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    products = pd.read_csv(files.products)
    products.loc[0, "unit_cost"] = products.loc[0, "unit_price"]
    products.to_csv(files.products, index=False)
    report = _validate(files)

    assert _has_error(report, "unit_cost deve ser menor")


def test_incoherent_customer_state_and_region_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    customers = pd.read_csv(files.customers)
    customers.loc[0, ["state", "region"]] = ["SP", "Norte"]
    customers.to_csv(files.customers, index=False)
    report = _validate(files)

    assert _has_error(report, "incoerência entre UF e região")


def test_order_before_customer_registration_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    orders = pd.read_csv(files.orders)
    customers = pd.read_csv(files.customers)
    customer_id = orders.loc[0, "customer_id"]
    customers.loc[customers["customer_id"] == customer_id, "registration_date"] = "2025-01-01"
    orders.loc[0, "order_date"] = "2024-01-01"
    orders.to_csv(files.orders, index=False)
    customers.to_csv(files.customers, index=False)
    report = _validate(files)

    assert _has_error(report, "pedido anterior ao cadastro")


def test_item_price_different_from_product_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    items = pd.read_csv(files.order_items)
    items.loc[0, "unit_price"] += 1
    items.to_csv(files.order_items, index=False)
    report = _validate(files)

    assert _has_error(report, "Preço do item difere")


def test_incorrect_line_total_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    items = pd.read_csv(files.order_items)
    items.loc[0, "line_total"] += 1
    items.to_csv(files.order_items, index=False)
    report = _validate(files)

    assert _has_error(report, "line_total possui cálculo incorreto")


def test_incorrect_order_total_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    orders = pd.read_csv(files.orders)
    orders.loc[0, "order_total"] += 1
    orders.to_csv(files.orders, index=False)
    report = _validate(files)

    assert _has_error(report, "order_total não reconcilia")


def test_repeated_product_in_order_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    items = pd.read_csv(files.order_items)
    order_id = items.groupby("order_id").size().loc[lambda counts: counts >= 2].index[0]
    order_indexes = items.index[items["order_id"] == order_id]
    items.loc[order_indexes[1], "product_id"] = items.loc[order_indexes[0], "product_id"]
    items.to_csv(files.order_items, index=False)
    report = _validate(files)

    assert _has_error(report, "Produto repetido")


def test_invalid_item_count_is_reported(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    orders = pd.read_csv(files.orders)
    items = pd.read_csv(files.order_items)
    items = items[items["order_id"] != orders.loc[0, "order_id"]]
    items.to_csv(files.order_items, index=False)
    report = _validate(files)

    assert _has_error(report, "entre 1 e 5 itens")


def test_multiple_errors_are_accumulated(
    tmp_path: Path,
    valid_raw_files: RawFiles,
) -> None:
    files = _copy_raw_files(tmp_path, valid_raw_files)
    products = pd.read_csv(files.products)
    orders = pd.read_csv(files.orders)
    products.loc[0, "unit_price"] = -1
    orders.loc[0, "customer_id"] = "CUST-999999"
    products.to_csv(files.products, index=False)
    orders.to_csv(files.orders, index=False)
    report = _validate(files)

    assert report["is_valid"] is False
    assert len(report["errors"]) >= 2
    assert _has_error(report, "unit_price deve ser maior")
    assert _has_error(report, "customer_id possui referências inexistentes")


def test_report_has_exact_structure(valid_raw_files: RawFiles) -> None:
    report = _validate(valid_raw_files)

    assert list(report) == ["is_valid", "errors", "warnings", "summary"]
    assert list(report["summary"]) == [
        "products_rows",
        "customers_rows",
        "orders_rows",
        "order_items_rows",
    ]
    assert isinstance(report["is_valid"], bool)
    assert isinstance(report["errors"], list)
    assert isinstance(report["warnings"], list)


def test_direct_execution_returns_zero_for_valid_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "summary": {
            "products_rows": 1,
            "customers_rows": 1,
            "orders_rows": 1,
            "order_items_rows": 1,
        },
    }
    monkeypatch.setattr(validator_module, "validate_raw_data", lambda: report)

    assert validator_module.main() == 0
    assert "Resultado: dados válidos" in capsys.readouterr().out


def test_direct_execution_returns_one_for_invalid_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "is_valid": False,
        "errors": ["erro simulado"],
        "warnings": [],
        "summary": {
            "products_rows": 0,
            "customers_rows": 0,
            "orders_rows": 0,
            "order_items_rows": 0,
        },
    }
    monkeypatch.setattr(validator_module, "validate_raw_data", lambda: report)

    assert validator_module.main() == 1
    output = capsys.readouterr().out
    assert "erro simulado" in output
    assert "Resultado: dados inválidos" in output
