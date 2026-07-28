import csv
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.extraction.generate_customers import generate_customers
from src.extraction.generate_orders import generate_orders
from src.extraction.generate_products import generate_products

EXPECTED_ORDER_COLUMNS = [
    "order_id",
    "customer_id",
    "order_date",
    "order_status",
    "payment_method",
    "sales_channel",
    "shipping_cost",
    "discount_amount",
    "order_total",
    "delivery_date",
]
EXPECTED_ORDER_ITEM_COLUMNS = [
    "order_item_id",
    "order_id",
    "product_id",
    "quantity",
    "unit_price",
    "unit_cost",
    "discount_percentage",
    "line_total",
]
ALLOWED_ORDER_STATUSES = {"Entregue", "Enviado", "Processando", "Cancelado"}
ALLOWED_PAYMENT_METHODS = {"Cartão de crédito", "Pix", "Boleto", "Carteira digital"}
ALLOWED_SALES_CHANNELS = {"Site", "Aplicativo", "Marketplace"}
MONEY_PATTERN = re.compile(r"^\d+\.\d{2}$")


class SourceData(NamedTuple):
    customers_path: Path
    products_path: Path
    customers: pd.DataFrame
    products: pd.DataFrame


class GeneratedOrders(NamedTuple):
    orders: pd.DataFrame
    order_items: pd.DataFrame
    source: SourceData


@pytest.fixture(scope="module")
def source_data(tmp_path_factory: pytest.TempPathFactory) -> SourceData:
    directory = tmp_path_factory.mktemp("orders_source")
    customers_path = directory / "customers.csv"
    products_path = directory / "products.csv"
    customers = generate_customers(
        quantity=120,
        seed=10,
        output_path=customers_path,
    )
    products = generate_products(
        quantity=50,
        seed=10,
        output_path=products_path,
    )
    return SourceData(customers_path, products_path, customers, products)


@pytest.fixture(scope="module")
def generated_orders(
    tmp_path_factory: pytest.TempPathFactory,
    source_data: SourceData,
) -> GeneratedOrders:
    directory = tmp_path_factory.mktemp("orders_output")
    orders, order_items = generate_orders(
        quantity=600,
        seed=42,
        customers_path=source_data.customers_path,
        products_path=source_data.products_path,
        orders_output_path=directory / "orders.csv",
        order_items_output_path=directory / "order_items.csv",
    )
    return GeneratedOrders(orders, order_items, source_data)


def _generate_in_directory(
    directory: Path,
    source: SourceData,
    quantity: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_orders(
        quantity=quantity,
        seed=seed,
        customers_path=source.customers_path,
        products_path=source.products_path,
        orders_output_path=directory / "orders.csv",
        order_items_output_path=directory / "order_items.csv",
    )


def test_generates_requested_quantity(
    tmp_path: Path,
    source_data: SourceData,
) -> None:
    orders, _ = _generate_in_directory(tmp_path, source_data, quantity=37)

    assert len(orders) == 37


def test_has_exact_expected_columns(generated_orders: GeneratedOrders) -> None:
    assert list(generated_orders.orders.columns) == EXPECTED_ORDER_COLUMNS
    assert list(generated_orders.order_items.columns) == EXPECTED_ORDER_ITEM_COLUMNS


def test_order_ids_are_unique_and_valid(generated_orders: GeneratedOrders) -> None:
    order_ids = generated_orders.orders["order_id"]

    assert order_ids.is_unique
    assert order_ids.str.fullmatch(r"ORD-\d{8}").all()


def test_order_item_ids_are_unique_and_valid(generated_orders: GeneratedOrders) -> None:
    item_ids = generated_orders.order_items["order_item_id"]

    assert item_ids.is_unique
    assert item_ids.str.fullmatch(r"ITEM-\d{8}").all()


def test_orders_reference_existing_customers(generated_orders: GeneratedOrders) -> None:
    customer_ids = set(generated_orders.source.customers["customer_id"])

    assert set(generated_orders.orders["customer_id"]).issubset(customer_ids)


def test_items_reference_existing_orders_and_products(
    generated_orders: GeneratedOrders,
) -> None:
    order_ids = set(generated_orders.orders["order_id"])
    product_ids = set(generated_orders.source.products["product_id"])

    assert set(generated_orders.order_items["order_id"]).issubset(order_ids)
    assert set(generated_orders.order_items["product_id"]).issubset(product_ids)


def test_has_no_unexpected_null_values(generated_orders: GeneratedOrders) -> None:
    required_orders = generated_orders.orders.drop(columns="delivery_date")

    assert not required_orders.isna().any().any()
    assert not generated_orders.order_items.isna().any().any()


def test_categorical_values_are_allowed(generated_orders: GeneratedOrders) -> None:
    orders = generated_orders.orders

    assert set(orders["order_status"]).issubset(ALLOWED_ORDER_STATUSES)
    assert set(orders["payment_method"]).issubset(ALLOWED_PAYMENT_METHODS)
    assert set(orders["sales_channel"]).issubset(ALLOWED_SALES_CHANNELS)


def test_order_dates_are_in_allowed_interval(generated_orders: GeneratedOrders) -> None:
    order_dates = pd.to_datetime(generated_orders.orders["order_date"])

    assert order_dates.between("2023-01-01", "2026-07-28").all()


def test_orders_occur_after_customer_registration(
    generated_orders: GeneratedOrders,
) -> None:
    orders = generated_orders.orders.merge(
        generated_orders.source.customers[["customer_id", "registration_date"]],
        on="customer_id",
        how="left",
    )

    assert (
        pd.to_datetime(orders["order_date"])
        >= pd.to_datetime(orders["registration_date"])
    ).all()


def test_delivery_dates_follow_status_rules(generated_orders: GeneratedOrders) -> None:
    orders = generated_orders.orders
    delivered = orders[orders["order_status"] == "Entregue"]
    sent = orders[orders["order_status"] == "Enviado"]
    without_delivery = orders[orders["order_status"].isin({"Processando", "Cancelado"})]

    assert delivered["delivery_date"].notna().all()
    assert (
        pd.to_datetime(delivered["delivery_date"])
        >= pd.to_datetime(delivered["order_date"])
    ).all()
    assert without_delivery["delivery_date"].isna().all()
    if sent["delivery_date"].notna().any():
        sent_with_delivery = sent.dropna(subset=["delivery_date"])
        assert (
            pd.to_datetime(sent_with_delivery["delivery_date"])
            >= pd.to_datetime(sent_with_delivery["order_date"])
        ).all()


def test_each_order_has_between_one_and_five_items(
    generated_orders: GeneratedOrders,
) -> None:
    item_counts = generated_orders.order_items.groupby("order_id").size()

    assert item_counts.between(1, 5).all()
    assert set(item_counts.index) == set(generated_orders.orders["order_id"])


def test_products_are_not_repeated_within_an_order(
    generated_orders: GeneratedOrders,
) -> None:
    duplicated = generated_orders.order_items.duplicated(["order_id", "product_id"])

    assert not duplicated.any()


def test_item_quantities_are_valid(generated_orders: GeneratedOrders) -> None:
    quantities = generated_orders.order_items["quantity"]

    assert quantities.between(1, 5).all()
    assert pd.api.types.is_integer_dtype(quantities)


def test_item_prices_and_costs_match_products(
    generated_orders: GeneratedOrders,
) -> None:
    items = generated_orders.order_items.merge(
        generated_orders.source.products[["product_id", "unit_price", "unit_cost"]],
        on="product_id",
        suffixes=("_item", "_product"),
    )

    assert items["unit_price_item"].eq(items["unit_price_product"]).all()
    assert items["unit_cost_item"].eq(items["unit_cost_product"]).all()


def test_item_discounts_are_valid(generated_orders: GeneratedOrders) -> None:
    assert generated_orders.order_items["discount_percentage"].between(0, 30).all()


def test_line_totals_are_correct(generated_orders: GeneratedOrders) -> None:
    items = generated_orders.order_items
    expected = items.apply(
        lambda item: float(
            (
                Decimal(int(item["quantity"]))
                * Decimal(str(item["unit_price"]))
                * (Decimal(100) - Decimal(int(item["discount_percentage"])))
                / Decimal(100)
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        ),
        axis=1,
    )

    assert items["line_total"].sub(expected).abs().lt(0.001).all()


def test_order_totals_are_correct(generated_orders: GeneratedOrders) -> None:
    item_totals = (
        generated_orders.order_items.groupby("order_id", as_index=False)["line_total"]
        .sum()
        .rename(columns={"line_total": "items_total"})
    )
    orders = generated_orders.orders.merge(item_totals, on="order_id")
    expected = (
        orders["items_total"] + orders["shipping_cost"] - orders["discount_amount"]
    ).round(2)

    assert orders["order_total"].sub(expected).abs().lt(0.001).all()
    assert orders.loc[orders["order_status"] != "Cancelado", "order_total"].gt(0).all()
    assert orders["shipping_cost"].ge(0).all()
    assert orders["discount_amount"].ge(0).all()


def test_delivered_orders_are_the_majority(generated_orders: GeneratedOrders) -> None:
    delivered_share = generated_orders.orders["order_status"].eq("Entregue").mean()

    assert delivered_share > 0.5


def test_customer_order_distribution_is_varied(
    generated_orders: GeneratedOrders,
) -> None:
    orders_per_customer = generated_orders.orders["customer_id"].value_counts()

    assert orders_per_customer.max() > 1
    assert orders_per_customer.size < len(generated_orders.source.customers)


def test_product_popularity_is_varied(generated_orders: GeneratedOrders) -> None:
    product_frequency = generated_orders.order_items["product_id"].value_counts()

    assert product_frequency.max() > product_frequency.median()


def test_same_seed_produces_same_data(
    tmp_path: Path,
    source_data: SourceData,
) -> None:
    first = _generate_in_directory(tmp_path / "first", source_data, seed=123)
    second = _generate_in_directory(tmp_path / "second", source_data, seed=123)

    assert_frame_equal(first[0], second[0])
    assert_frame_equal(first[1], second[1])


def test_different_seeds_produce_different_data(
    tmp_path: Path,
    source_data: SourceData,
) -> None:
    first = _generate_in_directory(tmp_path / "first", source_data, seed=123)
    second = _generate_in_directory(tmp_path / "second", source_data, seed=456)

    assert not first[0].equals(second[0])
    assert not first[1].equals(second[1])


def test_creates_both_csv_files(
    tmp_path: Path,
    source_data: SourceData,
) -> None:
    orders_path = tmp_path / "nested" / "orders.csv"
    items_path = tmp_path / "other" / "order_items.csv"
    generated_orders, generated_items = generate_orders(
        quantity=25,
        customers_path=source_data.customers_path,
        products_path=source_data.products_path,
        orders_output_path=orders_path,
        order_items_output_path=items_path,
    )
    saved_orders = pd.read_csv(orders_path)
    saved_items = pd.read_csv(items_path)

    assert orders_path.is_file()
    assert items_path.is_file()
    assert "Unnamed: 0" not in saved_orders.columns
    assert "Unnamed: 0" not in saved_items.columns
    assert_frame_equal(saved_orders.fillna(""), generated_orders.fillna(""))
    assert_frame_equal(saved_items, generated_items)
    orders_path.read_text(encoding="utf-8")
    items_path.read_text(encoding="utf-8")


def test_csv_monetary_values_have_two_decimal_places(
    tmp_path: Path,
    source_data: SourceData,
) -> None:
    _generate_in_directory(tmp_path, source_data, quantity=10)

    with (tmp_path / "orders.csv").open(encoding="utf-8", newline="") as orders_file:
        orders = list(csv.DictReader(orders_file))
    with (tmp_path / "order_items.csv").open(
        encoding="utf-8",
        newline="",
    ) as items_file:
        items = list(csv.DictReader(items_file))

    for order in orders:
        assert MONEY_PATTERN.fullmatch(order["shipping_cost"])
        assert MONEY_PATTERN.fullmatch(order["discount_amount"])
        assert MONEY_PATTERN.fullmatch(order["order_total"])
    for item in items:
        assert MONEY_PATTERN.fullmatch(item["unit_price"])
        assert MONEY_PATTERN.fullmatch(item["unit_cost"])
        assert MONEY_PATTERN.fullmatch(item["line_total"])


@pytest.mark.parametrize("quantity", [0, -1])
def test_rejects_non_positive_quantity(
    quantity: int,
    tmp_path: Path,
    source_data: SourceData,
) -> None:
    with pytest.raises(ValueError, match="maior que zero"):
        _generate_in_directory(tmp_path, source_data, quantity=quantity)


@pytest.mark.parametrize("missing_input", ["customers", "products"])
def test_rejects_missing_input_files(
    missing_input: str,
    tmp_path: Path,
    source_data: SourceData,
) -> None:
    customers_path = source_data.customers_path
    products_path = source_data.products_path
    if missing_input == "customers":
        customers_path = tmp_path / "missing_customers.csv"
    else:
        products_path = tmp_path / "missing_products.csv"

    with pytest.raises(FileNotFoundError, match="não encontrado"):
        generate_orders(
            customers_path=customers_path,
            products_path=products_path,
            orders_output_path=tmp_path / "orders.csv",
            order_items_output_path=tmp_path / "order_items.csv",
        )


@pytest.mark.parametrize("empty_input", ["customers", "products"])
def test_rejects_empty_input_files(
    empty_input: str,
    tmp_path: Path,
    source_data: SourceData,
) -> None:
    customers_path = source_data.customers_path
    products_path = source_data.products_path
    if empty_input == "customers":
        customers_path = tmp_path / "empty_customers.csv"
        pd.DataFrame(columns=["customer_id", "registration_date"]).to_csv(
            customers_path,
            index=False,
        )
    else:
        products_path = tmp_path / "empty_products.csv"
        pd.DataFrame(columns=["product_id", "unit_price", "unit_cost"]).to_csv(
            products_path,
            index=False,
        )

    with pytest.raises(ValueError, match="ao menos um registro"):
        generate_orders(
            customers_path=customers_path,
            products_path=products_path,
            orders_output_path=tmp_path / "orders.csv",
            order_items_output_path=tmp_path / "order_items.csv",
        )
