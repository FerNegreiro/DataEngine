from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.extraction.generate_products import generate_products

EXPECTED_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "brand",
    "unit_price",
    "unit_cost",
    "stock_quantity",
    "minimum_stock",
    "supplier",
    "created_at",
    "is_active",
]

ALLOWED_CATEGORIES = {
    "Eletrônicos",
    "Casa e Decoração",
    "Moda",
    "Esporte",
    "Beleza",
    "Livros",
    "Brinquedos",
    "Alimentos",
}


@pytest.fixture
def products(tmp_path: Path) -> pd.DataFrame:
    return generate_products(quantity=50, seed=42, output_path=tmp_path / "products.csv")


def test_generates_requested_quantity(tmp_path: Path) -> None:
    products = generate_products(quantity=17, output_path=tmp_path / "products.csv")

    assert len(products) == 17


def test_has_exact_expected_columns(products: pd.DataFrame) -> None:
    assert list(products.columns) == EXPECTED_COLUMNS


def test_product_ids_are_unique(products: pd.DataFrame) -> None:
    assert products["product_id"].is_unique


def test_product_ids_follow_expected_format(products: pd.DataFrame) -> None:
    assert products["product_id"].str.fullmatch(r"PROD-\d{6}").all()


def test_generated_data_has_no_null_values(products: pd.DataFrame) -> None:
    assert not products.isna().any().any()
    assert products["product_name"].str.strip().ne("").all()


def test_prices_and_costs_are_positive(products: pd.DataFrame) -> None:
    assert products["unit_price"].gt(0).all()
    assert products["unit_cost"].gt(0).all()


def test_unit_cost_is_lower_than_unit_price(products: pd.DataFrame) -> None:
    assert products["unit_cost"].lt(products["unit_price"]).all()


def test_stock_limits(products: pd.DataFrame) -> None:
    assert products["stock_quantity"].between(0, 500).all()
    assert products["minimum_stock"].between(5, 50).all()
    assert pd.api.types.is_integer_dtype(products["stock_quantity"])
    assert pd.api.types.is_integer_dtype(products["minimum_stock"])


def test_categories_are_allowed(products: pd.DataFrame) -> None:
    assert set(products["category"]).issubset(ALLOWED_CATEGORIES)


def test_dates_and_active_values_are_valid(products: pd.DataFrame) -> None:
    created_at = pd.to_datetime(products["created_at"])

    assert created_at.between("2022-01-01", "2026-07-28").all()
    assert pd.api.types.is_bool_dtype(products["is_active"])


def test_same_seed_produces_same_data(tmp_path: Path) -> None:
    first = generate_products(seed=123, output_path=tmp_path / "first.csv")
    second = generate_products(seed=123, output_path=tmp_path / "second.csv")

    assert_frame_equal(first, second)


def test_different_seeds_produce_different_data(tmp_path: Path) -> None:
    first = generate_products(seed=123, output_path=tmp_path / "first.csv")
    second = generate_products(seed=456, output_path=tmp_path / "second.csv")

    assert not first.equals(second)


def test_creates_csv_file(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "products.csv"
    generated = generate_products(quantity=5, output_path=output_path)

    assert output_path.is_file()
    assert_frame_equal(pd.read_csv(output_path), generated)


def test_csv_has_two_decimal_places_and_no_index(tmp_path: Path) -> None:
    output_path = tmp_path / "products.csv"
    generate_products(quantity=5, output_path=output_path)
    csv_lines = output_path.read_text(encoding="utf-8").splitlines()
    header = csv_lines[0].split(",")

    price_index = header.index("unit_price")
    cost_index = header.index("unit_cost")
    assert "Unnamed: 0" not in header
    for line in csv_lines[1:]:
        fields = line.split(",")
        assert fields[price_index].split(".")[-1].isdigit()
        assert len(fields[price_index].split(".")[-1]) == 2
        assert fields[cost_index].split(".")[-1].isdigit()
        assert len(fields[cost_index].split(".")[-1]) == 2


@pytest.mark.parametrize("quantity", [0, -1])
def test_rejects_non_positive_quantity(quantity: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="maior que zero"):
        generate_products(quantity=quantity, output_path=tmp_path / "products.csv")
